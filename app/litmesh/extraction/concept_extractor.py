"""
ConceptKey extraction: ClaimBlock -> ConceptKey candidates.

Extracts concept terms from claims and proposes candidate ConceptKeys.
Does NOT create active concepts directly. All candidates go to ConceptInbox.

Principle: LLM proposes, ConceptRegistry validates, human reviews.
"""

import json
from ..models.concept import ConceptKey, ConceptNamespace, MergePolicy


CONCEPT_EXTRACTION_PROMPT = """你是一个学术概念提取器。请从以下论文主张列表中提取核心概念。

对每个概念，以 JSON 数组格式返回：

[
  {
    "concept_term": "概念术语（中文）",
    "english_term": "概念术语（英文，如果可以翻译）",
    "namespace": "concept|framework|method|problem|theory|tool|metric",
    "definition": "一句话定义",
    "aliases": ["同义词1", "同义词2"],
    "parent_concept": "上位概念（如果有，留空表示无）"
  }
]

namespace 判断标准：
- concept: 一般概念（如 "认知负荷"、"人机共创"）
- framework: 命名框架或模型（如 "PACADI"、"CPE-3DF"）
- method: 方法或技术（如 "提示工程"、"虚拟实验"）
- problem: 问题或挑战（如 "AI 幻觉"、"数据隐私"）
- theory: 理论基础（如 "陶行知生活教育思想"）
- tool: 工具或技术产品（如 "ChatGPT"）
- metric: 度量或指标（如 "学习效果"）

注意：
- 不要提取过于通用的词（如 "教育"、"教学"、"学生"）
- 不要创造论文中没有的概念
- 同义词应该是在论文中实际出现的不同表述
- 每个概念必须是独立的、有边界的、可以定义的单元
- concept_term 必须是短术语，不是完整句子或章节标题
- 如果原文术语只是长句，请概括为一个稳定概念候选，并把原文长句放入 aliases

只返回 JSON 数组。

主张列表：
{claims_text}
"""


class ConceptExtractor:
    """Extracts ConceptKey candidates from claims."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_from_claims(
        self,
        claims: list,
        graph_id: str,
        extraction_run_id: str,
    ) -> list[ConceptKey]:
        """Extract concept candidates from a list of claims."""
        if not claims:
            return []

        claims_text = "\n".join(
            f"- [{c.claim_type.value}] {c.normalized_claim or c.claim_text}"
            for c in claims
        )

        prompt = CONCEPT_EXTRACTION_PROMPT.replace("{claims_text}", claims_text[:5000])
        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        concepts = []
        for item in items:
            term = _clean_concept_term(item.get("concept_term", ""))
            if not term:
                continue

            namespace = ConceptNamespace(item.get("namespace", "concept"))
            eng = _clean_concept_term(item.get("english_term", ""), allow_english=True)
            key_slug = _canonical_slug(namespace, term, eng)
            concept_key = f"{namespace.value}:{key_slug}"

            concept = ConceptKey(
                concept_key=concept_key,
                graph_id=graph_id,
                namespace=namespace,
                label_zh=term,
                label_en=eng,
                definition=item.get("definition", ""),
                aliases=_clean_aliases(item.get("aliases", []), term, eng),
                parent_keys=_clean_parent(item.get("parent_concept")),
                merge_policy=MergePolicy.STRICT,
                extraction_run_id=extraction_run_id,
            )
            concepts.append(concept)

        return concepts

    def _parse_response(self, raw: str) -> list:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []


def _slugify(text: str) -> str:
    """Simple slug for Chinese terms: use pinyin-friendly ASCII fallback."""
    # For MVP, just use a hash-based fallback for pure Chinese terms
    import hashlib
    if all('一' <= c <= '鿿' or c.isspace() for c in text if c.isalpha() or c.isspace()):
        return hashlib.md5(text.encode()).hexdigest()[:8]
    return text.lower().replace(" ", "_").replace("-", "_")


_GENERIC_TERMS = {
    "教育", "教学", "学生", "教师", "学习", "研究", "问题", "方法", "技术",
    "框架", "模型", "实践", "应用", "影响", "作用", "能力", "课程",
}

_KNOWN_SLUGS = {
    "认知负荷": "cognitive_load",
    "人机共创": "human_ai_cocreation",
    "AI幻觉": "AI_hallucination",
    "大模型幻觉": "LLM_hallucination",
    "思维外包": "cognitive_outsourcing",
    "数据隐私": "data_privacy",
    "虚拟实验": "virtual_lab",
    "提示工程": "prompt_engineering",
    "陶行知生活教育思想": "Tao_Xingzhi_life_education",
    "生活教育思想": "Tao_Xingzhi_life_education",
    "学术诚信": "academic_integrity",
    "算法偏见": "algorithmic_bias",
}


def _clean_concept_term(term: str, allow_english: bool = False) -> str:
    """Reject copied sentences and overly generic labels before key generation."""
    if not isinstance(term, str):
        return ""
    cleaned = " ".join(term.strip().split())
    cleaned = cleaned.strip("：:，,。.;；、")
    if not cleaned or cleaned in _GENERIC_TERMS:
        return ""
    sentence_marks = ["。", "；", "，", "认为", "提出", "表明", "必须", "可以", "应该"]
    if any(mark in cleaned for mark in sentence_marks):
        return ""
    if len(cleaned) > 24 and not allow_english:
        return ""
    if allow_english and len(cleaned.split()) > 6:
        return ""
    return cleaned


def _clean_aliases(aliases, label_zh: str, label_en: str) -> list[str]:
    if not isinstance(aliases, list):
        aliases = []
    result = []
    for alias in aliases + [label_zh, label_en]:
        if not isinstance(alias, str):
            continue
        cleaned = " ".join(alias.strip().split()).strip("：:，,。.;；、")
        if not cleaned or cleaned in result:
            continue
        if len(cleaned) > 48:
            continue
        result.append(cleaned)
    return result


def _clean_parent(parent: str) -> list[str]:
    cleaned = _clean_concept_term(parent)
    return [cleaned] if cleaned else []


def _canonical_slug(namespace: ConceptNamespace, term: str, english_term: str) -> str:
    """Generate stable routing keys from controlled terms, not copied sentences."""
    if namespace == ConceptNamespace.FRAMEWORK:
        framework = term.upper().replace(" ", "_").replace("-", "_")
        if len(framework) <= 24 and all(c.isalnum() or c == "_" for c in framework):
            return framework
    if term in _KNOWN_SLUGS:
        return _KNOWN_SLUGS[term]
    if english_term:
        return _ascii_slug(english_term)
    return _slugify(term)


def _ascii_slug(text: str) -> str:
    import re
    slug = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return slug[:64] or _slugify(text)
