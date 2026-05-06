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

        prompt = CONCEPT_EXTRACTION_PROMPT.format(claims_text=claims_text[:5000])
        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        concepts = []
        for item in items:
            term = item.get("concept_term", "")
            if not term:
                continue

            namespace = ConceptNamespace(item.get("namespace", "concept"))
            # Generate concept_key: namespace:english_term or namespace:chinese_pinyin
            eng = item.get("english_term", "")
            key_slug = eng.lower().replace(" ", "_").replace("-", "_") if eng else _slugify(term)
            concept_key = f"{namespace.value}:{key_slug}"

            concept = ConceptKey(
                concept_key=concept_key,
                graph_id=graph_id,
                namespace=namespace,
                label_zh=term,
                label_en=eng,
                definition=item.get("definition", ""),
                aliases=item.get("aliases", []),
                parent_keys=[item.get("parent_concept")] if item.get("parent_concept") else [],
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
