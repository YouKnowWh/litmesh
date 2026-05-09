"""
ClaimBlock extraction: SectionBlock -> ClaimBlock candidates.

Extracts the author's core claims/assertions from each section.
Claims are typed (definitional, empirical, theoretical, etc.) and
dual-confidenced (extraction_confidence vs claim_confidence).

Principle 4: LLM produces candidates, program controls writes.
Principle 5: No source_span, no active claim.
"""

import json
from typing import Optional

from ..models.claim import ClaimBlock, ClaimType, ClaimImportance
from ..models.source_span import SpanType
from ..ingestion.source_span import make_span_for_claim


CLAIM_EXTRACTION_PROMPT = """你是一个学术论证提取器。请从以下论文章节中提取核心主张（claims）。

一个主张（claim）是作者明确提出的观点、结论、定义或判断。不是事实陈述，不是背景介绍，不是文献引用。

对于每个主张，请以如下 JSON 数组格式返回，不要输出其他内容：

[
  {
    "claim_text": "作者的原话或近原话表述",
    "normalized_claim": "独立的、代词已解析的完整表述（一句完整的话）",
    "claim_type": "definitional|empirical|theoretical|methodological|normative|critical|synthesis|framework",
    "importance": "core|supporting|peripheral",
    "extraction_confidence": 0.0-1.0,
    "concept_terms": ["原文中的候选概念术语，不要生成 ConceptKey，不要写长句"]
  }
]

claim_type 判断标准：
- definitional: "X 被定义为 Y"、"X 是指 Y"
- empirical: "我们观察到 X"、"实验表明 X"、"数据显示 X"
- theoretical: 基于理论推理的主张，"因为理论 Z，所以 X 应该导致 Y"
- methodological: 关于方法的主张，"方法 X 比 Y 更好"、"应该用 X 方法"
- normative: 价值判断，"应该做 X"、"X 是重要的"
- critical: 对已有研究的批评，"前人工作 X 有缺陷"
- synthesis: 综合他人工作提出新观点
- framework: 提出新框架、模型、分类体系

importance 判断标准：
- core: 论文的核心论点，不读这篇论文就不知道这个观点
- supporting: 重要但不是核心
- peripheral: 有趣但可跳过

extraction_confidence 判断标准：
- 1.0: 原文明确表述，无歧义
- 0.7-0.9: 需要轻微推理，但基本确定
- 0.5-0.7: 需要一定推理和解释
- <0.5: 高度不确定，可能过度解释

concept_terms 只记录可作为概念候选的短术语，例如 "认知负荷"、"PACADI"、"AI幻觉"。
不要输出 concept:xxx/framework:xxx 这种正式 ConceptKey；正式 key 由程序和审核流程生成。
不要把完整句子、章节标题、普通词（教育、教学、学生、研究）当作概念。

只返回 JSON 数组，不要输出解释或其他文字。如果该章节没有明确的主张，返回空数组 []。

章节标题：{heading}
章节路径：{heading_path}

章节文本：
{text}
"""


class ClaimExtractor:
    """Extracts ClaimBlock candidates from SectionBlock text."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_from_section(
        self,
        section,
        extraction_run_id: str,
        previous_claims: Optional[list] = None,
    ) -> list[ClaimBlock]:
        """Extract claims from a single section.

        Args:
            section: SectionBlock with raw_text.
            extraction_run_id: FK to the ExtractionRun.
            previous_claims: Claims from earlier sections (for dedup context, not yet used).
        """
        prompt = (CLAIM_EXTRACTION_PROMPT
            .replace("{heading}", section.heading)
            .replace("{heading_path}", " > ".join(section.heading_path))
            .replace("{text}", section.raw_text[:4000]))

        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        claims = []
        for item in items:
            # Validate required fields
            if not item.get("claim_text"):
                continue

            confidence = max(0.0, min(1.0, float(item.get("extraction_confidence", 0.5))))

            claim = ClaimBlock(
                graph_id=section.graph_id,
                paper_id=section.paper_id,
                section_id=section.section_id,
                claim_text=item["claim_text"],
                normalized_claim=item.get("normalized_claim", item["claim_text"]),
                claim_type=ClaimType(item.get("claim_type", "theoretical")),
                concept_keys=_candidate_terms(item),
                extraction_confidence=confidence,
                importance=ClaimImportance(item.get("importance", "supporting")),
                source_span_id=None,  # Span insertion deferred to pipeline
                extraction_run_id=extraction_run_id,
            )
            claims.append(claim)

        return claims

    def _parse_response(self, raw: str) -> list:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            # Sometimes LLM returns {"claims": [...]}
            if isinstance(parsed, dict):
                return parsed.get("claims", [])
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []
        return []


def _candidate_terms(item: dict) -> list[str]:
    """Keep LLM output as raw candidate terms, not canonical ConceptKeys."""
    raw_terms = item.get("concept_terms")
    if raw_terms is None:
        raw_terms = item.get("concept_keys", [])
    return _clean_terms(raw_terms)


def _clean_terms(raw_terms) -> list[str]:
    if not isinstance(raw_terms, list):
        return []
    generic = {"教育", "教学", "学生", "教师", "学习", "研究", "实践", "问题", "方法", "技术"}
    cleaned = []
    for term in raw_terms:
        if not isinstance(term, str):
            continue
        value = term.strip()
        if not value or value in generic:
            continue
        if ":" in value:
            value = value.split(":", 1)[1].strip()
        if len(value) > 32 or len(value.split()) > 6:
            continue
        if value not in cleaned:
            cleaned.append(value)
    return cleaned
