"""
LimitationBlock extraction: SectionBlock + ClaimBlock -> LimitationBlock candidates.

Extracts limitations, risks, challenges, and unsolved problems.
Limitations CONSTRAINT claims; they do NOT contradict them (that's a separate relation type).
"""

import json
from ..models.limitation import LimitationBlock, RiskType, LimitationSeverity
from ..ingestion.source_span import make_span_for_claim


LIMITATION_EXTRACTION_PROMPT = """你是一个学术限制与风险分析器。请从以下论文章节中提取对给定主张的限制、风险、边界条件和未解决问题。

限制（limitation）不是对主张的否认，而是对主张适用范围、条件、可靠性的约束。

以 JSON 数组格式返回，不要输出其他内容：

[
  {
    "limitation_text": "限制原文表述",
    "affected_claim_text": "受此限制影响的主张（从给定主张中选择，或留空表示影响所有主张）",
    "risk_type": "scope|method|data|ethics|implementation|theoretical|unexplored|conflict|other",
    "severity": "critical|moderate|minor|unassessed",
    "concept_keys": ["相关概念"]
  }
]

risk_type 判断：
- scope: 适用范围有限，不能推广到其他场景
- method: 方法论上的弱点（样本小、缺乏对照等）
- data: 数据质量问题
- ethics: 伦理风险
- implementation: 实际落地挑战
- theoretical: 理论假设或空白
- unexplored: 作者承认没研究的方面
- conflict: 与其他研究矛盾

severity 判断：
- critical: 严重影响主张的可信度
- moderate: 需要限定但总体成立
- minor: 基本不影响

只返回 JSON 数组。如果该章节没有明确限制，返回 []。

章节标题：{heading}
相关主张：
{claims_text}

章节文本：
{text}
"""


class LimitationExtractor:
    """Extracts LimitationBlock candidates."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_for_claims(
        self,
        section,
        claims: list,
        extraction_run_id: str,
    ) -> list[LimitationBlock]:
        if not claims:
            return []

        claims_text = "\n".join(
            f"- [{c.claim_id}] {c.claim_text}" for c in claims
        )

        prompt = LIMITATION_EXTRACTION_PROMPT.format(
            heading=section.heading,
            claims_text=claims_text,
            text=section.raw_text[:4000],
        )

        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        blocks = []
        for item in items:
            if not item.get("limitation_text"):
                continue

            # Match affected claim
            affected = item.get("affected_claim_text", "")
            affected_ids = []
            for c in claims:
                if affected and (affected in c.claim_text or c.claim_text in affected):
                    affected_ids.append(c.claim_id)
            if not affected_ids:
                affected_ids = [c.claim_id for c in claims]

            span = make_span_for_claim(
                paper_id=section.paper_id,
                section_id=section.section_id,
                claim_text=item["limitation_text"],
                full_section_text=section.raw_text,
                page_start=section.page_start or 1,
            )

            limitation = LimitationBlock(
                graph_id=section.graph_id,
                paper_id=section.paper_id,
                section_id=section.section_id,
                limitation_text=item["limitation_text"],
                affected_claim_ids=affected_ids,
                risk_type=RiskType(item.get("risk_type", "other")),
                severity=LimitationSeverity(item.get("severity", "unassessed")),
                concept_keys=item.get("concept_keys", []),
                source_span_id=span.span_id,
                extraction_run_id=extraction_run_id,
            )
            blocks.append(limitation)

        return blocks

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
