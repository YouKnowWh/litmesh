"""
EvidenceBlock extraction: SectionBlock + ClaimBlock -> EvidenceBlock candidates.

Extracts what evidence the author uses to support each claim.
"""

import json
from ..models.evidence import EvidenceBlock, EvidenceType, EvidenceStrength
from ..ingestion.source_span import make_span_for_claim
from .claim_extractor import _candidate_terms


EVIDENCE_EXTRACTION_PROMPT = """你是一个学术证据提取器。请从以下论文章节中提取支持给定主张的证据。

对于每个证据，以 JSON 数组格式返回，不要输出其他内容：

[
  {
    "evidence_text": "证据原文",
    "evidence_type": "case|data|experiment|teaching_practice|survey|theoretical_reference|policy_reference|chapter_argument|literature_citation|other",
    "strength": "strong|moderate|weak|unassessed",
    "concept_terms": ["原文中的候选概念术语，不要生成 ConceptKey，不要写长句"]
  }
]

evidence_type 判断：
- case: 案例研究、具体例子
- data: 量化数据、统计数字
- experiment: 实验、对照实验
- teaching_practice: 教学观察、教学反思
- survey: 问卷调查
- theoretical_reference: 引用其他理论
- policy_reference: 引用政策文件
- chapter_argument: 教科书式的逻辑论证
- literature_citation: 引用前人文献

strength 判断：
- strong: 严格的实验数据、大样本统计
- moderate: 合理但有限度的证据
- weak: 个例、传闻、逻辑推理

concept_terms 只记录短术语，例如 "认知负荷"、"虚拟实验"、"AI幻觉"。
不要输出 concept:xxx/framework:xxx 这种正式 ConceptKey；正式 key 由程序和审核流程生成。
不要把完整句子、章节标题、普通词（教育、教学、学生、研究）当作概念。

只返回 JSON 数组。如果没有明确证据，返回 []。

章节标题：{heading}
主张（需要找证据的主张）：
{claims_text}

章节文本：
{text}
"""


class EvidenceExtractor:
    """Extracts EvidenceBlock candidates for given claims."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_for_claims(
        self,
        section,
        claims: list,
        extraction_run_id: str,
    ) -> list[EvidenceBlock]:
        """Extract evidence for a set of claims within a section."""
        if not claims:
            return []

        claims_text = "\n".join(
            f"- [{c.claim_id}] {c.claim_text}" for c in claims
        )

        prompt = (EVIDENCE_EXTRACTION_PROMPT
            .replace("{heading}", section.heading)
            .replace("{claims_text}", claims_text)
            .replace("{text}", section.raw_text[:4000]))

        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        evidence_blocks = []
        for item in items:
            if not item.get("evidence_text"):
                continue

            # Map evidence to claims
            supports_claim_ids = [c.claim_id for c in claims]

            evidence = EvidenceBlock(
                graph_id=section.graph_id,
                paper_id=section.paper_id,
                section_id=section.section_id,
                supports_claim_ids=supports_claim_ids,
                evidence_text=item["evidence_text"],
                evidence_type=EvidenceType(item.get("evidence_type", "other")),
                strength=EvidenceStrength(item.get("strength", "unassessed")),
                concept_keys=_candidate_terms(item),
                source_span_id=None,
                extraction_run_id=extraction_run_id,
            )
            evidence_blocks.append(evidence)

        return evidence_blocks

    def _parse_response(self, raw: str) -> list:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            parsed = json.loads(cleaned)
            return parsed if isinstance(parsed, list) else parsed.get("evidence", [])
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []
