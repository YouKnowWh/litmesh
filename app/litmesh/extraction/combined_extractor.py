"""
Combined extractor: claims + evidence + limitations in a single LLM call.

Replaces three separate per-section LLM calls with one, reducing total API calls
from ~3N to ~N. Shorter prompt + lower max_tokens targets 5-10s per call.
"""

import json
from ..models.claim import ClaimBlock, ClaimType, ClaimImportance
from ..models.evidence import EvidenceBlock, EvidenceType, EvidenceStrength
from ..models.limitation import LimitationBlock, RiskType, LimitationSeverity


COMBINED_PROMPT = """从以下章节同时提取主张、证据、局限性。返回紧凑JSON（不要换行、不要空格、不要markdown标记）：

{
"c":[{"t":"主张原文","n":"标准化表述","ct":"definitional|empirical|theoretical|methodological|normative|critical|synthesis|framework","im":"core|supporting|peripheral","cf":0.8,"k":["术语"]}],
"e":[{"t":"证据原文","et":"case|data|experiment|teaching_practice|survey|theoretical_reference|policy_reference|chapter_argument|literature_citation|other","st":"strong|moderate|weak|unassessed","k":["术语"]}],
"l":[{"t":"局限原文","rt":"method|scope|data|theory|generalizability|other","sv":"high|medium|low|unassessed","k":["术语"]}]
}

c=claims(主张), e=evidence(证据), l=limitations(局限性)
ct=claim_type, im=importance, cf=confidence(0-1), et=evidence_type, st=strength
rt=risk_type, sv=severity
k=concept_terms(短术语,不要长句,不要普通词如"教育""学生")

主张判断：作者明确提出的观点/结论/定义/判断，非事实陈述或背景介绍
证据判断：支持主张的案例/数据/实验/引用
局限判断：主张的适用范围限制、方法缺陷、数据不足等
cf>0.7=原文明确, 0.5-0.7=需推理, <0.5=不确定
无内容时返回空数组。只返回JSON。

章节：{heading}
文本：{text}"""


class CombinedExtractor:
    """Extracts claims, evidence, and limitations in one LLM call per section."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def extract_from_section(self, section, extraction_run_id: str) -> dict:
        """Returns {"claims": [...], "evidence": [...], "limitations": [...]}."""
        prompt = (COMBINED_PROMPT
            .replace("{heading}", section.heading)
            .replace("{text}", section.raw_text[:2000]))

        system = "You are a precise academic extractor. Output only compact JSON, no whitespace, no markdown. Keep responses under 800 tokens."
        raw = self.llm.complete(prompt, system=system, temperature=0.1, max_tokens=1024)
        data = self._parse_json(raw)

        claims = []
        evidence = []
        limitations = []

        for item in data.get("c", []):
            if not item.get("t"):
                continue
            claims.append(ClaimBlock(
                graph_id=section.graph_id, paper_id=section.paper_id,
                section_id=section.section_id,
                claim_text=item["t"],
                normalized_claim=item.get("n", item["t"]),
                claim_type=_parse_claim_type(item.get("ct")),
                concept_keys=_clean_terms(item.get("k", [])),
                extraction_confidence=float(item.get("cf", 0.5)),
                importance=_parse_importance(item.get("im")),
                extraction_run_id=extraction_run_id,
            ))

        for item in data.get("e", []):
            if not item.get("t"):
                continue
            evidence.append(EvidenceBlock(
                graph_id=section.graph_id, paper_id=section.paper_id,
                section_id=section.section_id,
                evidence_text=item["t"],
                evidence_type=_parse_ev_type(item.get("et")),
                strength=_parse_strength(item.get("st")),
                concept_keys=_clean_terms(item.get("k", [])),
                extraction_run_id=extraction_run_id,
            ))

        for item in data.get("l", []):
            if not item.get("t"):
                continue
            limitations.append(LimitationBlock(
                graph_id=section.graph_id, paper_id=section.paper_id,
                section_id=section.section_id,
                limitation_text=item["t"],
                risk_type=_parse_risk(item.get("rt")),
                severity=_parse_severity(item.get("sv")),
                concept_keys=_clean_terms(item.get("k", [])),
                extraction_run_id=extraction_run_id,
            ))

        return {"claims": claims, "evidence": evidence, "limitations": limitations}

    def _parse_json(self, raw: str) -> dict:
        cleaned = raw.strip()
        # Strip markdown fences
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return {"c": [], "e": [], "l": []}


def _parse_claim_type(v) -> ClaimType:
    if not v:
        return ClaimType.THEORETICAL
    mapping = {t.value: t for t in ClaimType}
    return mapping.get(v, ClaimType.THEORETICAL)


def _parse_importance(v) -> ClaimImportance:
    if not v:
        return ClaimImportance.SUPPORTING
    mapping = {t.value: t for t in ClaimImportance}
    return mapping.get(v, ClaimImportance.SUPPORTING)


def _parse_ev_type(v) -> EvidenceType:
    if not v:
        return EvidenceType.OTHER
    mapping = {t.value: t for t in EvidenceType}
    return mapping.get(v, EvidenceType.OTHER)


def _parse_strength(v) -> EvidenceStrength:
    if not v:
        return EvidenceStrength.UNASSESSED
    mapping = {t.value: t for t in EvidenceStrength}
    return mapping.get(v, EvidenceStrength.UNASSESSED)


def _parse_risk(v) -> RiskType:
    if not v:
        return RiskType.OTHER
    mapping = {t.value: t for t in RiskType}
    return mapping.get(v, RiskType.OTHER)


def _parse_severity(v) -> LimitationSeverity:
    if not v:
        return LimitationSeverity.UNASSESSED
    mapping = {t.value: t for t in LimitationSeverity}
    return mapping.get(v, LimitationSeverity.UNASSESSED)


_GENERIC = {"教育", "教学", "学生", "教师", "学习", "研究", "实践", "问题", "方法", "技术"}


def _clean_terms(raw) -> list[str]:
    if not isinstance(raw, list):
        return []
    cleaned = []
    for term in raw:
        if not isinstance(term, str):
            continue
        v = term.strip()
        if not v or v in _GENERIC or len(v) > 32:
            continue
        if ":" in v:
            v = v.split(":", 1)[1].strip()
        if v and v not in cleaned:
            cleaned.append(v)
    return cleaned
