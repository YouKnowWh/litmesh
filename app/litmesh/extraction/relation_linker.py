"""
Relation linker: connects ClaimBlocks to EvidenceBlocks and LimitationBlocks.

Creates GraphRelations:
- evidence -> claim: SUPPORTS
- limitation -> claim: CONSTRAINS

Also creates structural relations:
- claim -> section: BELONGS_TO
- claim -> concept: MENTIONS
"""

import json
from ..models.relation import GraphRelation, GraphRelationType


RELATION_LINKING_PROMPT = """你是一个学术论证关系分析器。请分析以下主张之间的关系。

对于每对关系，以 JSON 数组格式返回：

[
  {
    "source_index": 0,
    "target_index": 1,
    "relation_type": "supports|contradicts|refines|extends|derived_from",
    "confidence": 0.0-1.0,
    "reason": "简短理由"
  }
]

关系类型：
- supports: 主张 A 支持主张 B
- contradicts: 主张 A 与主张 B 矛盾
- refines: 主张 A 细化/完善了主张 B
- extends: 主张 A 扩展了主张 B 到新领域
- derived_from: 主张 B 从主张 A 推导而来

只返回 JSON 数组。如果主张之间没有显著关系，返回 []。

主张列表（带序号）：
{claims_text}
"""


class RelationLinker:
    """Creates GraphRelation edges between extracted blocks."""

    def __init__(self, llm_client):
        self.llm = llm_client

    def link_claims(self, claims: list, graph_id: str, extraction_run_id: str) -> list[GraphRelation]:
        """Find relations between claims within the same paper."""
        if len(claims) < 2:
            return []

        claims_text = "\n".join(
            f"[{i}] ({c.claim_type.value}) {c.normalized_claim or c.claim_text}"
            for i, c in enumerate(claims)
        )

        prompt = RELATION_LINKING_PROMPT.format(claims_text=claims_text[:4000])
        raw_response = self.llm.complete(prompt)
        items = self._parse_response(raw_response)

        relations = []
        for item in items:
            src_idx = item.get("source_index", -1)
            tgt_idx = item.get("target_index", -1)
            if src_idx < 0 or tgt_idx < 0 or src_idx >= len(claims) or tgt_idx >= len(claims):
                continue

            rel_type_str = item.get("relation_type", "")
            if rel_type_str not in [r.value for r in GraphRelationType]:
                continue

            relation = GraphRelation(
                graph_id=graph_id,
                source_id=claims[src_idx].claim_id,
                target_id=claims[tgt_idx].claim_id,
                source_type="claim",
                target_type="claim",
                relation_type=GraphRelationType(rel_type_str),
                confidence=float(item.get("confidence", 0.5)),
                extraction_run_id=extraction_run_id,
                evidence_json=json.dumps({"reason": item.get("reason", "")}, ensure_ascii=False),
            )
            relations.append(relation)

        return relations

    def link_evidence_to_claims(
        self, evidence_blocks: list, claims: list, graph_id: str, extraction_run_id: str
    ) -> list[GraphRelation]:
        """Create SUPPORTS relations from evidence to claims."""
        relations = []
        for ev in evidence_blocks:
            for claim_id in ev.supports_claim_ids:
                relation = GraphRelation(
                    graph_id=graph_id,
                    source_id=ev.evidence_id,
                    target_id=claim_id,
                    source_type="evidence",
                    target_type="claim",
                    relation_type=GraphRelationType.SUPPORTS,
                    confidence=0.8,
                    extraction_run_id=extraction_run_id,
                )
                relations.append(relation)
        return relations

    def link_limitations_to_claims(
        self, limitation_blocks: list, claims: list, graph_id: str, extraction_run_id: str
    ) -> list[GraphRelation]:
        """Create CONSTRAINS relations from limitations to claims."""
        relations = []
        for lim in limitation_blocks:
            for claim_id in lim.affected_claim_ids:
                relation = GraphRelation(
                    graph_id=graph_id,
                    source_id=lim.limitation_id,
                    target_id=claim_id,
                    source_type="limitation",
                    target_type="claim",
                    relation_type=GraphRelationType.CONSTRAINS,
                    confidence=0.8,
                    extraction_run_id=extraction_run_id,
                )
                relations.append(relation)
        return relations

    def _parse_response(self, raw: str) -> list:
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:])
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            import re
            match = re.search(r"\[.*\]", cleaned, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass
            return []
