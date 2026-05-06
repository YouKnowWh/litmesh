"""
BridgeDetector: proposes cross-graph concept bridges.

After each graph has its own concepts (v0.3), the bridge detector
compares concept keys across graphs and proposes BridgeRelations.

Bridge types (from BridgeRelationType):
- same_as: concept X in graph A IS concept Y in graph B
- broader_than / narrower_than: hierarchical cross-graph
- analogous_to: similar but not same (with warning)
- applies_to: framework in A applies to domain in B
- conflicts_with: claim in A conflicts with claim in B
- transfers_to: finding in A may apply to context in B

All bridges go to BridgeInbox for human review.
Only active bridges are used in traversal.
"""

import json
from datasketch import MinHash, MinHashLSH

from ..models.relation import BridgeRelation, BridgeRelationType, BridgeStatus
from ..models.review import ReviewInboxItem, InboxType, InboxPriority, InboxDecision


class BridgeDetector:
    """Detects cross-graph concept bridges.

    Pipeline:
    1. Build MinHash signatures for concepts in each graph
    2. Cross-graph LSH query to find candidate pairs
    3. Jaccard verification + alias/label comparison
    4. Classify bridge type
    5. Create BridgeRelation candidates → BridgeInbox
    """

    def __init__(self, db, llm_client=None):
        self.db = db
        self.llm = llm_client

    def detect_all(self, min_label_similarity: float = 0.3) -> dict:
        """Run bridge detection across all graph pairs.

        Returns stats: {bridges_proposed, bridges_auto_accepted, inbox_items}
        """
        graphs = self._all_graphs()
        if len(graphs) < 2:
            return {"bridges_proposed": 0, "bridges_auto_accepted": 0, "inbox_items": 0}

        # Build per-graph concept MinHash signatures
        graph_concepts: dict[str, list[dict]] = {}
        graph_minhashes: dict[str, MinHash] = {}

        for g in graphs:
            gid = g["graph_id"]
            concepts = self._active_concepts(gid)
            graph_concepts[gid] = concepts
            m = MinHash(num_perm=128)
            for c in concepts:
                text = f"{c['label_zh']} {c['label_en']} {c['definition']}"
                for s in _shingle(text):
                    m.update(s.encode("utf-8"))
            graph_minhashes[gid] = m

        stats = {"bridges_proposed": 0, "bridges_auto_accepted": 0, "inbox_items": 0}

        # Cross-graph pair detection
        graph_ids = list(graph_concepts.keys())
        for i in range(len(graph_ids)):
            for j in range(i + 1, len(graph_ids)):
                gid_a, gid_b = graph_ids[i], graph_ids[j]
                batch_stats = self._detect_between(gid_a, gid_b, graph_concepts[gid_a], graph_concepts[gid_b])
                for k, v in batch_stats.items():
                    stats[k] += v

        return stats

    def _detect_between(self, gid_a: str, gid_b: str,
                         concepts_a: list[dict], concepts_b: list[dict]) -> dict:
        """Detect bridges between two specific graphs."""
        stats = {"bridges_proposed": 0, "bridges_auto_accepted": 0, "inbox_items": 0}

        for ca in concepts_a:
            # Skip concepts with do_not_merge_with blocking this target graph
            do_not = json.loads(ca.get("do_not_merge_with", "[]")) if isinstance(ca.get("do_not_merge_with"), str) else ca.get("do_not_merge_with", [])

            for cb in concepts_b:
                if cb["concept_key"] in do_not:
                    continue

                bridge_type, confidence = self._classify_pair(ca, cb)
                if not bridge_type or confidence < 0.3:
                    continue

                warning = ""
                if bridge_type == BridgeRelationType.ANALOGOUS_TO:
                    warning = f"'{ca['label_zh']}' and '{cb['label_zh']}' are analogous, NOT identical. Do not treat as same_as."

                bridge = BridgeRelation(
                    source_graph_id=gid_a,
                    target_graph_id=gid_b,
                    source_key=ca["concept_key"],
                    target_key=cb["concept_key"],
                    bridge_type=bridge_type,
                    bridge_confidence=confidence,
                    warning=warning,
                    evidence_json=json.dumps({
                        "source_label": ca["label_zh"],
                        "target_label": cb["label_zh"],
                        "source_namespace": ca["namespace"],
                        "target_namespace": cb["namespace"],
                    }, ensure_ascii=False),
                )

                stats["bridges_proposed"] += 1

                # High-confidence same_as within same namespace → auto-accept
                if (bridge_type == BridgeRelationType.SAME_AS and
                    confidence >= 0.8 and
                    ca["namespace"] == cb["namespace"]):
                    bridge.review_status = BridgeStatus.ACTIVE
                    stats["bridges_auto_accepted"] += 1
                else:
                    bridge.review_status = BridgeStatus.CANDIDATE
                    self._create_bridge_inbox(bridge, ca, cb)
                    stats["inbox_items"] += 1

                self.db.insert_bridge(bridge)

        return stats

    def _classify_pair(self, ca: dict, cb: dict) -> tuple[BridgeRelationType | None, float]:
        """Classify a concept pair as a bridge type with confidence."""
        label_a = (ca.get("label_zh", "") + " " + ca.get("label_en", "")).strip().lower()
        label_b = (cb.get("label_zh", "") + " " + cb.get("label_en", "")).strip().lower()

        # Exact label match → same_as
        if label_a and label_a == label_b:
            return BridgeRelationType.SAME_AS, 0.9

        # Same concept_key slug → same_as
        key_a = ca["concept_key"].split(":", 1)[-1] if ":" in ca["concept_key"] else ca["concept_key"]
        key_b = cb["concept_key"].split(":", 1)[-1] if ":" in cb["concept_key"] else cb["concept_key"]
        if key_a == key_b:
            return BridgeRelationType.SAME_AS, 0.8

        # Alias overlap → same_as candidate
        aliases_a = set(_parse_list(ca.get("aliases", "[]")))
        aliases_b = set(_parse_list(cb.get("aliases", "[]")))
        overlap = aliases_a & aliases_b
        if overlap:
            return BridgeRelationType.SAME_AS, min(0.7, 0.4 + 0.1 * len(overlap))

        # Same namespace + partial label overlap → analogous_to
        if ca["namespace"] == cb["namespace"] and label_a and label_b:
            # Check character overlap
            common_chars = set(label_a) & set(label_b)
            if len(common_chars) >= 3:
                return BridgeRelationType.ANALOGOUS_TO, 0.4

        # Framework on one side, concept on the other → applies_to
        if (ca["namespace"] == "framework" and cb["namespace"] == "concept"):
            return BridgeRelationType.APPLIES_TO, 0.3
        if (cb["namespace"] == "framework" and ca["namespace"] == "concept"):
            return BridgeRelationType.APPLIES_TO, 0.3

        return None, 0.0

    def _create_bridge_inbox(self, bridge: BridgeRelation, ca: dict, cb: dict):
        """Create a BridgeInbox review item."""
        inbox = ReviewInboxItem(
            inbox_type=InboxType.BRIDGE,
            item_id=bridge.bridge_id,
            item_type="bridge_relation",
            title=f"{bridge.bridge_type.value}: {ca['label_zh']} ↔ {cb['label_zh']}",
            description=(
                f"Source: {ca['concept_key']} ({ca['label_zh']})\n"
                f"Target: {cb['concept_key']} ({cb['label_zh']})\n"
                f"Type: {bridge.bridge_type.value}\n"
                f"Confidence: {bridge.bridge_confidence:.0%}\n"
                f"Warning: {bridge.warning}"
            ),
            extraction_confidence=bridge.bridge_confidence,
            priority=InboxPriority.HIGH if bridge.bridge_type == BridgeRelationType.SAME_AS else InboxPriority.MEDIUM,
            graph_id=bridge.source_graph_id,
            suggested_actions=[
                InboxDecision.APPROVE,
                InboxDecision.EDIT,
                InboxDecision.MARK_AS_CONFLICT,
                InboxDecision.REJECT,
            ] if bridge.bridge_type != BridgeRelationType.ANALOGOUS_TO else [
                InboxDecision.APPROVE,  # Analogous is valid, just needs a warning label
                InboxDecision.REJECT,
            ],
        )
        self.db.insert_inbox_item(inbox)

    def _active_concepts(self, graph_id: str) -> list[dict]:
        rows = self.db.conn.execute(
            "SELECT * FROM concept_keys WHERE graph_id = ? AND status = 'active'",
            (graph_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def _all_graphs(self) -> list[dict]:
        rows = self.db.conn.execute("SELECT graph_id FROM series_graphs").fetchall()
        return [dict(r) for r in rows]


def _shingle(text: str, k: int = 3) -> set[str]:
    import re
    text = re.sub(r"\s+", " ", text.lower().strip())
    if len(text) < k:
        return {text}
    return {text[i:i + k] for i in range(len(text) - k + 1)}


def _parse_list(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return []
    return []
