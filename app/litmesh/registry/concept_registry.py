"""
ConceptRegistry: validates and deduplicates ConceptKey candidates.

Flow:
1. Receive candidate ConceptKey from ConceptExtractor
2. Check exact key collision
3. Check alias overlap with existing concepts
4. Embedding similarity check (future, via retrieval module)
5. merge_policy gate
6. Create ReviewInboxItem (ConceptInbox)
7. Return validated (but still candidate-status) ConceptKey

Principle: LLM cannot create active concepts. Registry + human review only.
"""

from ..models.concept import ConceptKey, ConceptStatus, ReviewStatus


class ConceptRegistry:
    """Validates and deduplicates ConceptKey candidates."""

    def __init__(self, db):
        self.db = db

    def register(self, concept: ConceptKey, extraction_run_id: str) -> dict:
        """Validate a candidate concept and return a decision.

        Returns:
            {
                "decision": "accept_candidate" | "merge" | "reject_duplicate" | "needs_review",
                "reason": str,
                "existing_key": str or None (if merge),
            }
        """
        # 1. Exact key collision
        existing = self._get_by_key(concept.concept_key)
        if existing:
            return {
                "decision": "reject_duplicate",
                "reason": f"Exact key collision: {concept.concept_key}",
                "existing_key": concept.concept_key,
            }

        # 2. Alias overlap check
        for alias in concept.aliases:
            matches = self.db.find_concept_by_alias(alias, concept.graph_id)
            for match in matches:
                if match["concept_key"] in concept.do_not_merge_with:
                    continue
                return {
                    "decision": "needs_review",
                    "reason": f"Alias '{alias}' overlaps with existing concept '{match['concept_key']}'",
                    "existing_key": match["concept_key"],
                }

        # 3. merge_policy gate
        if concept.merge_policy.value == "strict":
            # Even stricter: require explicit human review for all new concepts
            return {
                "decision": "needs_review",
                "reason": "STRICT merge policy: all new concepts require human review",
                "existing_key": None,
            }

        # 4. Default: accept as candidate (still requires human review to become active)
        return {
            "decision": "accept_candidate",
            "reason": "Passed initial validation, awaiting human review",
            "existing_key": None,
        }

    def lookup(self, term: str, graph_id: str) -> list[dict]:
        """Look up concepts by term (exact label or alias match)."""
        results = []
        # Exact label match
        rows = self.db.conn.execute(
            "SELECT * FROM concept_keys WHERE (label_zh = ? OR label_en = ?) AND graph_id = ?",
            (term, term, graph_id)
        ).fetchall()
        results.extend(dict(r) for r in rows)

        # Alias match
        alias_matches = self.db.find_concept_by_alias(term, graph_id)
        for m in alias_matches:
            if m["concept_key"] not in [r["concept_key"] for r in results]:
                results.append(m)

        return results

    def activate(self, concept_key: str) -> bool:
        """Activate a reviewed concept (human-confirmed)."""
        self.db.conn.execute(
            "UPDATE concept_keys SET status = ?, review_status = ?, updated_at = datetime('now') WHERE concept_key = ?",
            (ConceptStatus.ACTIVE.value, ReviewStatus.APPROVED.value, concept_key)
        )
        self.db.conn.commit()
        return True

    def reject(self, concept_key: str) -> bool:
        """Reject a concept."""
        self.db.conn.execute(
            "UPDATE concept_keys SET status = ?, review_status = ?, updated_at = datetime('now') WHERE concept_key = ?",
            (ConceptStatus.REJECTED.value, ReviewStatus.REJECTED.value, concept_key)
        )
        self.db.conn.commit()
        return True

    def merge(self, source_key: str, target_key: str) -> bool:
        """Merge source_key into target_key (source becomes MERGED status)."""
        self.db.conn.execute(
            "UPDATE concept_keys SET status = ?, do_not_merge_with = do_not_merge_with || ?, updated_at = datetime('now') WHERE concept_key = ?",
            (ConceptStatus.MERGED.value, f'"{target_key}"', source_key)
        )
        # Optionally: update all blocks referencing source_key to target_key
        self.db.conn.commit()
        return True

    def _get_by_key(self, concept_key: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM concept_keys WHERE concept_key = ?", (concept_key,)
        ).fetchone()
        return dict(row) if row else None
