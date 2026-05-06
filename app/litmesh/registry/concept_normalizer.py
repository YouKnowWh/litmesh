"""
Concept normalization pipeline.

Full flow:
  1. Accept candidate ConceptKeys from ConceptExtractor
  2. Resolve each against the ConceptRegistry (dedup, alias check)
  3. Create ConceptInbox items for new/ambiguous concepts
  4. Link resolved concept_keys back to Claim/Evidence/Limitation blocks
  5. Generate "mentions" GraphRelations for concept<->block links
  6. Detect potential conflicts between overlapping concepts

Principle 4: LLM produces candidates, normalizer controls writes.
"""

import json
import hashlib
from collections import defaultdict

from ..models.concept import ConceptKey, ConceptNamespace, ConceptStatus, ReviewStatus, MergePolicy
from ..models.relation import GraphRelation, GraphRelationType
from ..models.review import ReviewInboxItem, InboxType, InboxPriority, InboxDecision


class ConceptNormalizer:
    """Resolves, deduplicates, and links concept candidates to blocks."""

    def __init__(self, db, concept_registry):
        self.db = db
        self.registry = concept_registry

    def normalize_extraction(
        self,
        candidate_concepts: list[ConceptKey],
        claims: list,
        evidence_blocks: list,
        limitations: list,
        extraction_run_id: str,
        auto_activate_low_risk: bool = False,
    ) -> dict:
        """Run the full normalization pipeline for one extraction batch.

        Args:
            candidate_concepts: Concepts extracted by ConceptExtractor
            claims: ClaimBlocks from the same extraction run
            evidence_blocks: EvidenceBlocks from the same extraction run
            limitations: LimitationBlocks from the same extraction run
            extraction_run_id: FK to ExtractionRun
            auto_activate_low_risk: If True, auto-activate concepts with no conflicts

        Returns:
            {
                "new_concepts": int,       # Created as candidates
                "merged_concepts": int,     # Merged into existing
                "activated_concepts": int,  # Auto-activated
                "inbox_items": int,         # ReviewInbox entries created
                "block_links_updated": int, # Blocks updated with canonical keys
                "relations_created": int,   # New GraphRelations
                "conflicts_detected": int,  # Potential concept conflicts
            }
        """
        stats = {
            "new_concepts": 0,
            "merged_concepts": 0,
            "activated_concepts": 0,
            "inbox_items": 0,
            "block_links_updated": 0,
            "relations_created": 0,
            "conflicts_detected": 0,
        }

        # Phase 1: Resolve each candidate against the registry
        resolved_map: dict[str, str] = {}  # raw_term -> canonical_concept_key

        for candidate in candidate_concepts:
            decision = self.registry.register(candidate, extraction_run_id)

            if decision["decision"] == "reject_duplicate":
                resolved_map[candidate.concept_key] = decision["existing_key"]
                stats["merged_concepts"] += 1
                continue

            elif decision["decision"] == "needs_review":
                # Save as candidate, add to ConceptInbox
                candidate.status = ConceptStatus.CANDIDATE
                candidate.review_status = ReviewStatus.PENDING
                self.db.insert_concept(candidate)
                self._create_concept_inbox(
                    candidate, decision.get("existing_key"), extraction_run_id
                )
                resolved_map[candidate.concept_key] = candidate.concept_key
                stats["new_concepts"] += 1
                stats["inbox_items"] += 1

            elif decision["decision"] == "accept_candidate":
                candidate.status = ConceptStatus.CANDIDATE
                self.db.insert_concept(candidate)

                if auto_activate_low_risk:
                    self.registry.activate(candidate.concept_key)
                    stats["activated_concepts"] += 1
                else:
                    self._create_concept_inbox(candidate, None, extraction_run_id)
                    stats["inbox_items"] += 1

                resolved_map[candidate.concept_key] = candidate.concept_key
                stats["new_concepts"] += 1

        # Phase 2: Resolve raw concept terms in blocks against the registry
        all_blocks = []
        for c in claims:
            all_blocks.append(("claim", c))
        for e in evidence_blocks:
            all_blocks.append(("evidence", e))
        for l in limitations:
            all_blocks.append(("limitation", l))

        for block_type, block in all_blocks:
            raw_keys = getattr(block, "concept_keys", [])
            if not raw_keys:
                continue

            canonical_keys = set()
            for raw_key in raw_keys:
                # Check if already resolved
                if raw_key in resolved_map:
                    canonical_keys.add(resolved_map[raw_key])
                    continue

                # Try exact concept_key match in registry
                matches = self.registry.lookup(raw_key, block.graph_id)
                if matches:
                    canonical_keys.add(matches[0]["concept_key"])
                    resolved_map[raw_key] = matches[0]["concept_key"]
                else:
                    # Try alias lookup
                    alias_matches = self.db.find_concept_by_alias(raw_key, block.graph_id)
                    if alias_matches:
                        canonical_keys.add(alias_matches[0]["concept_key"])
                        resolved_map[raw_key] = alias_matches[0]["concept_key"]
                    else:
                        # Unresolvable — leave as-is for now, will be checked again
                        # after ConceptInbox review
                        canonical_keys.add(raw_key)

            # Update the block's concept_keys with canonical keys
            setattr(block, "concept_keys", sorted(canonical_keys))

            # Persist the update
            block_id_col = f"{block_type}_id"
            block_id = getattr(block, block_id_col, None)
            if block_id:
                self.db.conn.execute(
                    f"UPDATE {block_type}_blocks SET concept_keys = ? WHERE {block_id_col} = ?",
                    (json.dumps(sorted(canonical_keys), ensure_ascii=False), block_id)
                )
                stats["block_links_updated"] += 1

        # Phase 3: Generate "mentions" relations for concept<->claim links
        for claim in claims:
            for ckey in claim.concept_keys:
                # Only create if it looks like a valid concept_key (has namespace prefix)
                if ":" not in ckey or ckey.startswith("concept_"):
                    continue
                rel = GraphRelation(
                    graph_id=claim.graph_id,
                    source_id=claim.claim_id,
                    target_id=ckey,
                    source_type="claim",
                    target_type="concept",
                    relation_type=GraphRelationType.MENTIONS,
                    confidence=0.7,  # LLM-extracted concept, moderate confidence
                )
                self.db.insert_relation(rel)
                stats["relations_created"] += 1

        self.db.conn.commit()

        # Phase 4: Detect potential concept conflicts (overlapping aliases)
        stats["conflicts_detected"] = self._detect_conflicts(candidate_concepts)

        return stats

    def resolve_concept_keys_on_block(self, block, graph_id: str) -> list[str]:
        """Resolve a single block's concept_keys to canonical keys.

        Returns the resolved list of canonical concept_keys.
        """
        raw_keys = getattr(block, "concept_keys", [])
        if not raw_keys:
            return []

        canonical = []
        for raw_key in raw_keys:
            if ":" in raw_key and not raw_key.startswith("concept_"):
                # Already looks canonical
                canonical.append(raw_key)
                continue

            # Try registry lookup
            matches = self.registry.lookup(raw_key, graph_id)
            if matches:
                canonical.append(matches[0]["concept_key"])
            else:
                alias_matches = self.db.find_concept_by_alias(raw_key, graph_id)
                if alias_matches:
                    canonical.append(alias_matches[0]["concept_key"])

        return sorted(set(canonical))

    def _create_concept_inbox(self, concept: ConceptKey, existing_key: str | None,
                               extraction_run_id: str):
        """Create a ConceptInbox review item."""
        title = f"[{concept.namespace.value}] {concept.label_zh or concept.concept_key}"
        description = f"Definition: {concept.definition}" if concept.definition else ""
        if existing_key:
            description += f"\n\nPossible duplicate of: {existing_key}"

        inbox = ReviewInboxItem(
            inbox_type=InboxType.CONCEPT,
            item_id=concept.concept_key,
            item_type="concept",
            title=title,
            description=description,
            source_text=concept.definition,
            extraction_confidence=0.7,
            priority=InboxPriority.MEDIUM,
            extraction_run_id=extraction_run_id,
            graph_id=concept.graph_id,
            suggested_actions=[
                InboxDecision.APPROVE,
                InboxDecision.EDIT,
                InboxDecision.MERGE,
                InboxDecision.REJECT,
            ],
        )
        if existing_key:
            inbox.merge_target_id = existing_key
            inbox.suggested_actions.insert(0, InboxDecision.MERGE)

        self.db.insert_inbox_item(inbox)

    def _detect_conflicts(self, candidates: list[ConceptKey]) -> int:
        """Detect candidates with overlapping aliases in the same graph.

        Returns count of conflicts detected.
        """
        conflicts = 0
        # Group candidates by graph_id
        by_graph = defaultdict(list)
        for c in candidates:
            by_graph[c.graph_id].append(c)

        for graph_id, concepts in by_graph.items():
            for i, c1 in enumerate(concepts):
                for c2 in concepts[i + 1:]:
                    # Check alias overlap
                    overlap = set(c1.aliases) & set(c2.aliases)
                    if overlap:
                        # Check if either is on do_not_merge_with list
                        if (c2.concept_key in c1.do_not_merge_with or
                            c1.concept_key in c2.do_not_merge_with):
                            continue

                        # Create conflict inbox item
                        inbox = ReviewInboxItem(
                            inbox_type=InboxType.CONFLICT,
                            item_id=f"{c1.concept_key}|{c2.concept_key}",
                            item_type="concept_conflict",
                            title=f"Possible duplicate: {c1.label_zh} vs {c2.label_zh}",
                            description=(
                                f"Overlapping aliases: {overlap}\n"
                                f"Concept 1: {c1.concept_key} ({c1.label_zh})\n"
                                f"Concept 2: {c2.concept_key} ({c2.label_zh})"
                            ),
                            priority=InboxPriority.HIGH,
                            graph_id=graph_id,
                            suggested_actions=[
                                InboxDecision.MERGE,
                                InboxDecision.REJECT,
                                InboxDecision.EDIT,
                            ],
                        )
                        self.db.insert_inbox_item(inbox)
                        conflicts += 1

        return conflicts
