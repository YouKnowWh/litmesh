"""
Review module: programmatic interface to the ReviewInbox.

Provides approve/reject/edit/merge/split operations with side effects:
- Approving a claim: sets its status to 'active'
- Rejecting a claim: sets its status to 'rejected'
- Merging a concept: calls ConceptRegistry.merge()
- etc.
"""

from ..models.review import ReviewInboxItem, InboxType, InboxDecision
from ..models.claim import ClaimStatus
from ..models.evidence import EvidenceStatus
from ..models.limitation import LimitationStatus
from ..models.concept import ConceptStatus, ReviewStatus


class ReviewManager:
    """Handles review decisions and their side effects on the knowledge base."""

    def __init__(self, db, concept_registry=None):
        self.db = db
        self.concept_registry = concept_registry

    def approve(self, inbox_id: str, decided_by: str = "human") -> dict:
        """Approve an inbox item and activate the corresponding object."""
        inbox = self._get_inbox(inbox_id)
        if not inbox:
            return {"ok": False, "error": "Inbox item not found"}

        # Activate the target object
        item_type = inbox["item_type"]
        item_id = inbox["item_id"]

        if item_type == "claim":
            self.db.update_claim_status(item_id, ClaimStatus.ACTIVE.value)
        elif item_type == "evidence":
            self.db.conn.execute(
                "UPDATE evidence_blocks SET status = ? WHERE evidence_id = ?",
                (EvidenceStatus.ACTIVE.value, item_id)
            )
        elif item_type == "limitation":
            self.db.conn.execute(
                "UPDATE limitation_blocks SET status = ? WHERE limitation_id = ?",
                (LimitationStatus.ACTIVE.value, item_id)
            )
        elif item_type == "concept":
            if self.concept_registry:
                self.concept_registry.activate(item_id)
            else:
                self.db.conn.execute(
                    "UPDATE concept_keys SET status = ?, review_status = ? WHERE concept_key = ?",
                    (ConceptStatus.ACTIVE.value, ReviewStatus.APPROVED.value, item_id)
                )

        self.db.resolve_inbox_item(
            inbox_id,
            decision=InboxDecision.APPROVE.value,
            decided_by=decided_by,
            notes="Approved",
        )
        self.db.conn.commit()
        return {"ok": True, "action": "approved", "item_id": item_id}

    def reject(self, inbox_id: str, reason: str = "", decided_by: str = "human") -> dict:
        """Reject an inbox item."""
        inbox = self._get_inbox(inbox_id)
        if not inbox:
            return {"ok": False, "error": "Inbox item not found"}

        item_type = inbox["item_type"]
        item_id = inbox["item_id"]

        if item_type == "claim":
            self.db.update_claim_status(item_id, ClaimStatus.REJECTED.value)
        elif item_type == "evidence":
            self.db.conn.execute(
                "UPDATE evidence_blocks SET status = ? WHERE evidence_id = ?",
                (EvidenceStatus.REJECTED.value, item_id)
            )
        elif item_type == "limitation":
            self.db.conn.execute(
                "UPDATE limitation_blocks SET status = ? WHERE limitation_id = ?",
                (LimitationStatus.REJECTED.value, item_id)
            )
        elif item_type == "concept":
            if self.concept_registry:
                self.concept_registry.reject(item_id)
            else:
                self.db.conn.execute(
                    "UPDATE concept_keys SET status = ?, review_status = ? WHERE concept_key = ?",
                    (ConceptStatus.REJECTED.value, ReviewStatus.REJECTED.value, item_id)
                )

        self.db.resolve_inbox_item(
            inbox_id,
            decision=InboxDecision.REJECT.value,
            decided_by=decided_by,
            notes=reason,
        )
        self.db.conn.commit()
        return {"ok": True, "action": "rejected", "item_id": item_id}

    def downgrade_confidence(self, inbox_id: str, new_confidence: float = 0.3,
                              decided_by: str = "human") -> dict:
        """Downgrade confidence of an item (keep as active but low confidence)."""
        inbox = self._get_inbox(inbox_id)
        if not inbox:
            return {"ok": False, "error": "Inbox item not found"}

        item_type = inbox["item_type"]
        item_id = inbox["item_id"]

        if item_type == "claim":
            self.db.conn.execute(
                "UPDATE claim_blocks SET extraction_confidence = ?, claim_confidence = ? WHERE claim_id = ?",
                (new_confidence, new_confidence, item_id)
            )

        self.db.resolve_inbox_item(
            inbox_id,
            decision=InboxDecision.DOWNGRADE_CONFIDENCE.value,
            decided_by=decided_by,
            notes=f"Confidence downgraded to {new_confidence}",
        )
        self.db.conn.commit()
        return {"ok": True, "action": "downgraded", "item_id": item_id}

    def mark_as_limitation(self, inbox_id: str, decided_by: str = "human") -> dict:
        """Reclassify a claim as a limitation."""
        inbox = self._get_inbox(inbox_id)
        if not inbox:
            return {"ok": False, "error": "Inbox item not found"}

        # This is a conceptual reclassification; for MVP we just note it
        self.db.resolve_inbox_item(
            inbox_id,
            decision=InboxDecision.MARK_AS_LIMITATION.value,
            decided_by=decided_by,
            notes="Reclassified as limitation",
        )
        self.db.conn.commit()
        return {"ok": True, "action": "marked_as_limitation"}

    def get_pending_count(self, inbox_type: InboxType | None = None) -> int:
        """Count pending review items."""
        items = self.db.get_pending_inbox(inbox_type.value if inbox_type else None)
        return len(items)

    def _get_inbox(self, inbox_id: str) -> dict | None:
        row = self.db.conn.execute(
            "SELECT * FROM review_inbox WHERE inbox_id = ?", (inbox_id,)
        ).fetchone()
        return dict(row) if row else None
