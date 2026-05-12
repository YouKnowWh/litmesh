"""
Orchestrator for the structural repair pipeline.

Wires together:
  CandidateDetector → RerankerClient → RepairPolicy
  → (optional) FallbackLLM → RepairLog

Usage:
    detector = CandidateDetector()
    reranker = RerankerClient()
    policy = RepairPolicy()
    log = RepairLog()

    executor = RepairExecutor(detector, reranker, policy, log=log)
    sections, report = executor.repair(sections, paper_id="...", mode="dry_run")
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional

from ..models.section import SectionBlock
from .candidate_detector import CandidateDetector, RepairCandidate
from .reranker_client import RerankerClient, RerankerScore
from .repair_policy import RepairPolicy
from .repair_log import RepairLog, RepairLogEntry
from .fallback_llm import FallbackLLM

logger = logging.getLogger("litmesh.repair")


class RepairExecutor:
    """Orchestrates the full repair pipeline."""

    def __init__(
        self,
        detector: Optional[CandidateDetector] = None,
        reranker: Optional[RerankerClient] = None,
        policy: Optional[RepairPolicy] = None,
        fallback_llm: Optional[FallbackLLM] = None,
        log: Optional[RepairLog] = None,
    ):
        self.detector = detector or CandidateDetector()
        self.reranker = reranker or RerankerClient()
        self.policy = policy or RepairPolicy()
        self.fallback_llm = fallback_llm
        self.log = log or RepairLog()

    def repair(
        self,
        sections: List[SectionBlock],
        paper_id: str = "",
        graph_id: str = "",
        mode: str = "dry_run",
    ) -> tuple[List[SectionBlock], dict]:
        """Run repair pipeline on a list of SectionBlocks.

        Args:
            sections: SectionBlocks from section_splitter.
            paper_id: Paper identifier for logging.
            graph_id: Graph identifier for logging.
            mode: "dry_run" (log only), "auto_only" (apply high-confidence auto-fixes),
                  "full" (auto-fixes + LLM fallback for grey-zone).

        Returns:
            (modified_sections, report_dict)
        """
        t0 = time.monotonic()
        n_initial = len(sections)

        phase = {
            "dry_run": "phase2_observe",
            "auto_only": "phase3_auto",
            "full": "phase4_llm",
        }.get(mode, "phase2_observe")

        report = {
            "candidates_found": 0,
            "auto_fixed": 0,
            "grey_zone": 0,
            "skipped": 0,
            "llm_triggered": 0,
            "llm_applied": 0,
            "sections_before": n_initial,
            "sections_after": n_initial,
            "elapsed_ms": 0,
        }

        # Step 1: Detect candidates
        candidates = self.detector.detect(sections)
        report["candidates_found"] = len(candidates)
        if not candidates:
            report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            return sections, report

        # Observation mode should never block the ingestion pipeline on local
        # reranker model loading/download. Log candidates only and return.
        if mode == "dry_run":
            for c in candidates:
                self._log_entry(
                    paper_id, graph_id, phase, c, None,
                    classification="observe", action_taken="logged_only",
                    before=self._snapshot(c, sections),
                )
            report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            return sections, report

        # Step 2: Score with reranker
        try:
            scores = self.reranker.score(candidates, sections)
        except ImportError as e:
            logger.warning("Reranker not available: %s — logging candidates only", e)
            for c in candidates:
                self._log_entry(
                    paper_id, graph_id, phase, c, None,
                    classification="skip", action_taken="skipped",
                    before=self._snapshot(c, sections),
                )
            report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)
            return sections, report

        # Step 3: Classify and apply
        score_map = {s.candidate_id: s for s in scores}
        apply_mode = mode in ("auto_only", "full")

        for candidate in candidates:
            score = score_map.get(candidate.candidate_id)
            if score is None:
                self._log_entry(
                    paper_id, graph_id, phase, candidate, None,
                    classification="skip", action_taken="skipped",
                    before=self._snapshot(candidate, sections),
                )
                report["skipped"] += 1
                continue

            classification = self.policy.classify(score)

            if classification == "auto_fix" and apply_mode:
                # Apply auto fix
                before = self._snapshot(candidate, sections)
                sections = self.policy.apply_repair(candidate, score, sections)
                after = self._snapshot(candidate, sections)
                self._log_entry(
                    paper_id, graph_id, phase, candidate, score,
                    classification="auto_fix", action_taken="applied",
                    before=before, after=after,
                )
                report["auto_fixed"] += 1

            elif (
                classification == "grey_zone"
                and mode == "full"
                and self.fallback_llm is not None
                and self.policy.needs_llm_fallback(score, candidate)
            ):
                # Trigger LLM fallback
                report["llm_triggered"] += 1
                llm_decision = self.fallback_llm.judge(candidate, sections, score)

                llm_applied = False
                if llm_decision.confidence >= self.policy.thresholds.llm_decision_min:
                    before = self._snapshot(candidate, sections)
                    # Map LLM decision to a synthetic RerankerScore for apply_repair
                    synthetic_score = RerankerScore(
                        candidate_id=candidate.candidate_id,
                        score=llm_decision.confidence,
                        label=llm_decision.decision,
                        confidence=llm_decision.confidence,
                    )
                    sections = self.policy.apply_repair(candidate, synthetic_score, sections)
                    after = self._snapshot(candidate, sections)
                    llm_applied = True
                    report["llm_applied"] += 1
                else:
                    before = self._snapshot(candidate, sections)
                    after = None

                entry = RepairLogEntry(
                    paper_id=paper_id, graph_id=graph_id, phase=phase,
                    candidate_id=candidate.candidate_id,
                    repair_type=candidate.repair_type,
                    section_ids=candidate.section_ids,
                    rule_priority=candidate.priority,
                    reranker_score=score.score,
                    reranker_confidence=score.confidence,
                    classification="grey_zone",
                    llm_triggered=True,
                    llm_decision=llm_decision.decision,
                    llm_reasoning=llm_decision.reasoning,
                    action_taken="applied" if llm_applied else "logged_only",
                    before_state=before,
                    after_state=after,
                )
                self.log.log(entry)

            else:
                # Log only
                before = self._snapshot(candidate, sections)
                action = "logged_only" if classification == "grey_zone" else "skipped"
                self._log_entry(
                    paper_id, graph_id, phase, candidate, score,
                    classification=classification, action_taken=action,
                    before=before,
                )
                if classification == "grey_zone":
                    report["grey_zone"] += 1
                else:
                    report["skipped"] += 1

        report["sections_after"] = len(sections)
        report["elapsed_ms"] = int((time.monotonic() - t0) * 1000)

        logger.info(
            "Repair done: mode=%s candidates=%d auto_fixed=%d grey=%d skipped=%d "
            "llm_triggered=%d llm_applied=%d sections %d→%d elapsed=%dms",
            mode, report["candidates_found"], report["auto_fixed"],
            report["grey_zone"], report["skipped"],
            report["llm_triggered"], report["llm_applied"],
            n_initial, report["sections_after"], report["elapsed_ms"],
        )

        return sections, report

    # ---- Internal helpers ----

    def _log_entry(
        self,
        paper_id: str,
        graph_id: str,
        phase: str,
        candidate: RepairCandidate,
        score: Optional[RerankerScore],
        classification: str,
        action_taken: str,
        before: dict,
        after: Optional[dict] = None,
    ):
        entry = RepairLogEntry(
            paper_id=paper_id,
            graph_id=graph_id,
            phase=phase,
            candidate_id=candidate.candidate_id,
            repair_type=candidate.repair_type,
            section_ids=candidate.section_ids,
            rule_priority=candidate.priority,
            reranker_score=score.score if score else 0.0,
            reranker_confidence=score.confidence if score else 0.0,
            classification=classification,
            llm_triggered=False,
            action_taken=action_taken,
            before_state=before,
            after_state=after,
        )
        self.log.log(entry)

    @staticmethod
    def _snapshot(candidate: RepairCandidate, sections: List[SectionBlock]) -> dict:
        """Capture key fields of affected sections for audit log."""
        id_map = {s.section_id: s for s in sections}
        snap = {}
        for sid in candidate.section_ids:
            s = id_map.get(sid)
            if s:
                snap[sid] = {
                    "heading": s.heading,
                    "heading_level": s.heading_level.value,
                    "heading_path": s.heading_path,
                    "raw_text_len": len(s.raw_text),
                    "raw_text_preview": s.raw_text[:120],
                    "page_start": s.page_start,
                    "page_end": s.page_end,
                }
        return snap
