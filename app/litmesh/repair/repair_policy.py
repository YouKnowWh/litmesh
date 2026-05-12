"""
Threshold management, classification, and repair application.

Centralizes all tunable parameters for the repair pipeline.
"""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Set

from ..models.section import SectionBlock, HeadingLevel, StructureStatus
from .candidate_detector import RepairCandidate
from .reranker_client import RerankerScore


@dataclass
class RepairThresholds:
    auto_approve: float = 0.85   # >= this: auto-fix
    grey_zone_low: float = 0.5   # >= this and < auto_approve: grey zone
    # < grey_zone_low: skip
    llm_decision_min: float = 0.7  # LLM fallback must have confidence >= this to apply


class RepairPolicy:
    """Classifies reranker scores and applies approved repairs to SectionBlocks."""

    def __init__(
        self,
        thresholds: Optional[RepairThresholds] = None,
        enabled_types: Optional[Set[str]] = None,
        high_impact_types: Optional[Set[str]] = None,
    ):
        self.thresholds = thresholds or RepairThresholds()
        self.enabled_types = enabled_types or {
            "adjacent_merge", "heading_role", "toc_boundary",
            "front_matter_boundary", "chapter_boundary", "structure_gap",
        }
        self.high_impact_types = high_impact_types or {
            "toc_boundary", "chapter_boundary", "front_matter_boundary",
        }

    # ---- Classification ----

    def classify(self, score: RerankerScore) -> str:
        """Classify a reranker score into auto_fix / grey_zone / skip."""
        if score.confidence >= self.thresholds.auto_approve:
            return "auto_fix"
        elif score.confidence >= self.thresholds.grey_zone_low:
            return "grey_zone"
        else:
            return "skip"

    def needs_llm_fallback(
        self, score: RerankerScore, candidate: RepairCandidate
    ) -> bool:
        """True if grey-zone + high-impact → should trigger LLM fallback."""
        return (
            self.classify(score) == "grey_zone"
            and candidate.repair_type in self.high_impact_types
        )

    # ---- Repair Application ----

    def apply_repair(
        self,
        candidate: RepairCandidate,
        score: RerankerScore,
        sections: List[SectionBlock],
    ) -> List[SectionBlock]:
        """Apply the repair decision to a copy of the sections list."""
        sections = deepcopy(sections)
        if candidate.repair_type == "adjacent_merge":
            self._apply_merge(candidate, sections)
        elif candidate.repair_type == "heading_role":
            self._apply_heading_role(candidate, score, sections)
        elif candidate.repair_type == "toc_boundary":
            self._apply_toc_boundary(candidate, sections)
        elif candidate.repair_type == "front_matter_boundary":
            self._apply_front_matter(candidate, sections)
        elif candidate.repair_type == "chapter_boundary":
            self._apply_chapter_boundary(candidate, sections)
        elif candidate.repair_type == "structure_gap":
            # structure_gap doesn't modify — just logged
            pass
        return sections

    def _apply_merge(self, candidate: RepairCandidate, sections: List[SectionBlock]):
        """Merge two adjacent blocks into the first one."""
        id_map = {s.section_id: i for i, s in enumerate(sections)}
        if len(candidate.section_ids) < 2:
            return
        idx_a = id_map.get(candidate.section_ids[0])
        idx_b = id_map.get(candidate.section_ids[1])
        if idx_a is None or idx_b is None or idx_b - idx_a != 1:
            return

        a = sections[idx_a]
        b = sections[idx_b]

        # Merge text with a paragraph break
        a.raw_text = a.raw_text.rstrip() + "\n\n" + b.raw_text.lstrip()
        if a.summary and b.summary:
            a.summary = a.summary.rstrip() + " " + b.summary.lstrip()
        elif b.summary:
            a.summary = b.summary

        # Extend page range
        if a.page_end is not None and b.page_end is not None:
            a.page_end = max(a.page_end, b.page_end)
        elif b.page_end is not None:
            a.page_end = b.page_end

        # Mark structure as repaired
        a.structure_status = StructureStatus.RECONSTRUCTED

        # Remove the merged block
        sections.pop(idx_b)

    def _apply_heading_role(
        self,
        candidate: RepairCandidate,
        score: RerankerScore,
        sections: List[SectionBlock],
    ):
        """Fix heading role misclassification."""
        id_map = {s.section_id: i for i, s in enumerate(sections)}
        idx = id_map.get(candidate.section_ids[0])
        if idx is None:
            return
        s = sections[idx]

        if score.label == "not_heading" and s.heading:
            # Demote heading to display_title, merge heading into raw_text
            s.raw_text = f"{s.heading}\n{s.raw_text}"
            s.heading = ""
            s.heading_level = HeadingLevel.PARAGRAPH_GROUP
            s.heading_confidence = 0.4
            s.structure_status = StructureStatus.RECONSTRUCTED
        elif score.label == "heading":
            # Promote first line to heading if missing
            first_line = s.raw_text.split("\n")[0][:80]
            if not s.heading and first_line:
                s.heading = first_line
                s.heading_level = HeadingLevel.SECTION
                s.heading_confidence = 0.6
                s.structure_status = StructureStatus.RECONSTRUCTED

    def _apply_toc_boundary(self, candidate: RepairCandidate, sections: List[SectionBlock]):
        """Mark TOC residual blocks so they are skipped during extraction."""
        id_map = {s.section_id: i for i, s in enumerate(sections)}
        idx = id_map.get(candidate.section_ids[0])
        if idx is None:
            return
        s = sections[idx]
        s.structure_status = StructureStatus.RECONSTRUCTED
        s.heading = "目录"
        s.heading_level = HeadingLevel.PARAGRAPH_GROUP
        s.heading_confidence = 0.5

    def _apply_front_matter(self, candidate: RepairCandidate, sections: List[SectionBlock]):
        """Mark front matter blocks with appropriate heading."""
        id_map = {s.section_id: i for i, s in enumerate(sections)}
        idx = id_map.get(candidate.section_ids[0])
        if idx is None:
            return
        s = sections[idx]
        s.structure_status = StructureStatus.RECONSTRUCTED
        if not s.heading:
            s.heading = "前言"
            s.heading_confidence = 0.5

    def _apply_chapter_boundary(self, candidate: RepairCandidate, sections: List[SectionBlock]):
        """Insert a synthetic heading for missing intermediate levels."""
        # chapter_boundary doesn't insert blocks (would shift indices),
        # but marks the later block's structure for review
        if len(candidate.section_ids) < 2:
            return
        id_map = {s.section_id: i for i, s in enumerate(sections)}
        idx = id_map.get(candidate.section_ids[1])
        if idx is None:
            return
        sections[idx].structure_status = StructureStatus.NEEDS_REVIEW
        sections[idx].heading_confidence = 0.5

    # ---- Factory ----

    @classmethod
    def from_env(cls) -> "RepairPolicy":
        auto = float(os.environ.get("LITMESH_REPAIR_AUTO_THRESHOLD", "0.85"))
        grey = float(os.environ.get("LITMESH_REPAIR_GREY_THRESHOLD", "0.5"))
        llm_min = float(os.environ.get("LITMESH_REPAIR_LLM_DECISION_MIN", "0.7"))
        return cls(thresholds=RepairThresholds(
            auto_approve=auto, grey_zone_low=grey, llm_decision_min=llm_min,
        ))
