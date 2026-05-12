"""
Rule-based scanner that detects structural grey zones in SectionBlock lists.

Produces RepairCandidate objects for the reranker to score.
Does NOT modify sections — only flags suspicious patterns.
"""

import re
import hashlib
from dataclasses import dataclass, field
from typing import List, Optional

from ..models.section import SectionBlock


# ---- Patterns ----

_SENTENCE_END = re.compile(r"[。！？.!?）」》\"]$")
_CHAPTER_START = re.compile(r"^第[零一二三四五六七八九十百千]+[章节篇部]")
_CHAPTER_START_EN = re.compile(r"^(Chapter|Part|Section)\s+\d+", re.IGNORECASE)
_TOC_DOTS = re.compile(r"[……]{2,}|\.{4,}")
_FRONT_MATTER_TITLES = re.compile(r"(前言|编写说明|序言|Preface|Foreword|Introduction to)")
_FRONT_MATTER_PAGES = re.compile(r"^(I|II|III|IV|V|VI|VII|VIII|IX|X|XI|XII)$")


@dataclass
class RepairCandidate:
    """A structural grey zone flagged by rule-based detection."""
    candidate_id: str
    repair_type: str  # adjacent_merge | heading_role | toc_boundary | front_matter_boundary | chapter_boundary | structure_gap
    section_ids: list[str]
    description: str = ""
    features: dict = field(default_factory=dict)
    priority: float = 0.5


class CandidateDetector:
    """Scans SectionBlock lists for structural patterns that may need repair.

    Each detection method returns candidates with a priority score.
    Candidates with priority < min_priority are discarded.
    """

    def __init__(self, min_priority: float = 0.3):
        self.min_priority = min_priority

    # ---- Public API ----

    def detect(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Run all detection rules and return filtered candidates."""
        candidates: List[RepairCandidate] = []
        n = len(sections)
        if n == 0:
            return []

        candidates.extend(self._detect_adjacent_merges(sections))
        candidates.extend(self._detect_heading_roles(sections))
        candidates.extend(self._detect_toc_boundary(sections))
        candidates.extend(self._detect_front_matter_boundary(sections))
        candidates.extend(self._detect_chapter_boundaries(sections))
        candidates.extend(self._detect_structure_gaps(sections))

        # Filter low-priority
        return [c for c in candidates if c.priority >= self.min_priority]

    # ---- Detection Methods ----

    def _detect_adjacent_merges(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect adjacent blocks that might be a single split paragraph."""
        candidates = []
        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]

            features = {}
            score = 0.0

            # Rule 1: Block A ends without sentence-ending punctuation
            text_a = a.raw_text.strip()
            if text_a and not _SENTENCE_END.search(text_a[-20:]):
                features["no_sentence_end"] = True
                score += 0.3

            # Rule 2: Block B starts lowercase or continues naturally
            text_b = b.raw_text.strip()
            if text_b and not text_b[0].isupper() and not text_b[0].isdigit():
                features["starts_lower"] = True
                score += 0.2

            # Rule 3: Short block sandwiched between same heading_path
            if len(text_a) < 100:
                features["short_block"] = True
                features["short_text_len"] = len(text_a)
                score += 0.3

            # Rule 4: Both blocks share the same heading path
            if a.heading_path == b.heading_path and a.heading_path:
                features["same_heading_path"] = True
                score += 0.2

            # Rule 5: No heading on block B (pure continuation)
            if not b.heading or b.heading_level.value == "paragraph_group":
                features["no_heading_on_b"] = True
                score += 0.1

            if score >= 0.4:
                candidates.append(RepairCandidate(
                    candidate_id=f"adjacent_merge:{a.section_id}:{b.section_id}",
                    repair_type="adjacent_merge",
                    section_ids=[a.section_id, b.section_id],
                    description=f"'{text_a[:30]}...' + '{text_b[:30]}...'",
                    features=features,
                    priority=round(min(score, 1.0), 2),
                ))
        return candidates

    def _detect_heading_roles(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect blocks where heading assignment looks wrong."""
        candidates = []
        for s in sections:
            text = s.raw_text.strip()

            # Heading too long to be a real heading (> 80 chars)
            if s.heading and len(s.heading) > 80:
                candidates.append(RepairCandidate(
                    candidate_id=f"heading_role:long:{s.section_id}",
                    repair_type="heading_role",
                    section_ids=[s.section_id],
                    description=f"Heading too long ({len(s.heading)} chars): '{s.heading[:40]}...'",
                    features={"heading_len": len(s.heading), "suspicion": "too_long"},
                    priority=0.5,
                ))

            # Text looks like a chapter heading but isn't marked as one
            if (not s.heading or s.heading_level.value == "paragraph_group") and text:
                first_line = text.split("\n")[0][:80]
                if _CHAPTER_START.match(first_line) or _CHAPTER_START_EN.match(first_line):
                    candidates.append(RepairCandidate(
                        candidate_id=f"heading_role:missing:{s.section_id}",
                        repair_type="heading_role",
                        section_ids=[s.section_id],
                        description=f"Possible missed heading: '{first_line[:40]}...'",
                        features={"first_line": first_line, "suspicion": "missing_heading"},
                        priority=0.4,
                    ))

        return candidates

    def _detect_toc_boundary(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect blocks that look like TOC residuals."""
        candidates = []
        for i, s in enumerate(sections[:10]):  # TOC is always near the start
            text = s.raw_text
            dot_density = len(_TOC_DOTS.findall(text))

            if dot_density >= 3:
                candidates.append(RepairCandidate(
                    candidate_id=f"toc_boundary:{s.section_id}",
                    repair_type="toc_boundary",
                    section_ids=[s.section_id],
                    description=f"TOC-like block with {dot_density} dot-leader patterns",
                    features={"dot_density": dot_density, "position": i},
                    priority=0.8,
                ))
            elif dot_density >= 1 and i <= 3:
                candidates.append(RepairCandidate(
                    candidate_id=f"toc_boundary:faint:{s.section_id}",
                    repair_type="toc_boundary",
                    section_ids=[s.section_id],
                    description=f"Possible TOC residual (faint signal, position {i})",
                    features={"dot_density": dot_density, "position": i},
                    priority=0.5,
                ))

        return candidates

    def _detect_front_matter_boundary(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect front matter boundaries (preface, foreword, etc.)."""
        candidates = []
        for i, s in enumerate(sections[:5]):
            combined = (s.heading or "") + " " + s.raw_text[:200]
            if _FRONT_MATTER_TITLES.search(combined):
                candidates.append(RepairCandidate(
                    candidate_id=f"front_matter_boundary:{s.section_id}",
                    repair_type="front_matter_boundary",
                    section_ids=[s.section_id],
                    description=f"Possible front matter at position {i}: '{s.heading or s.raw_text[:30]}'",
                    features={"position": i, "heading": s.heading},
                    priority=0.3,
                ))
        return candidates

    def _detect_chapter_boundaries(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect heading level jumps that suggest missing intermediate headings."""
        candidates = []
        level_order = {
            "title": 0, "chapter": 1, "section": 2,
            "subsection": 3, "subsubsection": 4, "paragraph_group": 5,
        }

        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]

            a_lv = level_order.get(a.heading_level.value, 5)
            b_lv = level_order.get(b.heading_level.value, 5)

            # Skip if either has no meaningful heading
            if a_lv >= 4 and b_lv >= 4:
                continue

            jump = b_lv - a_lv
            if jump > 1:
                candidates.append(RepairCandidate(
                    candidate_id=f"chapter_boundary:{a.section_id}:{b.section_id}",
                    repair_type="chapter_boundary",
                    section_ids=[a.section_id, b.section_id],
                    description=f"Heading level jump {a.heading_level.value} -> {b.heading_level.value}",
                    features={
                        "from_level": a.heading_level.value,
                        "to_level": b.heading_level.value,
                        "jump": jump,
                    },
                    priority=0.7,
                ))

        return candidates

    def _detect_structure_gaps(self, sections: List[SectionBlock]) -> List[RepairCandidate]:
        """Detect page gaps or missing content between adjacent blocks."""
        candidates = []
        for i in range(len(sections) - 1):
            a = sections[i]
            b = sections[i + 1]

            if a.page_end is not None and b.page_start is not None:
                gap = b.page_start - a.page_end
                if gap > 2:
                    candidates.append(RepairCandidate(
                        candidate_id=f"structure_gap:{a.section_id}:{b.section_id}",
                        repair_type="structure_gap",
                        section_ids=[a.section_id, b.section_id],
                        description=f"Page gap of {gap} pages between blocks",
                        features={"gap_pages": gap, "from_page": a.page_end, "to_page": b.page_start},
                        priority=0.4,
                    ))

        return candidates
