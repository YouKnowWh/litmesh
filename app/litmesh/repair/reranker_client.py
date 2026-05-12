"""
Cross-Encoder reranker client for structural repair decisions.

Uses BAAI/bge-reranker-v2-m3 (multilingual) to score whether
two blocks should be merged, whether a block is a heading, etc.

First version: transformers + torch, local loading.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Optional, Dict

from .candidate_detector import RepairCandidate
from ..models.section import SectionBlock


@dataclass
class RerankerScore:
    candidate_id: str
    score: float       # 0-1 raw model score (sigmoid)
    label: str         # merge | split | keep | heading | not_heading | toc | not_toc | ...
    confidence: float  # 0-1 mapped confidence


# ---- Query templates per repair type ----

_QUERIES: Dict[str, str] = {
    "adjacent_merge": "这两段文本是否属于同一个段落，应该合并在一起？",
    "heading_role": "这段文本的标签是否正确地反映了它是标题还是正文？",
    "toc_boundary": "这段文本是否为目录或目录残留，不是正文内容？",
    "front_matter_boundary": "这段文本是否为前言、序言或编写说明？",
    "chapter_boundary": "这两个相邻块的标题层级是否出现了不正常的跳跃？",
    "structure_gap": "这两段文本之间的页面空缺是否代表有缺失的内容？",
}


def _build_query(candidate: RepairCandidate) -> str:
    """Get the query template for a repair type, falling back to a generic query."""
    return _QUERIES.get(candidate.repair_type, "这个文档结构是否需要修复？")


def _build_document(candidate: RepairCandidate, sections: List[SectionBlock]) -> str:
    """Build the document text that the query will be paired with."""
    id_map = {s.section_id: s for s in sections}
    parts = []

    if candidate.repair_type == "adjacent_merge":
        for sid in candidate.section_ids:
            s = id_map.get(sid)
            if s:
                heading = f"[{s.heading}]" if s.heading else "[无标题]"
                text = s.raw_text[:500]
                parts.append(f"{heading}\n{text}")
        return "\n---\n".join(parts)

    elif candidate.repair_type == "heading_role":
        s = id_map.get(candidate.section_ids[0])
        if s:
            heading = s.heading or "(无)"
            text = s.raw_text[:300]
            return f"标签/标题: {heading}\n正文开头: {text}"
        return ""

    elif candidate.repair_type in ("toc_boundary", "front_matter_boundary"):
        s = id_map.get(candidate.section_ids[0])
        if s:
            heading = s.heading or "(无)"
            text = s.raw_text[:500]
            return f"标题: {heading}\n内容: {text}"
        return ""

    elif candidate.repair_type == "chapter_boundary":
        parts = []
        for sid in candidate.section_ids:
            s = id_map.get(sid)
            if s:
                parts.append(f"层级={s.heading_level.value} 标题={s.heading or '(无)'}")
        return "\n→\n".join(parts)

    elif candidate.repair_type == "structure_gap":
        parts = []
        for sid in candidate.section_ids:
            s = id_map.get(sid)
            if s:
                parts.append(f"页{s.page_start}-{s.page_end}: {s.raw_text[:200]}")
        return "\n--- 缺页 ---\n".join(parts)

    # Generic fallback
    for sid in candidate.section_ids[:2]:
        s = id_map.get(sid)
        if s:
            parts.append(s.raw_text[:300])
    return "\n---\n".join(parts)


class RerankerClient:
    """Cross-encoder reranker for structural repair scoring."""

    def __init__(
        self,
        model_name: str = "",
        device: Optional[str] = None,
        batch_size: int = 8,
    ):
        self.model_name = model_name or os.environ.get(
            "LITMESH_RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"
        )
        self.device = device or os.environ.get("LITMESH_RERANKER_DEVICE", "cpu")
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @property
    def model_loaded(self) -> bool:
        return self._model is not None

    def _ensure_model(self):
        """Lazy-load the reranker model."""
        if self._model is not None:
            return
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers not installed. "
                "Install with: pip install transformers torch"
            )
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name
        )
        self._model.to(self.device)
        self._model.eval()

    def score(
        self,
        candidates: List[RepairCandidate],
        sections: List[SectionBlock],
    ) -> List[RerankerScore]:
        """Score all candidates against their respective repair queries."""
        if not candidates:
            return []

        self._ensure_model()

        pairs = []
        for c in candidates:
            query = _build_query(c)
            doc = _build_document(c, sections)
            pairs.append((query, doc))

        scores = self._compute_scores(pairs)
        return [
            _to_reranker_score(candidate, score)
            for candidate, score in zip(candidates, scores)
        ]

    def _compute_scores(self, pairs: List[tuple]) -> List[float]:
        """Run cross-encoder on (query, document) pairs."""
        import torch

        all_scores = []
        for i in range(0, len(pairs), self.batch_size):
            batch = pairs[i:i + self.batch_size]
            inputs = self._tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                batch_scores = torch.sigmoid(outputs.logits).squeeze(-1)
                all_scores.extend(batch_scores.cpu().tolist())

        # If single scalar, wrap
        if not isinstance(all_scores, list):
            all_scores = [all_scores]
        return [float(s) if hasattr(s, '__float__') else float(s) for s in all_scores]


def _to_reranker_score(candidate: RepairCandidate, raw_score: float) -> RerankerScore:
    """Convert raw sigmoid score to a labelled RerankerScore."""
    score = raw_score

    # Label mapping depends on repair type
    repair_type = candidate.repair_type
    if repair_type == "adjacent_merge":
        if score >= 0.5:
            label, confidence = "merge", score
        else:
            label, confidence = "keep", 1.0 - score
    elif repair_type == "heading_role":
        if score >= 0.5:
            label, confidence = "not_heading", score
        else:
            label, confidence = "heading", 1.0 - score
    elif repair_type == "toc_boundary":
        if score >= 0.5:
            label, confidence = "toc", score
        else:
            label, confidence = "not_toc", 1.0 - score
    elif repair_type == "front_matter_boundary":
        if score >= 0.5:
            label, confidence = "front_matter", score
        else:
            label, confidence = "body", 1.0 - score
    elif repair_type == "chapter_boundary":
        if score >= 0.5:
            label, confidence = "jump_anomaly", score
        else:
            label, confidence = "normal", 1.0 - score
    elif repair_type == "structure_gap":
        if score >= 0.5:
            label, confidence = "gap_significant", score
        else:
            label, confidence = "gap_minor", 1.0 - score
    else:
        label, confidence = "unknown", score

    return RerankerScore(
        candidate_id=candidate.candidate_id,
        score=round(score, 4),
        label=label,
        confidence=round(confidence, 4),
    )
