"""TOC-first outline extraction for textbook PDFs.

The resulting OutlineItem list is the authoritative source for chapter and
section assignment. LLM headings are only fallback signals when this extractor
cannot build a usable outline.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from .parsed_document import OutlineItem

logger = logging.getLogger("litmesh.toc")

_TOC_TITLE_RE = re.compile(r"^\s*目\s*录\s*$")
_DOT_ENTRY_RE = re.compile(r"^(?P<title>.+?)[\.\u2026·•]{2,}\s*(?P<page>\d{1,4})\s*$")
_SPACE_ENTRY_RE = re.compile(r"^(?P<title>第\s*[一二三四五六七八九十\d]+\s*[章节].+?)\s+(?P<page>\d{1,4})\s*$")
_CHAPTER_RE = re.compile(r"第\s*([一二三四五六七八九十\d]+)\s*章")
_SECTION_RE = re.compile(r"第\s*([一二三四五六七八九十\d]+)\s*节")


class TOCExtractor:
    """Extract and align a document outline from page-level text."""

    def __init__(self, paper_id: str = "", audit_dir: str = ""):
        self.paper_id = paper_id
        self._audit_path = None
        if paper_id and audit_dir:
            self._audit_path = Path(audit_dir) / f"{paper_id}.jsonl"
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)

    def extract(self, pages: list[dict], parser_outline: Iterable[OutlineItem] | None = None) -> tuple[list[OutlineItem], dict]:
        logger.info("toc_extract_start pages=%d", len(pages))
        parser_items = list(parser_outline or [])
        if parser_items:
            outline = self._from_parser_outline(parser_items)
            source = "parser_outline"
        else:
            outline = self._from_text_toc(pages)
            source = "text_toc" if outline else ""

        for item in outline:
            item.source = item.source or source
            if not item.normalized_title:
                item.normalized_title = normalize_title(item.title)
            self._write_audit("toc_entry_detected", **asdict(item))

        outline, meta = self._align_body_pages(outline, pages)
        meta["toc_source"] = source or ("none" if not outline else "unknown")
        meta["toc_entry_count"] = len(outline)
        meta["toc_page_count"] = len({i.toc_page for i in outline if i.toc_page})
        logger.info(
            "toc_extract_done entries=%d toc_pages=%d source=%s",
            meta["toc_entry_count"], meta["toc_page_count"], meta["toc_source"],
        )
        logger.info(
            "toc_alignment_done aligned=%d unaligned=%d offset=%s confidence=%.2f",
            meta.get("toc_aligned_entries", 0),
            meta.get("toc_unaligned_entries", 0),
            meta.get("toc_printed_page_offset", 0),
            meta.get("toc_alignment_confidence", 0.0),
        )
        return outline, meta

    @staticmethod
    def _from_parser_outline(items: list[OutlineItem]) -> list[OutlineItem]:
        outline = []
        for idx, item in enumerate(items, start=1):
            title = clean_title(item.title)
            if not title:
                continue
            outline.append(OutlineItem(
                title=title,
                level=max(1, item.level or infer_level(title)),
                page=item.page,
                element_id=item.element_id or f"parser_outline_{idx}",
                toc_page=item.toc_page,
                printed_page=item.printed_page or item.page,
                body_page=item.body_page or item.page,
                normalized_title=normalize_title(title),
                confidence=item.confidence or 0.9,
                source=item.source or "parser_outline",
            ))
        return outline

    def _from_text_toc(self, pages: list[dict]) -> list[OutlineItem]:
        toc_pages = self._find_toc_pages(pages)
        entries: list[OutlineItem] = []
        seen = set()
        for page in toc_pages:
            page_num = int(page.get("page_num", 0))
            for line in page.get("text", "").splitlines():
                item = parse_toc_line(line, page_num, len(entries) + 1)
                if not item:
                    continue
                key = (item.level, item.normalized_title, item.printed_page)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(item)
        return entries

    @staticmethod
    def _find_toc_pages(pages: list[dict]) -> list[dict]:
        candidates = []
        started = False
        for page in pages[:20]:
            text = page.get("text", "")
            lines = [l.strip() for l in text.splitlines() if l.strip()]
            dot_entries = sum(1 for l in lines if _DOT_ENTRY_RE.match(l) or _SPACE_ENTRY_RE.match(l))
            has_toc_title = any(_TOC_TITLE_RE.match(l) for l in lines)
            if has_toc_title or dot_entries >= 2:
                started = True
                candidates.append(page)
                continue
            if started and dot_entries >= 1:
                candidates.append(page)
                continue
            if started and dot_entries == 0:
                break
        return candidates

    def _align_body_pages(self, outline: list[OutlineItem], pages: list[dict]) -> tuple[list[OutlineItem], dict]:
        if not outline:
            return [], {
                "toc_alignment_confidence": 0.0,
                "toc_printed_page_offset": 0,
                "toc_unaligned_entries": 0,
                "toc_aligned_entries": 0,
            }

        offsets = []
        aligned = 0
        for item in outline:
            body_page = find_body_page(item, pages)
            if body_page:
                item.body_page = body_page
                item.page = body_page
                offsets.append(body_page - item.printed_page)
                aligned += 1
                item.confidence = max(item.confidence, 0.9)
                self._write_audit("toc_entry_aligned", **asdict(item))
            else:
                self._write_audit("toc_entry_unaligned", **asdict(item))

        offset = Counter(offsets).most_common(1)[0][0] if offsets else 0
        if offset:
            for item in outline:
                if not item.body_page and item.printed_page:
                    item.body_page = max(1, item.printed_page + offset)
                    item.page = item.body_page
                    item.confidence = max(item.confidence, 0.7)

        inferred = sum(1 for item in outline if item.body_page) - aligned
        unaligned = sum(1 for item in outline if not item.body_page)
        confidence = (aligned + inferred * 0.7) / max(len(outline), 1)
        return sorted(outline, key=lambda i: (i.body_page or 10**9, i.level, i.title)), {
            "toc_alignment_confidence": round(confidence, 3),
            "toc_printed_page_offset": offset,
            "toc_unaligned_entries": unaligned,
            "toc_aligned_entries": aligned,
        }

    def _write_audit(self, event: str, **kwargs):
        if not self._audit_path:
            return
        record = {
            "event": event,
            "paper_id": self.paper_id,
            "timestamp": time.time(),
            **kwargs,
        }
        try:
            with open(self._audit_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass


def parse_toc_line(line: str, toc_page: int, idx: int = 1) -> OutlineItem | None:
    raw = line.strip()
    if not raw:
        return None
    match = _DOT_ENTRY_RE.match(raw) or _SPACE_ENTRY_RE.match(raw)
    if not match:
        return None
    title = clean_title(match.group("title"))
    if not title or not (_CHAPTER_RE.search(title) or _SECTION_RE.search(title)):
        return None
    printed_page = int(match.group("page"))
    return OutlineItem(
        title=title,
        level=infer_level(title),
        page=0,
        element_id=f"toc_{toc_page}_{idx}",
        toc_page=toc_page,
        printed_page=printed_page,
        body_page=0,
        normalized_title=normalize_title(title),
        confidence=0.8,
        source="text_toc",
    )


def infer_level(title: str) -> int:
    if _CHAPTER_RE.search(title):
        return 1
    if _SECTION_RE.search(title):
        return 2
    return 2


def clean_title(title: str) -> str:
    title = title.replace("\u3000", " ")
    title = re.sub(r"[\.\u2026·•]{2,}.*$", "", title)
    title = re.sub(r"\s+", " ", title).strip()
    return title


def normalize_title(title: str) -> str:
    return re.sub(r"\s+", "", clean_title(title)).lower()


def find_body_page(item: OutlineItem, pages: list[dict]) -> int:
    key = item.normalized_title or normalize_title(item.title)
    short_key = key[: min(len(key), 12)]
    occurrences = []
    for page in pages:
        page_num = int(page.get("page_num", 0))
        if item.toc_page and page_num <= item.toc_page:
            continue
        text = normalize_title(page.get("text", ""))
        if key and key in text:
            occurrences.append(page_num)
        elif len(short_key) >= 6 and short_key in text:
            occurrences.append(page_num)
    return occurrences[0] if occurrences else 0
