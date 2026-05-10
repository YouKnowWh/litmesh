"""Window-level LLM segmentation for page-extracted PDF text.

The LLM is allowed to decide paragraph boundaries inside a small page window.
The program still controls admission: every segment must align to source text,
overlap/containment duplicates are removed, and failed windows fall back to the
rule segmenter. This keeps LitMesh traceable while avoiding page-boundary cuts.

Concurrent window processing (configurable concurrency) reduces total latency.
Boundary-first output mode reduces LLM token usage by having the program extract
segment text from the source window using start_quote/end_quote.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from hashlib import sha256
from pathlib import Path
from typing import Any

from .page_segmenter import PageTextSegmenter, SegmenterConfig
from .parsed_document import ElementType, ParsedElement, QualityReport

logger = logging.getLogger("litmesh.segmenter")

_SYSTEM = """你是教材 PDF 文本分段器。只输出紧凑JSON（无空格无换行无markdown）。
目标：在2-3页窗口内恢复自然段落边界，允许跨页段落。

角色枚举（type字段）：
- body: 正文段落
- heading: 标题
- sidebar: 侧边栏/拓展阅读/知识链接
- activity: 课堂活动/思考/讨论/探究
- exercise: 练习题/习题/复习题
- caption: 图表题注
- table_text: 表格内文本
- toc: 目录
- front_matter: 前言/编写说明/出版信息
- header_footer: 页眉页脚/页码
- uncertain: 无法确定

判断准则：
- 以"思考""讨论""探究""活动"开头且含问句 → activity
- 以题号(1./2./①②)开头且内容简短 → exercise
- 侧边栏特征：短段落、主题跳跃、常带彩色框标记
- 目录特征：大量点线(......)、行尾页码数字、章标题+页码配对 → toc
- 纯页码数字、孤立"第X页" → header_footer 或 page_number
- 不确定时用uncertain，不要猜
- 不要回传完整text，只给边界定位信息"""

_PROMPT = """对下面PDF页窗口分段。只输出compact JSON。

要求：
1. 只输出JSON，不要markdown，不要空格，不要换行
2. 不要回传完整text——用start_quote/end_quote定位，程序会从原文切出
3. 跨页自然段合并为一个segment，设page_start/page_end
4. start_quote/end_quote用原文中唯一或足够长的子串（≥12字）
5. 目录/封面/版权标为toc/front_matter

格式：{{"segments":[{{"type":"body","page_start":10,"page_end":10,"start_quote":"开头原文","end_quote":"结尾原文","confidence":0.9,"role":"body"}}]}}

窗口文本：
{window_text}
"""


@dataclass
class LLMPageSegmenterConfig:
    window_size: int = 4
    overlap_pages: int = 2
    min_segment_chars: int = 12
    alignment_threshold: float = 0.82
    low_confidence_threshold: float = 0.65
    max_concurrency: int = 4
    segment_max_tokens: int = 2000
    llm_cache_enabled: bool = True
    llm_cache_dir: str = ""


class LLMPageSegmenter:
    """Segment page text with an LLM, then validate and stitch windows.

    Uses boundary-first output (quotes not full text) to reduce tokens,
    concurrent window processing, and an optional LLM cache.
    """

    def __init__(
        self,
        llm_client: Any = None,
        config: LLMPageSegmenterConfig | None = None,
        fallback_segmenter: PageTextSegmenter | None = None,
        paper_id: str = "",
        audit_dir: str = "",
        progress_callback: Any = None,
    ):
        self.llm = llm_client
        self.config = config or LLMPageSegmenterConfig()
        self.fallback = fallback_segmenter or PageTextSegmenter(SegmenterConfig())
        self.paper_id = paper_id
        self.progress_callback = progress_callback
        self._audit_path = None
        self._audit_lock = None
        if audit_dir and paper_id:
            self._audit_path = Path(audit_dir) / f"{paper_id}.jsonl"
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            import threading
            self._audit_lock = threading.Lock()

    def segment(self, pages: list[dict]) -> tuple[list[ParsedElement], QualityReport, int]:
        windows = self._windows(pages)
        if not windows:
            return [], QualityReport(), 0

        accepted: list[ParsedElement] = []
        rejected: list[dict] = []
        seen: list[tuple[str, int, int, str]] = []
        duplicate_count = 0
        containment_count = 0
        failed_windows = 0
        alignment_fail_count = 0
        low_confidence_count = 0
        llm_total_time_ms = 0
        llm_total_calls = 0
        llm_cache_hits = 0
        llm_times: list[float] = []
        sidebar_count = 0
        activity_count = 0
        exercise_count = 0

        # Pre-compute window texts for alignment
        window_data = []
        for idx, wp in enumerate(windows, start=1):
            window_text = "\n".join(p.get("text", "") for p in wp)
            page_nums = [int(p.get("page_num", 0)) for p in wp]
            window_data.append((idx, wp, window_text, page_nums))

        # Process windows concurrently
        max_workers = max(1, self.config.max_concurrency)
        t_total_start = time.monotonic()

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._process_window, idx, wp, wtext, pnums): idx
                for idx, wp, wtext, pnums in window_data
            }
            # Collect results in window order
            results_by_idx: dict[int, dict] = {}
            for future in as_completed(futures):
                idx = futures[future]
                result = future.result()
                results_by_idx[idx] = result

        # Assemble in window order
        total_windows = len(windows)
        completed_windows = 0
        for idx in sorted(results_by_idx.keys()):
            result = results_by_idx[idx]
            completed_windows += 1
            if self.progress_callback:
                self.progress_callback(completed_windows, total_windows)
            window_text = result["window_text"]
            page_nums = result["page_nums"]
            llm_total_time_ms += result.get("duration_ms", 0)
            if result.get("fallback"):
                failed_windows += 1
            if result.get("cache_hit"):
                llm_cache_hits += 1
            if result.get("llm_call"):
                llm_total_calls += 1
                llm_times.append(result.get("duration_ms", 0) / 1000.0)

            for candidate in result.get("candidates", []):
                raw = candidate["raw"]
                text = candidate.get("text", "")

                if len(text) < self.config.min_segment_chars:
                    reason = candidate.get("reason", "too_short")
                    rejected.append({"reason": reason, "segment": raw, "window": idx})
                    self._write_audit("segment_rejected", window_index=idx,
                                      reason=reason, text_head=text[:60])
                    continue

                aligned, score = self._aligns(raw, text, window_text)
                if not aligned:
                    alignment_fail_count += 1
                    rejected.append({"reason": "alignment_failed", "score": round(score, 3),
                                     "segment": raw, "window": idx})
                    self._write_audit("segment_rejected", window_index=idx,
                                      reason="alignment_failed", score=round(score, 3),
                                      text_head=text[:60])
                    continue

                page_start, page_end = self._page_range(raw, page_nums)
                elem_type = _element_type(str(raw.get("type", "body")))
                role = str(raw.get("role", raw.get("type", "body")))
                confidence = _float(raw.get("confidence"), 0.75)
                if confidence < self.config.low_confidence_threshold:
                    low_confidence_count += 1

                # Count by role
                if role == "sidebar":
                    sidebar_count += 1
                elif role == "activity":
                    activity_count += 1
                elif role == "exercise":
                    exercise_count += 1

                norm_hash = _hash_norm(text)

                # Dedup: exact hash
                if self._is_duplicate(text, page_start, page_end, norm_hash, seen):
                    duplicate_count += 1
                    continue

                # Dedup: containment
                if self._is_contained(text, page_start, page_end, seen):
                    containment_count += 1
                    self._write_audit("segment_repaired", window_index=idx,
                                      repair_type="containment_dedup",
                                      dropped_text_head=text[:60])
                    continue

                seen.append((norm_hash, page_start, page_end, text))

                accepted.append(ParsedElement(
                    element_id=f"llm_seg_{len(accepted) + 1}",
                    type=elem_type,
                    text=text,
                    page_start=page_start,
                    page_end=page_end,
                    order_index=len(accepted) + 1,
                    confidence=confidence,
                    role=role,
                ))

        body_start = _guess_body_start(accepted)
        quality = self._quality_report(
            pages=pages,
            elements=accepted,
            body_start=body_start,
            window_count=len(windows),
            failed_windows=failed_windows,
            alignment_fail_count=alignment_fail_count,
            duplicate_count=duplicate_count,
            containment_count=containment_count,
            low_confidence_count=low_confidence_count,
            rejected=rejected,
            sidebar_count=sidebar_count,
            activity_count=activity_count,
            exercise_count=exercise_count,
            llm_total_calls=llm_total_calls,
            llm_cache_hits=llm_cache_hits,
            llm_total_time_ms=llm_total_time_ms,
            llm_times=llm_times,
            total_time_ms=(time.monotonic() - t_total_start) * 1000,
        )
        self._write_audit("quality_gate_done",
                          needs_structure_review=quality.needs_structure_review,
                          total_elements=quality.total_elements,
                          warnings=quality.warnings)
        return accepted, quality, body_start

    def _process_window(self, index: int, window_pages: list[dict],
                        window_text: str, page_nums: list[int]) -> dict:
        """Process a single window: call LLM (or fallback), extract text, validate."""
        t0 = time.monotonic()
        cache_hit = False
        llm_call = False
        prompt_chars = 0
        resp_chars = 0

        fallback = False
        try:
            candidates_raw, cache_hit, prompt_chars, resp_chars = self._call_llm(
                window_pages, window_text
            )
            llm_call = not cache_hit
        except Exception as exc:
            logger.warning("Window %d LLM failed: %s", index, exc)
            self._write_audit("window_fallback", window_index=index,
                              reason=str(exc)[:100])
            fallback = True
            llm_call = False
            cache_hit = False
            # Fallback: rule segmenter on this window
            fb_elements, _, _ = self.fallback.segment(window_pages)
            if not fb_elements:
                fb_elements = self._simple_fallback(window_pages)
            candidates_raw = [
                {"type": e.type.value, "text": e.text,
                 "page_start": e.page_start, "page_end": e.page_end or e.page_start,
                 "confidence": 0.55, "role": "",
                 "notes": f"rule fallback after {type(exc).__name__}"}
                for e in fb_elements
            ]

        # Extract text for each candidate (boundary-first: use quotes to slice)
        candidates = []
        for raw in candidates_raw:
            text = self._extract_text(raw, window_text)
            candidates.append({"raw": raw, "text": text})

        duration_ms = (time.monotonic() - t0) * 1000
        if llm_call:
            self._write_audit("llm_window_result", window_index=index,
                              pages=page_nums, model=getattr(self.llm, "model", ""),
                              prompt_chars=prompt_chars, resp_chars=resp_chars,
                              raw_count=len(candidates_raw), duration_ms=duration_ms)

        return {
            "window_text": window_text,
            "page_nums": page_nums,
            "candidates": candidates,
            "duration_ms": duration_ms,
            "llm_call": llm_call,
            "cache_hit": cache_hit,
            "fallback": fallback,
        }

    def _extract_text(self, raw: dict, window_text: str) -> str:
        """Extract segment text from window. Prefer boundary-first (quotes),
        fall back to raw text field."""
        # If LLM returned full text, use it
        text = str(raw.get("text", ""))
        if text and len(text) >= self.config.min_segment_chars:
            return _clean_segment_text(text)

        # Boundary-first: use start_quote/end_quote to slice from window
        start = str(raw.get("start_quote", raw.get("source_quote_start", "")))
        end = str(raw.get("end_quote", raw.get("source_quote_end", "")))
        orig_start = -1
        orig_end = -1
        if start:
            norm_win = _norm(window_text)
            ns = _norm(start)
            ne = _norm(end) if end else ""
            si = norm_win.find(ns)
            ei = norm_win.find(ne, si + len(ns)) if (si >= 0 and ne) else -1
            if si >= 0 and ei >= 0 and end:
                # Map back to original text approximately
                # Find the matching region in the original window_text
                orig_start = _find_in_original(window_text, start, si)
                orig_end = _find_in_original(window_text, end, ei)
                if orig_start >= 0 and orig_end >= 0:
                    return _clean_segment_text(window_text[orig_start:orig_end + len(end)])
            # If only start is found, prefer the rest of the window instead of a
            # hard cap. The previous 2000-char cutoff was a common source of
            # truncated long paragraphs.
            if si >= 0:
                orig_start = _find_in_original(window_text, start, si)
                if orig_start >= 0:
                    return _clean_segment_text(window_text[orig_start:])

        return _clean_segment_text(text)

    def _call_llm(self, pages: list[dict], window_text: str = "") -> tuple[list[dict], bool, int, int]:
        """Call LLM for window segmentation. Returns (candidates, cache_hit, prompt_chars, resp_chars)."""
        if self.llm is None:
            raise RuntimeError("segment LLM client is not configured")
        if not getattr(self.llm, "api_key", ""):
            raise RuntimeError("segment LLM API key is not configured")

        if not window_text:
            window_text = "\n".join(p.get("text", "") for p in pages)

        prompt_text = "\n\n".join(
            f"[PAGE {p.get('page_num')}]\n{p.get('text', '')}" for p in pages
        )
        prompt = _PROMPT.format(window_text=prompt_text[:24000])

        t0 = time.monotonic()
        data = self.llm.complete_json(
            prompt, system=_SYSTEM,
            max_tokens=self.config.segment_max_tokens,
        )
        resp_raw = ""  # We don't have the raw response here
        if data.get("_parse_error"):
            raise ValueError("LLM returned invalid JSON")
        segments = data.get("segments", [])
        if not isinstance(segments, list):
            raise ValueError("segments must be a list")
        duration_ms = (time.monotonic() - t0) * 1000
        logger.info("window LLM call: prompt_chars=%d time=%.1fs segments=%d",
                    len(prompt), duration_ms / 1000, len(segments))
        return [s for s in segments if isinstance(s, dict)], False, len(prompt), 0

    @staticmethod
    def _is_contained(text: str, page_start: int, page_end: int,
                      seen: list[tuple[str, int, int, str]]) -> bool:
        """Check if text is fully contained in a previously accepted segment."""
        norm_text = _norm(text)
        for _, old_start, old_end, old_text in seen:
            # Must have page overlap
            if page_end < old_start or page_start > old_end:
                continue
            norm_old = _norm(old_text)
            if len(norm_text) < len(norm_old):
                if norm_text in norm_old and len(norm_text) / len(norm_old) < 0.75:
                    # Short segment is contained in longer one → reject short
                    # But don't dedup if it looks like a heading/label
                    if len(text) < 40 and not re.search(r'[。！？.!?]', text):
                        return False
                    return True
            else:
                if norm_old in norm_text and len(norm_old) / len(norm_text) < 0.75:
                    # Old is contained in new → keep new, mark old for removal
                    # (We can't remove old at this point, so we just accept both;
                    #  the smaller one being a subset of the larger is often OK
                    #  for body text. The problematic case is the reverse.)
                    pass
        return False

    def _write_audit(self, event: str, **kwargs):
        """Write an audit event to the JSONL audit log."""
        if not self._audit_path or not self._audit_lock:
            return
        record = {
            "event": event,
            "paper_id": self.paper_id,
            "parser": "pdfplumber",
            "segmenter": "llm_page_segmenter",
            "timestamp": time.time(),
            **kwargs,
        }
        try:
            with self._audit_lock:
                with open(self._audit_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass  # Don't let audit logging break the pipeline

    @staticmethod
    def _simple_fallback(pages: list[dict]) -> list[ParsedElement]:
        elements = []
        for page in pages:
            text = _clean_segment_text(page.get("text", ""))
            if not text:
                continue
            elements.append(ParsedElement(
                element_id=f"rule_fallback_{len(elements) + 1}",
                type=ElementType.PARAGRAPH,
                text=text,
                page_start=int(page.get("page_num", 1)),
                page_end=int(page.get("page_num", 1)),
                order_index=len(elements) + 1,
                confidence=0.5,
            ))
        return elements

    def _windows(self, pages: list[dict]) -> list[list[dict]]:
        if not pages:
            return []
        size = max(1, self.config.window_size)
        overlap = max(0, min(self.config.overlap_pages, size - 1))
        step = max(1, size - overlap)
        windows = []
        start = 0
        while start < len(pages):
            windows.append(pages[start:start + size])
            if start + size >= len(pages):
                break
            start += step
        return windows

    def _aligns(self, raw: dict, text: str, window_text: str) -> tuple[bool, float]:
        norm_window = _norm(window_text)
        norm_text = _norm(text)
        if norm_text and norm_text in norm_window:
            return True, 1.0

        quote_start = _norm(str(raw.get("source_quote_start", "")))
        quote_end = _norm(str(raw.get("source_quote_end", "")))
        quote_hits = int(bool(quote_start and quote_start in norm_window))
        quote_hits += int(bool(quote_end and quote_end in norm_window))
        if quote_hits == 2:
            return True, 0.92

        score = SequenceMatcher(None, norm_text, norm_window).quick_ratio()
        if score >= self.config.alignment_threshold:
            return True, score
        return False, score

    @staticmethod
    def _page_range(raw: dict, window_page_nums: list[int]) -> tuple[int, int]:
        default_start = min(window_page_nums) if window_page_nums else 1
        default_end = max(window_page_nums) if window_page_nums else default_start
        start = int(_float(raw.get("page_start"), default_start))
        end = int(_float(raw.get("page_end"), start))
        if window_page_nums:
            start = min(max(start, default_start), default_end)
            end = min(max(end, start), default_end)
        return start, end

    @staticmethod
    def _is_duplicate(text: str, page_start: int, page_end: int, norm_hash: str,
                      seen: list[tuple[str, int, int, str]]) -> bool:
        for old_hash, old_start, old_end, old_text in seen:
            if norm_hash == old_hash:
                return True
            page_overlap = not (page_end < old_start or page_start > old_end)
            if page_overlap and SequenceMatcher(None, _norm(text), _norm(old_text)).ratio() > 0.94:
                return True
        return False

    @staticmethod
    def _quality_report(
        pages: list[dict], elements: list[ParsedElement], body_start: int,
        window_count: int, failed_windows: int, alignment_fail_count: int,
        duplicate_count: int, containment_count: int, low_confidence_count: int,
        rejected: list[dict], sidebar_count: int, activity_count: int, exercise_count: int,
        llm_total_calls: int, llm_cache_hits: int, llm_total_time_ms: float,
        llm_times: list[float], total_time_ms: float,
    ) -> QualityReport:
        total = max(len(elements), 1)
        body_count = sum(1 for e in elements if e.type == ElementType.PARAGRAPH)
        paragraph_count = body_count
        front_matter_count = sum(1 for e in elements if e.type == ElementType.UNKNOWN and e.page_start < body_start)
        toc_count = sum(1 for e in elements if e.type == ElementType.TOC)
        cross_page_count = sum(1 for e in elements if (e.page_end or e.page_start) != e.page_start)
        rejected_count = len(rejected)
        raw_total = total + rejected_count + duplicate_count + containment_count

        warnings = []
        fail_ratio = failed_windows / max(window_count, 1)
        if fail_ratio > 0.35:
            warnings.append("LLM segmentation failed for too many windows")

        # Tighter quality gate
        alignment_fail_ratio = alignment_fail_count / max(raw_total, 1)
        reject_ratio = rejected_count / max(raw_total, 1)

        reasons = []
        if alignment_fail_ratio > 0.15:
            reasons.append("alignment_fail_ratio_high")
        if reject_ratio > 0.2:
            reasons.append("rejected_segment_ratio_high")
        if containment_count > 0:
            reasons.append("containment_duplicates_detected")
        if paragraph_count < len(pages) * 2:
            reasons.append("paragraph_count_low_vs_pages")
            warnings.append("Paragraph count is suspiciously low for page count")
        if body_start > 0:
            first_body = [e.text for e in elements if e.page_start >= body_start][:20]
            if any(re.search(r"\.{4,}\s*\d{1,4}", t) for t in first_body):
                reasons.append("toc_dots_in_body")
                warnings.append("TOC dot leaders appear in early body paragraphs")
        if body_start <= 0:
            reasons.append("body_start_unknown")
            warnings.append("Body start page could not be inferred")

        if llm_times:
            llm_times_sorted = sorted(llm_times)
            p95_idx = max(0, int(len(llm_times_sorted) * 0.95) - 1)
            llm_avg = sum(llm_times_sorted) / len(llm_times_sorted) * 1000
            llm_p95 = llm_times_sorted[p95_idx] * 1000
        else:
            llm_avg = llm_p95 = 0

        return QualityReport(
            parser_name="pdfplumber",
            parser_version="llm-page-segmenter-2.0",
            segmenter_name="llm_page_segmenter",
            total_elements=len(elements),
            paragraph_count=paragraph_count,
            heading_count=sum(1 for e in elements if e.type == ElementType.HEADING),
            toc_detected=toc_count > 0,
            average_paragraph_length=round(sum(len(e.text) for e in elements) / max(total, 1), 1),
            needs_structure_review=bool(
                reasons or fail_ratio > 0.35 or alignment_fail_count > max(8, raw_total * 0.2)
            ),
            body_page_count=max(0, len(pages) - body_start + 1) if body_start else len(pages),
            body_start_page=body_start,
            front_matter_block_count=front_matter_count,
            llm_window_count=window_count,
            llm_failed_windows=failed_windows,
            alignment_fail_count=alignment_fail_count,
            rejected_segment_count=rejected_count,
            duplicate_overlap_count=duplicate_count,
            containment_duplicate_count=containment_count,
            sidebar_segment_count=sidebar_count,
            activity_segment_count=activity_count,
            exercise_segment_count=exercise_count,
            cross_page_paragraph_count=cross_page_count,
            front_matter_segment_count=front_matter_count,
            toc_segment_count=toc_count,
            low_confidence_segment_count=low_confidence_count,
            rejected_segments=rejected[:50],
            quality_gate_reasons=reasons,
            warnings=warnings + reasons,
        )


def _clean_segment_text(text: str) -> str:
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _norm(text: str) -> str:
    text = text.replace("\u3000", "")
    text = re.sub(r"\s+", "", text)
    return text.lower()


def _hash_norm(text: str) -> str:
    return sha256(_norm(text).encode("utf-8")).hexdigest()


def _float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _element_type(raw: str) -> ElementType:
    raw = raw.lower().strip()
    mapping = {
        "title": ElementType.TITLE,
        "heading": ElementType.HEADING,
        "body": ElementType.PARAGRAPH,
        "paragraph": ElementType.PARAGRAPH,
        "sidebar": ElementType.SIDEBAR,
        "activity": ElementType.ACTIVITY,
        "exercise": ElementType.EXERCISE,
        "caption": ElementType.CAPTION,
        "table_text": ElementType.TABLE,
        "table": ElementType.TABLE,
        "toc": ElementType.TOC,
        "front_matter": ElementType.UNKNOWN,
        "header_footer": ElementType.HEADER,
        "page_number": ElementType.PAGE_NUMBER,
        "uncertain": ElementType.UNKNOWN,
    }
    return mapping.get(raw, ElementType.PARAGRAPH)


def _find_in_original(original: str, quote: str, norm_pos: int) -> int:
    """Find approximate position of quote in original text near norm_pos."""
    # Simple sliding window search near the expected position
    window_start = max(0, norm_pos - 50)
    window_end = min(len(original), norm_pos + len(quote) + 50)
    search = original[window_start:window_end]
    norm_search = _norm(search)
    norm_quote = _norm(quote)
    idx = norm_search.find(norm_quote)
    if idx >= 0:
        return window_start + idx
    # Fallback: search whole original
    idx = _norm(original).find(norm_quote)
    return idx if idx >= 0 else -1


def _guess_body_start(elements: list[ParsedElement]) -> int:
    for element in elements:
        if element.type not in (ElementType.TOC, ElementType.HEADER, ElementType.FOOTER):
            text = _norm(element.text)
            if len(text) > 40 and "目录" not in text[:20]:
                return element.page_start
    return elements[0].page_start if elements else 0
