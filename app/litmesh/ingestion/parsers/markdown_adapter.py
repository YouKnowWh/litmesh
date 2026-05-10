"""
Markdown parser adapters for outsourced PDF structure recovery.

LitMesh should not spend its complexity budget on PDF layout reconstruction.
These adapters consume Markdown produced by tools such as MinerU, Marker, or
Docling, then convert headings and paragraphs into ParsedDocument.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from ..parsed_document import (
    ElementType,
    OutlineItem,
    ParsedDocument,
    ParsedElement,
    QualityReport,
)


class MarkdownAdapter:
    """Parse an existing Markdown file or sidecar Markdown for a PDF."""

    def __init__(self, parser_name: str = "markdown"):
        self.name = parser_name
        self.progress_callback = None

    def parse(self, path: str) -> ParsedDocument:
        source = Path(path)
        md_path = _find_markdown_source(source)
        if not md_path:
            raise ImportError(f"No markdown sidecar found for {source}")
        markdown = md_path.read_text(encoding="utf-8", errors="ignore")
        return _document_from_markdown(markdown, source, md_path, self.name)


class ExternalMarkdownAdapter:
    """Run a configured command to convert PDF to Markdown, then parse it."""

    def __init__(self, parser_name: str = "external_markdown"):
        self.name = parser_name
        self.progress_callback = None

    def parse(self, pdf_path: str) -> ParsedDocument:
        source = Path(pdf_path)
        if source.suffix.lower() in {".md", ".markdown"}:
            return MarkdownAdapter(self.name).parse(str(source))

        existing = _find_markdown_source(source)
        if existing:
            markdown = existing.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, existing, self.name)

        command = _configured_command(source)
        if not command:
            raise ImportError("No markdown sidecar or LITMESH_MARKDOWN_COMMAND configured")

        with tempfile.TemporaryDirectory(prefix="litmesh_md_") as tmp:
            out_dir = Path(tmp)
            rendered = [
                part.format(pdf=str(source), output=str(out_dir), stem=source.stem)
                for part in command
            ]
            subprocess.run(rendered, check=True, capture_output=True, text=True, timeout=900)
            md_path = _largest_markdown(out_dir)
            if not md_path:
                raise RuntimeError("External markdown command produced no .md file")
            markdown = md_path.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, md_path, self.name)


class RemoteMarkdownAdapter:
    """Call a remote PDF-to-Markdown API and parse the returned Markdown."""

    def __init__(self, parser_name: str = "mineru_api"):
        self.name = parser_name
        self.progress_callback = None

    def parse(self, pdf_path: str) -> ParsedDocument:
        source = Path(pdf_path)
        existing = _find_markdown_source(source)
        if existing:
            markdown = existing.read_text(encoding="utf-8", errors="ignore")
            if self.progress_callback:
                self.progress_callback(1, 1)
            return _document_from_markdown(markdown, source, existing, self.name)

        endpoint = os.getenv("LITMESH_MINERU_API_URL", "").strip()
        if not endpoint:
            raise ImportError("LITMESH_MINERU_API_URL is not configured")

        import httpx
        import logging
        import threading
        logger = logging.getLogger("litmesh.parser")

        file_size = source.stat().st_size
        logger.info("mineru_api start: file=%s size=%dKB", source.name, file_size // 1024)

        # Heartbeat thread: report elapsed time while waiting for the API
        heartbeat_stop = threading.Event()
        heartbeat_start = [0.0]  # mutable closure

        def _heartbeat():
            import time as _time
            heartbeat_start[0] = _time.monotonic()
            interval = 15.0  # report every 15 seconds
            while not heartbeat_stop.is_set():
                heartbeat_stop.wait(interval)
                if not heartbeat_stop.is_set() and self.progress_callback:
                    elapsed = int(_time.monotonic() - heartbeat_start[0])
                    # Report as a page-like progress: -elapsed means "waiting, N seconds"
                    self.progress_callback(-1, elapsed)

        heartbeat = threading.Thread(target=_heartbeat, daemon=True)
        heartbeat.start()

        try:
            timeout = float(os.getenv("LITMESH_MINERU_API_TIMEOUT", "1800"))
            with source.open("rb") as f:
                response = httpx.post(
                    endpoint,
                    files={"file": (source.name, f, "application/pdf")},
                    timeout=timeout,
                )
            response.raise_for_status()
            dt = time.monotonic() - heartbeat_start[0] if heartbeat_start[0] else 0
            logger.info("mineru_api done: file=%s time=%.1fs", source.name, dt)
            if self.progress_callback:
                self.progress_callback(1, 1)
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=1)

        markdown = _markdown_from_response(response)
        if not markdown.strip():
            raise RuntimeError("Remote MinerU API returned empty markdown")
        return _document_from_markdown(markdown, source, Path("<mineru_api>"), self.name)


class MinerUMarkdownAdapter(ExternalMarkdownAdapter):
    """Use MinerU/magic-pdf when available, otherwise consume sidecar Markdown."""

    def __init__(self):
        super().__init__("mineru_markdown")

    def parse(self, pdf_path: str) -> ParsedDocument:
        source = Path(pdf_path)
        existing = _find_markdown_source(source)
        if existing:
            markdown = existing.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, existing, self.name)

        binary = shutil.which("magic-pdf") or shutil.which("mineru")
        if not binary:
            raise ImportError("MinerU/magic-pdf command not found and no markdown sidecar exists")

        with tempfile.TemporaryDirectory(prefix="litmesh_mineru_") as tmp:
            out_dir = Path(tmp)
            if Path(binary).name == "magic-pdf":
                cmd = [binary, "-p", str(source), "-o", str(out_dir), "-m", "auto"]
            else:
                cmd = [binary, str(source), "-o", str(out_dir)]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
            md_path = _largest_markdown(out_dir)
            if not md_path:
                raise RuntimeError("MinerU produced no .md file")
            markdown = md_path.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, md_path, self.name)


class MarkerMarkdownAdapter(ExternalMarkdownAdapter):
    """Use Marker when available, otherwise consume sidecar Markdown."""

    def __init__(self):
        super().__init__("marker_markdown")

    def parse(self, pdf_path: str) -> ParsedDocument:
        source = Path(pdf_path)
        existing = _find_markdown_source(source)
        if existing:
            markdown = existing.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, existing, self.name)

        binary = shutil.which("marker_single") or shutil.which("marker")
        if not binary:
            raise ImportError("Marker command not found and no markdown sidecar exists")

        with tempfile.TemporaryDirectory(prefix="litmesh_marker_") as tmp:
            out_dir = Path(tmp)
            cmd = [binary, str(source), "--output_dir", str(out_dir), "--output_format", "markdown"]
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=1800)
            md_path = _largest_markdown(out_dir)
            if not md_path:
                raise RuntimeError("Marker produced no .md file")
            markdown = md_path.read_text(encoding="utf-8", errors="ignore")
            return _document_from_markdown(markdown, source, md_path, self.name)


def _find_markdown_source(source: Path) -> Path | None:
    if source.suffix.lower() in {".md", ".markdown"} and source.exists():
        return source
    candidates = [
        source.with_suffix(".md"),
        source.with_suffix(".markdown"),
        Path("data/parsed_markdown") / f"{source.stem}.md",
        Path("data/parsed_markdown") / f"{source.stem}.markdown",
    ]
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _configured_command(source: Path) -> list[str]:
    raw = os.getenv("LITMESH_MARKDOWN_COMMAND", "").strip()
    if not raw:
        return []
    return raw.split()


def _largest_markdown(root: Path) -> Path | None:
    files = [p for p in root.rglob("*") if p.suffix.lower() in {".md", ".markdown"}]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_size)


def _markdown_from_response(response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    if "application/json" in content_type:
        data = response.json()
        for key in ("markdown", "md", "content", "text"):
            value = data.get(key)
            if isinstance(value, str):
                return value
        raise RuntimeError("Remote markdown API JSON has no markdown/md/content/text field")
    return response.text


def _document_from_markdown(
    markdown: str,
    source: Path,
    md_path: Path,
    parser_name: str,
) -> ParsedDocument:
    elements = _elements_from_markdown(markdown)
    outline = [
        OutlineItem(
            title=e.text,
            level=max(1, e.level),
            page=e.page_start,
            body_page=e.page_start,
            element_id=e.element_id,
            confidence=e.confidence,
            source=parser_name,
        )
        for e in elements
        if e.type == ElementType.HEADING
    ]
    paragraphs = [e for e in elements if e.type == ElementType.PARAGRAPH]
    headings = [e for e in elements if e.type == ElementType.HEADING]
    avg_len = sum(len(e.text) for e in paragraphs) / max(len(paragraphs), 1)
    quality = QualityReport(
        parser_name=parser_name,
        parser_version="markdown",
        segmenter_name="markdown_structure",
        total_elements=len(elements),
        paragraph_count=len(paragraphs),
        heading_count=len(headings),
        toc_detected=bool(outline),
        toc_entry_count=len(outline),
        toc_source=parser_name,
        toc_alignment_confidence=1.0 if outline else 0.0,
        outline=[item.__dict__ for item in outline],
        average_paragraph_length=round(avg_len, 1),
        needs_structure_review=len(paragraphs) < 10,
        warnings=[] if len(paragraphs) >= 10 else ["markdown produced few paragraphs"],
    )
    return ParsedDocument(
        pages=[{"page_num": 1, "text": _strip_markdown(markdown)}],
        elements=elements,
        outline=outline,
        metadata={"filename": source.name, "markdown_source": str(md_path)},
        parser_name=parser_name,
        parser_version="markdown",
        quality_report=quality,
        full_text=_strip_markdown(markdown),
    )


def _elements_from_markdown(markdown: str) -> list[ParsedElement]:
    elements: list[ParsedElement] = []
    paragraph_lines: list[str] = []
    order = 0
    current_heading = ""

    def flush_paragraph():
        nonlocal order, paragraph_lines
        text = _clean_text("\n".join(paragraph_lines))
        paragraph_lines = []
        if not text or _is_noise_line(text):
            return
        order += 1
        is_toc = _is_toc_heading(current_heading) and _looks_like_toc_block(text)
        elements.append(ParsedElement(
            element_id=f"markdown_elem_{order}",
            type=ElementType.TOC if is_toc else ElementType.PARAGRAPH,
            text=text,
            page_start=1,
            page_end=1,
            order_index=order,
            confidence=0.9,
            role="toc" if is_toc else "body",
        ))

    in_fence = False
    for raw_line in markdown.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            paragraph_lines.append(line)
            continue
        if not line:
            flush_paragraph()
            continue
        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            text = _clean_text(heading.group(2))
            if text:
                current_heading = text
                order += 1
                elements.append(ParsedElement(
                    element_id=f"markdown_elem_{order}",
                    type=ElementType.HEADING,
                    text=text,
                    page_start=1,
                    page_end=1,
                    level=len(heading.group(1)),
                    order_index=order,
                    confidence=0.95,
                    role="heading",
                ))
            continue
        paragraph_lines.append(line)

    flush_paragraph()
    return elements


def _clean_text(text: str) -> str:
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\u3000", " ")
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _strip_markdown(markdown: str) -> str:
    lines = []
    for line in markdown.splitlines():
        line = re.sub(r"^#{1,6}\s+", "", line.strip())
        line = _clean_text(line)
        if line:
            lines.append(line)
    return "\n".join(lines)


def _is_noise_line(text: str) -> bool:
    stripped = text.strip()
    if len(stripped) < 8:
        return True
    if re.fullmatch(r"[-*_`~\s]+", stripped):
        return True
    if re.fullmatch(r"\d{1,4}", stripped):
        return True
    return False


def _is_toc_heading(text: str) -> bool:
    compact = re.sub(r"\s+", "", text)
    return compact in {"目录", "目錄"}


def _looks_like_toc_block(text: str) -> bool:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return False
    hits = 0
    for line in lines:
        if re.search(r"[\.\u2026·•]{2,}\s*\d{1,4}\s*$", line):
            hits += 1
        elif re.search(r"第\s*[一二三四五六七八九十\d]+\s*[章节].+\s+\d{1,4}\s*$", line):
            hits += 1
    return hits >= 1
