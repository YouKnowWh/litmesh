import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.litmesh.ingestion.parsed_document import (
    ElementType,
    OutlineItem,
    ParsedDocument,
    ParsedElement,
    QualityReport,
)
from app.litmesh.ingestion import parsers
from app.litmesh.ingestion.parsers import PARSER_PRIORITY, create_parser
from app.litmesh.ingestion.parsers.markdown_adapter import ExternalMarkdownAdapter
from app.litmesh.ingestion.parsers.markdown_adapter import MarkdownAdapter
from app.litmesh.ingestion.section_splitter import split_parsed_document
from app.litmesh.ingestion.toc_extractor import TOCExtractor, parse_toc_line


def test_auto_parser_uses_pdfplumber_as_source_of_truth():
    assert PARSER_PRIORITY == ["mineru_api", "external_markdown", "pdfplumber"]
    assert isinstance(create_parser("auto"), ExternalMarkdownAdapter)


def test_markdown_adapter_builds_heading_and_paragraph_elements(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "# 第一章 细胞\n\n"
        "细胞是生命活动的基本单位。这是一段正文。\n\n"
        "## 第一节 细胞结构\n\n"
        "细胞膜、细胞质和细胞核共同参与生命活动。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    assert parsed.parser_name == "markdown"
    assert parsed.quality_report.paragraph_count == 2
    assert parsed.quality_report.heading_count == 2
    assert [item.title for item in parsed.outline] == ["第一章 细胞", "第一节 细胞结构"]


def test_auto_parser_does_not_replace_pdfplumber_with_sparse_pymupdf(monkeypatch):
    assert "pymupdf_blocks" not in parsers.PARSER_PRIORITY


def test_parse_toc_line_with_dot_leaders():
    item = parse_toc_line("第1章 走近细胞 ........ 1", toc_page=4)
    assert item is not None
    assert item.title == "第1章 走近细胞"
    assert item.level == 1
    assert item.printed_page == 1


def test_toc_extractor_aligns_printed_page_offset():
    pages = [
        {"page_num": 4, "text": "目录\n第1章 走近细胞 ........ 1\n第1节 细胞是生命活动的基本单位 ........ 5"},
        {"page_num": 11, "text": "第1章 走近细胞\n正文开始。"},
        {"page_num": 15, "text": "第1节 细胞是生命活动的基本单位\n细胞相关正文。"},
    ]
    outline, meta = TOCExtractor().extract(pages)
    assert len(outline) == 2
    assert outline[0].body_page == 11
    assert outline[1].body_page == 15
    assert meta["toc_printed_page_offset"] == 10
    assert meta["toc_alignment_confidence"] == 1.0


def test_split_parsed_document_prefers_toc_outline_over_llm_heading():
    parsed = ParsedDocument(
        pages=[
            {"page_num": 4, "text": "目录\n第1章 走近细胞 ........ 1"},
            {"page_num": 11, "text": "第1章 走近细胞\n问题探讨正文。"},
        ],
        parser_name="pdfplumber",
        parser_version="test",
        quality_report=QualityReport(parser_name="pdfplumber"),
        outline=[
            OutlineItem(
                title="第1章 走近细胞",
                level=1,
                page=11,
                toc_page=4,
                printed_page=1,
                body_page=11,
                normalized_title="第1章走近细胞",
                confidence=0.9,
                source="text_toc",
            )
        ],
        elements=[
            ParsedElement(
                element_id="h_bad",
                type=ElementType.HEADING,
                text="问题探讨",
                page_start=11,
                order_index=1,
                confidence=0.7,
                role="heading",
            ),
            ParsedElement(
                element_id="p1",
                type=ElementType.PARAGRAPH,
                text="问题探讨正文足够长，用来验证目录优先时不会把问题探讨当成章节标题。",
                page_start=11,
                order_index=2,
                confidence=0.9,
                role="body",
            ),
        ],
        full_text="第1章 走近细胞\n问题探讨正文。",
    )
    sections = split_parsed_document(parsed, "paper_toc", "graph_toc")
    assert len(sections) == 1
    assert sections[0].heading_path == ["第1章 走近细胞"]
    assert sections[0].heading == "第1章 走近细胞"
    assert "P001" not in sections[0].display_title
    assert parsed.quality_report.outline_assigned_section_count == 1


def test_toc_and_front_matter_do_not_enter_body_sections():
    parsed = ParsedDocument(
        pages=[{"page_num": 1, "text": "目录\n第1章 遗传因子的发现 ........ 1"}],
        parser_name="pdfplumber",
        outline=[
            OutlineItem(
                title="第1章 遗传因子的发现",
                level=1,
                page=3,
                body_page=3,
                normalized_title="第1章遗传因子的发现",
            )
        ],
        elements=[
            ParsedElement(
                element_id="toc1",
                type=ElementType.TOC,
                text="第1章 遗传因子的发现 ........ 1",
                page_start=1,
                order_index=1,
                role="toc",
            ),
            ParsedElement(
                element_id="p1",
                type=ElementType.PARAGRAPH,
                text="孟德尔通过豌豆杂交实验提出了遗传学中的重要问题。",
                page_start=3,
                order_index=2,
                role="body",
            ),
        ],
    )
    sections = split_parsed_document(parsed, "paper_toc2", "graph_toc")
    assert len(sections) == 2
    assert sections[0].heading == "目录"
    assert sections[0].heading_path == ["目录"]
    assert sections[0].display_title == "目录"
    assert "........" not in sections[1].raw_text


def test_front_matter_is_reserved_with_front_matter_title():
    parsed = ParsedDocument(
        pages=[{"page_num": 1, "text": "前言\n这是教材前言说明。"}],
        parser_name="pdfplumber",
        elements=[
            ParsedElement(
                element_id="fm1",
                type=ElementType.UNKNOWN,
                text="前言\n这是教材前言说明，应该保留为前言结构块。",
                page_start=1,
                order_index=1,
                role="front_matter",
            ),
            ParsedElement(
                element_id="p1",
                type=ElementType.PARAGRAPH,
                text="正文段落足够长，用于验证前言不会被当成正文标题。",
                page_start=3,
                order_index=2,
                role="body",
            ),
        ],
    )
    sections = split_parsed_document(parsed, "paper_fm", "graph_toc")
    assert sections[0].heading == "前言"
    assert sections[0].heading_path == ["前言"]
    assert sections[0].display_title == "前言"


def test_false_positive_toc_role_is_not_reserved_without_toc_text():
    parsed = ParsedDocument(
        pages=[{"page_num": 1, "text": "普通正文。"}],
        parser_name="pdfplumber",
        elements=[
            ParsedElement(
                element_id="bad_toc",
                type=ElementType.PARAGRAPH,
                text="这是一段被模型误标的普通正文内容，里面没有目录点线或章节页码配对。",
                page_start=1,
                order_index=1,
                role="toc",
            )
        ],
    )
    sections = split_parsed_document(parsed, "paper_bad_toc", "graph_toc")
    assert len(sections) == 0
