import sys
import sqlite3
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
from app.litmesh.ingestion.parsers.markdown_adapter import (
    ExternalMarkdownAdapter,
    MarkdownAdapter,
    RemoteMarkdownAdapter,
    _markdown_from_response,
)
from app.litmesh.ingestion.section_splitter import split_parsed_document, _strip_page_number
from app.litmesh.ingestion.toc_extractor import TOCExtractor, parse_toc_line
from app.litmesh.models.section import HeadingLevel, SectionBlock
from app.litmesh.storage.sqlite import LitMeshDB
from app.litmesh.structure.group_builder import GroupBuilder


def test_auto_parser_prefers_remote_markdown_when_configured(monkeypatch):
    monkeypatch.setenv("LITMESH_MINERU_API_URL", "https://example.com/file_parse")
    assert PARSER_PRIORITY == ["mineru_api", "external_markdown", "pdfplumber"]
    assert isinstance(create_parser("auto"), RemoteMarkdownAdapter)


def test_auto_parser_falls_back_to_external_markdown_without_remote_config(monkeypatch):
    monkeypatch.delenv("LITMESH_MINERU_API_URL", raising=False)
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


def test_markdown_adapter_recovers_toc_when_decorative_heading_interrupts(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "# 生物学\n\n"
        "# 必修 2\n\n"
        "## 目 录\n\n"
        "## 科学家访谈 毕生追求的“禾下乘凉梦”与袁隆平院士一席谈\n\n"
        "第1章 遗传因子的发现 ........ 1\n"
        "第1节 孟德尔的豌豆杂交实验（一） ........ 3\n\n"
        "# 第1章 遗传因子的发现\n\n"
        "正文开始。\n\n"
        "## 第1节 孟德尔的豌豆杂交实验（一）\n\n"
        "更多正文。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    toc_blocks = [e for e in parsed.elements if e.type == ElementType.TOC]
    assert toc_blocks
    titles = [item.title for item in parsed.outline]
    assert "第1章 遗传因子的发现" in titles
    assert "第1节 孟德尔的豌豆杂交实验（一）" in titles
    assert "问题探讨" not in titles


def test_markdown_outline_fallback_keeps_structural_headings_only(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "# 第1章 遗传因子的发现\n\n"
        "正文。\n\n"
        "## 问题探讨\n\n"
        "问题探讨内容。\n\n"
        "## 讨论\n\n"
        "讨论内容。\n\n"
        "## 第1节 孟德尔的豌豆杂交实验（一）\n\n"
        "更多正文。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    titles = [item.title for item in parsed.outline]
    assert titles == ["第1章 遗传因子的发现", "第1节 孟德尔的豌豆杂交实验（一）"]


def test_markdown_adapter_toc_can_parse_page_less_entries(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "## 目录\n\n"
        "第1章 走近细胞.\n"
        "第1节 细胞是生命活动的基本单位.\n"
        "第2章 组成细胞的分子. 15\n\n"
        "# 第1章 走近细胞\n\n"
        "正文。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    titles = [item.title for item in parsed.outline]
    assert "第1章 走近细胞" in titles
    assert "第1节 细胞是生命活动的基本单位" in titles
    assert "第2章 组成细胞的分子" in titles


def test_markdown_adapter_skips_page_less_toc_entries_without_body_match(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "## 目录\n\n"
        "第3节 细胞中的糖类和脂质. .23\n"
        "第1章 走近细胞 ........ 1\n\n"
        "# 第1章 走近细胞\n\n"
        "正文。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    titles = [item.title for item in parsed.outline]
    assert "第1章 走近细胞" in titles
    assert "第3节 细胞中的糖类和脂质. .23" not in titles


def test_markdown_adapter_prefers_matched_body_heading_title(tmp_path):
    md = tmp_path / "book.md"
    md.write_text(
        "## 目录\n\n"
        "第3节 细胞中的糖类和脂质. .23\n\n"
        "# 第3节 细胞中的糖类和脂质\n\n"
        "正文。\n",
        encoding="utf-8",
    )
    parsed = MarkdownAdapter().parse(str(md))
    titles = [item.title for item in parsed.outline]
    assert "第3节 细胞中的糖类和脂质" in titles
    assert "第3节 细胞中的糖类和脂质. .23" not in titles


def test_remote_markdown_response_reads_results_md_content():
    class FakeResponse:
        headers = {"content-type": "application/json"}

        @staticmethod
        def json():
            return {
                "results": {
                    "book.pdf": {
                        "md_content": "# 第一章\n\n正文内容"
                    }
                }
            }

    markdown = _markdown_from_response(FakeResponse())
    assert markdown == "# 第一章\n\n正文内容"


def test_auto_parser_does_not_replace_pdfplumber_with_sparse_pymupdf(monkeypatch):
    assert "pymupdf_blocks" not in parsers.PARSER_PRIORITY


def test_parse_toc_line_with_dot_leaders():
    item = parse_toc_line("第1章 走近细胞 ........ 1", toc_page=4)
    assert item is not None
    assert item.title == "第1章 走近细胞"
    assert item.level == 1
    assert item.printed_page == 1


def test_parse_toc_line_without_page_number():
    item = parse_toc_line("第1章 走近细胞.", toc_page=4)
    assert item is not None
    assert item.title == "第1章 走近细胞"
    assert item.level == 1
    assert item.printed_page == 0


def test_parse_toc_line_strips_trailing_dot_page_noise():
    item = parse_toc_line("第3节 细胞中的糖类和脂质. .23", toc_page=4)
    assert item is not None
    assert item.title == "第3节 细胞中的糖类和脂质"


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


def test_strip_page_number_preserves_year_prefixes():
    assert _strip_page_number("2007 年 1 月，印度共产党召开会议。").startswith("2007 年 1 月")
    assert _strip_page_number("1949 年，中国新民主主义革命胜利后。").startswith("1949 年")
    assert _strip_page_number("21 世纪以来，研究不断推进。").startswith("21 世纪以来")


def test_split_parsed_document_preserves_leading_year_in_raw_text():
    parsed = ParsedDocument(
        pages=[{"page_num": 1, "text": "1949 年，中国新民主主义革命胜利后。"}],
        parser_name="mineru_api",
        elements=[
            ParsedElement(
                element_id="p1",
                type=ElementType.PARAGRAPH,
                text="1949 年，中国新民主主义革命胜利后，中国共产党在毛泽东的领导下开始建设新中国。",
                page_start=1,
                order_index=1,
                confidence=0.95,
                role="body",
            )
        ],
    )
    sections = split_parsed_document(parsed, "paper_year", "graph_year")
    assert len(sections) == 1
    assert sections[0].raw_text.startswith("1949 年，中国新民主主义革命胜利后")


def test_order_only_outline_refines_late_toc_entries_from_section_text():
    parsed = ParsedDocument(
        pages=[{"page_num": 1, "text": "markdown-like document"}],
        parser_name="mineru_api",
        quality_report=QualityReport(parser_name="mineru_api"),
        outline=[
            OutlineItem(
                title="第5章 细胞的能量供应和利用",
                level=1,
                page=1,
                body_page=100,
                normalized_title="第5章细胞的能量供应和利用",
                confidence=0.9,
                source="mineru_api",
            ),
            OutlineItem(
                title="第1节 降低化学反应活化能的酶",
                level=2,
                page=1,
                body_page=110,
                normalized_title="第1节降低化学反应活化能的酶",
                confidence=0.9,
                source="mineru_api",
            ),
            OutlineItem(
                title="第2节 细胞的能量“货币”ATP",
                level=2,
                page=1,
                body_page=900,
                normalized_title="第2节细胞的能量货币atp",
                confidence=0.9,
                source="mineru_api",
            ),
            OutlineItem(
                title="第6章 细胞的生命历程",
                level=1,
                page=1,
                body_page=1600,
                normalized_title="第6章细胞的生命历程",
                confidence=0.9,
                source="mineru_api",
            ),
            OutlineItem(
                title="第1节 细胞的增殖",
                level=2,
                page=1,
                body_page=1700,
                normalized_title="第1节细胞的增殖",
                confidence=0.9,
                source="mineru_api",
            ),
        ],
        elements=[
            ParsedElement(
                element_id="p1",
                type=ElementType.PARAGRAPH,
                text="酶能够降低化学反应的活化能，从而显著提高细胞代谢效率。",
                page_start=1,
                order_index=120,
                confidence=0.95,
                role="body",
            ),
            ParsedElement(
                element_id="p2",
                type=ElementType.PARAGRAPH,
                text="ATP是细胞生命活动的直接能源物质，ATP与ADP可以相互转化。",
                page_start=1,
                order_index=240,
                confidence=0.95,
                role="body",
            ),
            ParsedElement(
                element_id="p3",
                type=ElementType.PARAGRAPH,
                text="细胞通过分裂进行增殖，这是个体生长、发育和组织更新的重要基础。",
                page_start=1,
                order_index=360,
                confidence=0.95,
                role="body",
            ),
        ],
    )

    sections = split_parsed_document(parsed, "paper_order_fix", "graph_order_fix")

    assert len(sections) == 3
    assert sections[0].toc_anchor_title == "第1节 降低化学反应活化能的酶"
    assert sections[1].toc_anchor_title == "第2节 细胞的能量“货币”ATP"
    assert sections[2].heading_path == ["第6章 细胞的生命历程", "第1节 细胞的增殖"]
    assert sections[2].chapter_index == 2


def test_group_builder_accepts_outlineitem_dataclasses_for_toc_title():
    section = SectionBlock(
        graph_id="graph_test",
        paper_id="paper_test",
        heading="第1章 走近细胞",
        heading_path=["第1章 走近细胞"],
        heading_level=HeadingLevel.CHAPTER,
        raw_text="细胞是生命活动的基本单位。",
        display_title="走近细胞",
        toc_anchor_title="第1章 走近细胞",
        global_order_index=1,
    )
    outline = [
        OutlineItem(
            title="第1章 走近细胞",
            level=1,
            page=10,
            body_page=10,
            normalized_title="第1章走近细胞",
            confidence=0.9,
            source="text_toc",
        )
    ]

    groups = GroupBuilder().build(
        [section],
        paper_id="paper_test",
        graph_id="graph_test",
        outline_nodes=outline,
    )

    assert groups
    assert groups[0].structure_title == "第1章 走近细胞"
    assert section.structure_title == "第1章 走近细胞"


def test_init_schema_adds_toc_anchor_columns_for_legacy_db(tmp_path):
    db_path = tmp_path / "legacy.db"
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE section_blocks (
            section_id TEXT PRIMARY KEY,
            graph_id TEXT NOT NULL,
            paper_id TEXT NOT NULL,
            heading TEXT NOT NULL DEFAULT '',
            heading_path TEXT NOT NULL DEFAULT '[]',
            heading_level TEXT NOT NULL DEFAULT 'section',
            raw_text TEXT NOT NULL DEFAULT '',
            summary TEXT NOT NULL DEFAULT '',
            page_start INTEGER,
            page_end INTEGER,
            concept_keys TEXT NOT NULL DEFAULT '[]',
            parent_section_id TEXT,
            prev_section_id TEXT,
            next_section_id TEXT,
            content_hash TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now'))
        );
        """
    )
    conn.commit()
    conn.close()

    db = LitMeshDB(str(db_path))
    db.connect()
    db.init_schema()
    cols = {
        row["name"] for row in db.conn.execute("PRAGMA table_info(section_blocks)").fetchall()
    }
    db.close()

    assert "toc_anchor_id" in cols
    assert "toc_anchor_title" in cols
