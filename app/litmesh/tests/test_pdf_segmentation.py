import os

from app.litmesh.extraction.llm_config import load_all_endpoints, load_endpoint
from app.litmesh.ingestion.llm_page_segmenter import LLMPageSegmenter, LLMPageSegmenterConfig
from app.litmesh.ingestion.parsed_document import ElementType


class FakeSegmentLLM:
    model = "deepseek-chat"
    base_url = "https://api.deepseek.com/v1"
    api_key = "test-key"

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def complete_json(self, prompt, system=None, **kwargs):
        self.calls += 1
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_deepseek_defaults_are_openai_compatible(monkeypatch):
    for key in list(os.environ):
        if key.startswith("LITMESH_") or key in {
            "OPENAI_API_KEY", "OPENAI_MODEL", "OPENAI_BASE_URL",
            "DEEPSEEK_API_KEY", "ANTHROPIC_AUTH_TOKEN",
            "ANTHROPIC_MODEL", "ANTHROPIC_BASE_URL",
        }:
            monkeypatch.delenv(key, raising=False)

    endpoint = load_endpoint("DEFAULT")
    segment = load_endpoint("SEGMENT")
    all_endpoints = load_all_endpoints()

    assert endpoint.provider == "openai_compatible"
    assert endpoint.base_url == "https://api.deepseek.com/v1"
    assert endpoint.model == "deepseek-chat"
    assert segment.provider == "openai_compatible"
    assert "segment" in all_endpoints


def test_segment_role_reads_explicit_config(monkeypatch):
    monkeypatch.setenv("LITMESH_SEGMENT_PROVIDER", "openai_compatible")
    monkeypatch.setenv("LITMESH_SEGMENT_MODEL", "deepseek-chat")
    monkeypatch.setenv("LITMESH_SEGMENT_BASE_URL", "https://api.deepseek.com/v1")
    monkeypatch.setenv("LITMESH_SEGMENT_API_KEY", "ds-test")

    endpoint = load_endpoint("SEGMENT")

    assert endpoint.provider == "openai_compatible"
    assert endpoint.model == "deepseek-chat"
    assert endpoint.base_url == "https://api.deepseek.com/v1"
    assert endpoint.api_key == "ds-test"


def test_llm_segmenter_accepts_aligned_cross_page_segment():
    pages = [
        {"page_num": 1, "text": "第一段开始，介绍孟德尔实验，并在这里跨页"},
        {"page_num": 2, "text": "继续说明豌豆杂交实验的观察结果。第二段是新的内容。"},
    ]
    llm = FakeSegmentLLM([{
        "segments": [{
            "type": "paragraph",
            "text": "第一段开始，介绍孟德尔实验，并在这里跨页继续说明豌豆杂交实验的观察结果。",
            "page_start": 1,
            "page_end": 2,
            "source_quote_start": "第一段开始",
            "source_quote_end": "观察结果",
            "confidence": 0.93,
            "notes": "",
        }]
    }])

    elements, quality, _ = LLMPageSegmenter(
        llm,
        config=LLMPageSegmenterConfig(window_size=3, overlap_pages=1),
    ).segment(pages)

    assert len(elements) == 1
    assert elements[0].page_start == 1
    assert elements[0].page_end == 2
    assert quality.cross_page_paragraph_count == 1
    assert quality.segmenter_name == "llm_page_segmenter"


def test_llm_segmenter_rejects_hallucinated_text():
    pages = [{"page_num": 1, "text": "原文只讨论遗传因子，没有讨论细胞呼吸。"}]
    llm = FakeSegmentLLM([{
        "segments": [{
            "type": "paragraph",
            "text": "这段凭空加入了光合作用和能量代谢的大量内容。",
            "page_start": 1,
            "page_end": 1,
            "source_quote_start": "光合作用",
            "source_quote_end": "能量代谢",
            "confidence": 0.9,
        }]
    }])

    elements, quality, _ = LLMPageSegmenter(
        llm,
        config=LLMPageSegmenterConfig(window_size=3, overlap_pages=1),
    ).segment(pages)

    assert elements == []
    assert quality.alignment_fail_count == 1
    assert quality.rejected_segment_count == 1


def test_llm_segmenter_dedups_overlap_segments():
    pages = [
        {"page_num": 1, "text": "共同段落文本足够长，用于测试重叠窗口去重。"},
        {"page_num": 2, "text": "共同段落文本足够长，用于测试重叠窗口去重。"},
        {"page_num": 3, "text": "共同段落文本足够长，用于测试重叠窗口去重。第三页正文。"},
        {"page_num": 4, "text": "第四页正文。"},
    ]
    segment = {
        "type": "paragraph",
        "text": "共同段落文本足够长，用于测试重叠窗口去重。",
        "page_start": 2,
        "page_end": 2,
        "source_quote_start": "共同段落文本",
        "source_quote_end": "重叠窗口去重",
        "confidence": 0.88,
    }
    llm = FakeSegmentLLM([{"segments": [segment]}, {"segments": [segment]}])

    elements, quality, _ = LLMPageSegmenter(
        llm,
        config=LLMPageSegmenterConfig(window_size=3, overlap_pages=1),
    ).segment(pages)

    assert len(elements) == 1
    assert quality.duplicate_overlap_count == 1


def test_llm_failed_window_falls_back_to_rule_segmenter():
    pages = [{
        "page_num": 5,
        "text": (
            "这是一段超过最小长度的规则分段文本，用于验证 LLM 失败时不会阻塞导入。\n"
            "这里继续补充正文内容，确保规则分段器能够判断这是正文页面而不是目录或封面。\n"
            "最后再加入一行较长的教材正文，模拟真实 PDF 抽页之后的自然段文本。"
        ),
    }]
    llm = FakeSegmentLLM([RuntimeError("network down")])

    elements, quality, _ = LLMPageSegmenter(llm).segment(pages)

    assert len(elements) >= 1
    assert elements[0].type in (ElementType.PARAGRAPH, ElementType.HEADING)
    assert quality.llm_failed_windows == 1


def test_llm_segmenter_defaults_favor_more_context_and_budget():
    seg = LLMPageSegmenter(FakeSegmentLLM([{"segments": []}]))

    assert seg.config.window_size == 4
    assert seg.config.overlap_pages == 2
    assert seg.config.segment_max_tokens == 2000


def test_extract_text_does_not_truncate_long_segment_when_only_start_quote_matches():
    llm = FakeSegmentLLM([{"segments": []}])
    seg = LLMPageSegmenter(llm)
    long_tail = "甲" * 2600
    window_text = f"段落开头{long_tail}"

    text = seg._extract_text(
        {"start_quote": "段落开头", "end_quote": ""},
        window_text,
    )

    assert text == window_text
    assert len(text) > 2000
