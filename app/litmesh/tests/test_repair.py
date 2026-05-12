"""
Tests for the repair module.

Covers: CandidateDetector, RepairPolicy, RepairLog, RepairExecutor (dry_run).
RerankerClient and FallbackLLM are tested with mocks.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest

from app.litmesh.models.section import SectionBlock, HeadingLevel, StructureStatus
from app.litmesh.repair.candidate_detector import CandidateDetector, RepairCandidate
from app.litmesh.repair.reranker_client import (
    RerankerClient, RerankerScore, _build_query, _build_document, _to_reranker_score,
)
from app.litmesh.repair.repair_policy import (
    RepairPolicy, RepairThresholds,
)
from app.litmesh.repair.repair_log import RepairLog, RepairLogEntry
from app.litmesh.repair.fallback_llm import FallbackLLM, RepairLLMDecision
from app.litmesh.repair.repair_executor import RepairExecutor
from app.litmesh.repair.page_number_stripper import (
    PageNumberStripper, PageNumberCandidate, _LEADING_NUM_RE, _NEVER_STRIP_AFTER,
)


# ---- Helpers ----

def _make_section(
    section_id: str,
    heading: str = "",
    raw_text: str = "",
    heading_level: HeadingLevel = HeadingLevel.PARAGRAPH_GROUP,
    heading_path: list | None = None,
    page_start: int | None = None,
    page_end: int | None = None,
    graph_id: str = "g1",
    paper_id: str = "p1",
) -> SectionBlock:
    return SectionBlock(
        section_id=section_id,
        graph_id=graph_id,
        paper_id=paper_id,
        heading=heading,
        raw_text=raw_text,
        heading_level=heading_level,
        heading_path=heading_path or [],
        page_start=page_start,
        page_end=page_end,
    )


# ---- CandidateDetector Tests ----

class TestCandidateDetector:
    def test_empty_sections(self):
        detector = CandidateDetector()
        assert detector.detect([]) == []

    def test_single_section_no_candidates(self):
        detector = CandidateDetector()
        sections = [_make_section("sec1", raw_text="完整的段落。")]
        candidates = detector.detect(sections)
        # Single section shouldn't produce adjacent_merge or chapter_boundary
        assert all(c.repair_type not in ("adjacent_merge", "chapter_boundary") for c in candidates)

    def test_adjacent_merge_no_sentence_end(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", raw_text="不完整的段落", heading_path=["第一章"]),
            _make_section("sec2", raw_text="继续的内容。", heading_path=["第一章"]),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "adjacent_merge"]
        assert len(candidates) >= 1
        c = candidates[0]
        assert c.section_ids == ["sec1", "sec2"]
        assert c.priority >= 0.4

    def test_adjacent_merge_short_block(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", raw_text="短文本", heading_path=["Ch1"]),
            _make_section("sec2", raw_text="正常段落内容在这里。", heading_path=["Ch1"]),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "adjacent_merge"]
        assert len(candidates) >= 1

    def test_adjacent_merge_complete_sentences_no_candidate(self):
        detector = CandidateDetector()
        long_text = "这是一段足够长的完整文本，包含完整的句子结构和句号结尾。" * 5
        sections = [
            _make_section("sec1", raw_text=long_text, heading="标题A",
                         heading_level=HeadingLevel.SECTION),
            _make_section("sec2", raw_text=long_text, heading="标题B",
                         heading_level=HeadingLevel.SECTION),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "adjacent_merge"]
        # Both have proper headings, long text, complete sentences → weak or no merge signal
        strong = [c for c in candidates if c.priority >= 0.5]
        assert len(strong) == 0

    def test_heading_too_long(self):
        detector = CandidateDetector()
        long_heading = "这是一个非常长的标题" * 10  # ~100 chars
        sections = [_make_section("sec1", heading=long_heading, raw_text="正文")]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "heading_role"]
        assert len(candidates) >= 1
        assert candidates[0].features["suspicion"] == "too_long"

    def test_missing_chapter_heading(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", raw_text="第二章 细胞结构\n这是正文内容。",
                         heading_level=HeadingLevel.PARAGRAPH_GROUP),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "heading_role"
                     and c.features.get("suspicion") == "missing_heading"]
        assert len(candidates) >= 1

    def test_toc_boundary_dots(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", raw_text="第一章 细胞 …… 1\n第二章 组织 …… 25\n第三章 器官 …… 50"),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "toc_boundary"]
        assert len(candidates) >= 1
        assert candidates[0].priority >= 0.5

    def test_chapter_boundary_level_jump(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", heading="第一章", heading_level=HeadingLevel.CHAPTER,
                         raw_text="正文"),
            _make_section("sec2", heading="1.1.1 子子节", heading_level=HeadingLevel.SUBSUBSECTION,
                         raw_text="正文"),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "chapter_boundary"]
        assert len(candidates) >= 1
        c = candidates[0]
        assert c.features["jump"] >= 2
        assert c.priority >= 0.7

    def test_structure_gap(self):
        detector = CandidateDetector()
        sections = [
            _make_section("sec1", raw_text="第5页内容", page_start=5, page_end=5),
            _make_section("sec2", raw_text="第10页内容", page_start=10, page_end=10),
        ]
        candidates = [c for c in detector.detect(sections) if c.repair_type == "structure_gap"]
        assert len(candidates) >= 1
        assert candidates[0].features["gap_pages"] == 5

    def test_min_priority_filter(self):
        detector = CandidateDetector(min_priority=0.9)
        sections = [
            _make_section("sec1", raw_text="完整的句子。"),
            _make_section("sec2", raw_text="另一个句子。"),
        ]
        candidates = detector.detect(sections)
        # Very strict filter → few or no candidates
        assert all(c.priority >= 0.9 for c in candidates)


# ---- RerankerClient Tests ----

class TestRerankerClient:
    def test_build_query_per_type(self):
        for rtype in ["adjacent_merge", "heading_role", "toc_boundary",
                       "front_matter_boundary", "chapter_boundary", "structure_gap"]:
            c = RepairCandidate(
                candidate_id=f"test:{rtype}",
                repair_type=rtype,
                section_ids=["s1", "s2"],
            )
            q = _build_query(c)
            assert isinstance(q, str) and len(q) > 0

    def test_build_document_adjacent_merge(self):
        sections = [
            _make_section("s1", heading="标题A", raw_text="文本A"),
            _make_section("s2", heading="标题B", raw_text="文本B"),
        ]
        c = RepairCandidate(
            candidate_id="test",
            repair_type="adjacent_merge",
            section_ids=["s1", "s2"],
        )
        doc = _build_document(c, sections)
        assert "标题A" in doc
        assert "标题B" in doc
        assert "---" in doc

    def test_to_reranker_score_merge_high(self):
        c = RepairCandidate(candidate_id="c1", repair_type="adjacent_merge",
                           section_ids=["s1", "s2"])
        score = _to_reranker_score(c, 0.9)
        assert score.label == "merge"
        assert score.confidence == 0.9

    def test_to_reranker_score_merge_low(self):
        c = RepairCandidate(candidate_id="c1", repair_type="adjacent_merge",
                           section_ids=["s1", "s2"])
        score = _to_reranker_score(c, 0.1)
        assert score.label == "keep"
        assert score.confidence == 0.9  # 1.0 - 0.1

    def test_to_reranker_score_toc(self):
        c = RepairCandidate(candidate_id="c1", repair_type="toc_boundary",
                           section_ids=["s1"])
        score = _to_reranker_score(c, 0.8)
        assert score.label == "toc"


# ---- RepairPolicy Tests ----

class TestRepairPolicy:
    def test_classify_auto_fix(self):
        policy = RepairPolicy()
        score = RerankerScore(candidate_id="c1", score=0.9, label="merge", confidence=0.9)
        assert policy.classify(score) == "auto_fix"

    def test_classify_grey_zone(self):
        policy = RepairPolicy()
        score = RerankerScore(candidate_id="c1", score=0.7, label="merge", confidence=0.7)
        assert policy.classify(score) == "grey_zone"

    def test_classify_skip(self):
        policy = RepairPolicy()
        score = RerankerScore(candidate_id="c1", score=0.3, label="keep", confidence=0.3)
        assert policy.classify(score) == "skip"

    def test_classify_boundary(self):
        policy = RepairPolicy(thresholds=RepairThresholds(auto_approve=0.85, grey_zone_low=0.5))
        score = RerankerScore(candidate_id="c1", score=0.85, label="merge", confidence=0.85)
        assert policy.classify(score) == "auto_fix"
        score = RerankerScore(candidate_id="c1", score=0.5, label="merge", confidence=0.5)
        assert policy.classify(score) == "grey_zone"
        score = RerankerScore(candidate_id="c1", score=0.49, label="keep", confidence=0.49)
        assert policy.classify(score) == "skip"

    def test_needs_llm_fallback(self):
        policy = RepairPolicy()
        # Grey zone + high impact
        score = RerankerScore(candidate_id="c1", score=0.7, label="toc", confidence=0.7)
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="toc_boundary", section_ids=["s1"],
        )
        assert policy.needs_llm_fallback(score, candidate) is True

        # Grey zone + low impact
        score2 = RerankerScore(candidate_id="c2", score=0.6, label="merge", confidence=0.6)
        candidate2 = RepairCandidate(
            candidate_id="c2", repair_type="adjacent_merge", section_ids=["s1", "s2"],
        )
        assert policy.needs_llm_fallback(score2, candidate2) is False

    def test_apply_merge(self):
        policy = RepairPolicy()
        sections = [
            _make_section("s1", heading="标题", raw_text="第一段",
                         heading_level=HeadingLevel.SECTION, page_start=1, page_end=1),
            _make_section("s2", raw_text="第二段", page_start=2, page_end=2),
        ]
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="adjacent_merge", section_ids=["s1", "s2"],
        )
        score = RerankerScore(candidate_id="c1", score=0.9, label="merge", confidence=0.9)
        result = policy.apply_repair(candidate, score, sections)
        assert len(result) == 1
        assert "第一段" in result[0].raw_text
        assert "第二段" in result[0].raw_text
        assert result[0].structure_status == StructureStatus.RECONSTRUCTED
        assert result[0].page_end == 2  # Extended

    def test_apply_heading_role_demote(self):
        policy = RepairPolicy()
        sections = [
            _make_section("s1", heading="这其实是一段正文不应该作为标题存在因为太长了",
                         raw_text="真正的正文在这里。",
                         heading_level=HeadingLevel.SECTION),
        ]
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="heading_role", section_ids=["s1"],
        )
        score = RerankerScore(candidate_id="c1", score=0.9, label="not_heading", confidence=0.9)
        result = policy.apply_repair(candidate, score, sections)
        assert result[0].heading == ""
        assert result[0].heading_level == HeadingLevel.PARAGRAPH_GROUP
        assert "这其实是一段" in result[0].raw_text

    def test_apply_toc_boundary(self):
        policy = RepairPolicy()
        sections = [_make_section("s1", raw_text="第一章 …… 1")]
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="toc_boundary", section_ids=["s1"],
        )
        score = RerankerScore(candidate_id="c1", score=0.9, label="toc", confidence=0.9)
        result = policy.apply_repair(candidate, score, sections)
        assert result[0].heading == "目录"
        assert result[0].structure_status == StructureStatus.RECONSTRUCTED

    def test_from_env(self, monkeypatch):
        monkeypatch.setenv("LITMESH_REPAIR_AUTO_THRESHOLD", "0.9")
        monkeypatch.setenv("LITMESH_REPAIR_GREY_THRESHOLD", "0.6")
        policy = RepairPolicy.from_env()
        assert policy.thresholds.auto_approve == 0.9
        assert policy.thresholds.grey_zone_low == 0.6


# ---- RepairLog Tests ----

class TestRepairLog:
    def test_log_and_read(self, tmp_path):
        log = RepairLog(log_dir=str(tmp_path))
        entry = RepairLogEntry(
            paper_id="p1", graph_id="g1", phase="phase2_observe",
            candidate_id="c1", repair_type="adjacent_merge",
            section_ids=["s1", "s2"], rule_priority=0.6,
            reranker_score=0.8, reranker_confidence=0.8,
            classification="auto_fix", action_taken="logged_only",
            before_state={"s1": {"raw_text_len": 100}},
        )
        log.log(entry)
        entries = log.get_entries(paper_id="p1")
        assert len(entries) == 1
        assert entries[0].candidate_id == "c1"

    def test_summary(self, tmp_path):
        log = RepairLog(log_dir=str(tmp_path))
        for i in range(3):
            log.log(RepairLogEntry(
                paper_id="p1", candidate_id=f"c{i}",
                repair_type="adjacent_merge",
                section_ids=[f"s{i}"], classification="auto_fix",
                action_taken="logged_only",
                before_state={},
            ))
        log.log(RepairLogEntry(
            paper_id="p1", candidate_id="c_llm",
            repair_type="toc_boundary", section_ids=["s_toc"],
            classification="grey_zone", llm_triggered=True,
            action_taken="applied", before_state={},
        ))
        s = log.summary(paper_id="p1")
        assert s["total"] == 4
        assert s["llm_triggered"] == 1
        assert s["by_classification"]["auto_fix"] == 3


# ---- FallbackLLM Tests ----

class TestFallbackLLM:
    def test_parse_merge_response(self):
        llm = FallbackLLM(None)  # type: ignore
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="adjacent_merge", section_ids=["s1", "s2"],
        )
        score = RerankerScore(candidate_id="c1", score=0.7, label="merge", confidence=0.7)
        result = {"should_merge": True, "confidence": 0.85, "reasoning": "话题连续"}
        decision = llm._parse_response(candidate, result, score)
        assert decision.decision == "merge"
        assert decision.confidence == 0.85
        assert decision.reasoning == "话题连续"

    def test_parse_toc_response(self):
        llm = FallbackLLM(None)  # type: ignore
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="toc_boundary", section_ids=["s1"],
        )
        score = RerankerScore(candidate_id="c1", score=0.6, label="toc", confidence=0.6)
        result = {"is_toc": True, "confidence": 0.9, "reasoning": "包含目录结构"}
        decision = llm._parse_response(candidate, result, score)
        assert decision.decision == "mark_toc"

    def test_build_prompt_adjacent_merge(self):
        llm = FallbackLLM(None)  # type: ignore
        sections = [
            _make_section("s1", heading="标题A", raw_text="文本A"),
            _make_section("s2", heading="标题B", raw_text="文本B"),
        ]
        candidate = RepairCandidate(
            candidate_id="c1", repair_type="adjacent_merge", section_ids=["s1", "s2"],
        )
        prompt = llm._build_prompt(candidate, {s.section_id: s for s in sections})
        assert "标题A" in prompt
        assert "文本A" in prompt
        assert "标题B" in prompt
        assert "should_merge" in prompt


# ---- RepairExecutor Integration Tests ----

class TestRepairExecutor:
    def test_dry_run_does_not_load_reranker(self, tmp_path):
        executor = RepairExecutor(
            detector=CandidateDetector(min_priority=0.3),
            log=RepairLog(log_dir=str(tmp_path)),
        )
        sections = [
            _make_section("s1", raw_text="不完整的段落", heading_path=["Ch1"], page_start=1, page_end=1),
            _make_section("s2", raw_text="继续的内容。", heading_path=["Ch1"], page_start=2, page_end=2),
        ]

        def _boom(*args, **kwargs):
            raise AssertionError("dry_run should not call reranker.score")

        executor.reranker.score = _boom
        result_sections, report = executor.repair(
            sections, paper_id="test_paper", mode="dry_run",
        )
        assert len(result_sections) == 2
        assert report["candidates_found"] >= 1
        assert report["auto_fixed"] == 0

    def test_dry_run_no_modify(self):
        executor = RepairExecutor(
            detector=CandidateDetector(min_priority=0.3),
        )
        sections = [
            _make_section("s1", raw_text="不完整的段落",
                         heading_path=["Ch1"], page_start=1, page_end=1),
            _make_section("s2", raw_text="继续的内容。",
                         heading_path=["Ch1"], page_start=2, page_end=2),
        ]
        result_sections, report = executor.repair(
            sections, paper_id="test_paper", mode="dry_run",
        )
        # Sections unchanged in dry_run
        assert len(result_sections) == 2
        assert report["candidates_found"] >= 0
        # No auto_fixed in dry_run
        assert report["auto_fixed"] == 0

    def test_auto_only_applies_high_confidence(self, tmp_path, monkeypatch):
        # We need to mock the reranker since we can't load the model in tests
        import app.litmesh.repair.repair_executor as exec_mod

        executor = RepairExecutor(
            detector=CandidateDetector(min_priority=0.3),
            policy=RepairPolicy(thresholds=RepairThresholds(auto_approve=0.5, grey_zone_low=0.3)),
            log=RepairLog(log_dir=str(tmp_path)),
        )

        sections = [
            _make_section("s1", raw_text="不完整段落，没有句号结尾",
                         heading_path=["Ch1"], page_start=1, page_end=1),
            _make_section("s2", raw_text="继续的内容。",
                         heading_path=["Ch1"], page_start=2, page_end=2),
        ]

        # Mock the reranker to return a high-confidence merge score
        original_score = executor.reranker.score

        def mock_score(candidates, secs):
            return [
                RerankerScore(
                    candidate_id=c.candidate_id,
                    score=0.9, label="merge", confidence=0.9,
                )
                for c in candidates
            ]

        executor.reranker.score = mock_score

        try:
            result, report = executor.repair(
                sections, paper_id="test", mode="auto_only",
            )
            # Should have merged
            assert report["auto_fixed"] >= 1
            # Check log was written
            entries = executor.log.get_entries(paper_id="test")
            assert len(entries) >= 1
        finally:
            executor.reranker.score = original_score

    def test_mode_off_skips_repair(self):
        """When mode is not provided as dry_run/auto_only/full, defaults to dry_run behavior."""
        executor = RepairExecutor(
            detector=CandidateDetector(min_priority=0.98),  # Very strict, no candidates
        )
        sections = [_make_section("s1", raw_text="完成。")]
        result, report = executor.repair(sections, paper_id="test")
        assert len(result) == 1
        assert report["candidates_found"] == 0


# ---- PageNumberStripper Tests ----

class TestPageNumberStripper:
    def test_never_strip_years(self):
        """Years like 2007, 1949 should never be collected as candidates."""
        assert _NEVER_STRIP_AFTER.match("年 1 月，印度...")
        assert _NEVER_STRIP_AFTER.match("年，中国...")
        assert _NEVER_STRIP_AFTER.match("月")
        assert _NEVER_STRIP_AFTER.match("世纪以来")
        assert _NEVER_STRIP_AFTER.match("年代后期")
        assert not _NEVER_STRIP_AFTER.match("印度共产党")

    def test_leading_num_re(self):
        m = _LEADING_NUM_RE.match("144 印度共产党")
        assert m and m.group(1) == "144"
        m = _LEADING_NUM_RE.match("2007 年")
        assert m and m.group(1) == "2007"
        assert _LEADING_NUM_RE.match("正常文本") is None

    def test_collect_filters_years(self):
        stripper = PageNumberStripper(llm_client=None)
        sections = [
            _make_section("s1", raw_text="144 正文内容在这里。"),
            _make_section("s2", raw_text="2007 年 1 月，发生了大事。"),
            _make_section("s3", raw_text="1949 年，新中国成立。"),
            _make_section("s4", raw_text="35 本章小结。"),
            _make_section("s5", raw_text="没有数字开头。"),
            _make_section("s6", raw_text="21 世纪以来。"),
        ]
        candidates = stripper.collect_candidates(sections)
        # Only "144" and "35" should be collected
        nums = {c.number_str for c in candidates}
        assert "144" in nums
        assert "35" in nums
        # Years and centuries should be filtered out at collection time
        assert "2007" not in nums
        assert "1949" not in nums
        assert "21" not in nums

    def test_regex_fallback_strips_page_numbers(self):
        stripper = PageNumberStripper(llm_client=None)
        sections = [
            _make_section("s1", raw_text="144 印度共产党在1957年。"),
            _make_section("s2", raw_text="2007 年 1 月，印度共产党。"),
            _make_section("s3", raw_text="35 本章讨论三个问题。"),
            _make_section("s4", raw_text="正常段落。"),
        ]
        result, report = stripper.process(sections, paper_id="test")

        # Page numbers stripped
        assert not result[0].raw_text.startswith("144"), f"Got: {result[0].raw_text[:40]}"
        assert not result[2].raw_text.startswith("35"), f"Got: {result[2].raw_text[:40]}"
        # Year preserved
        assert result[1].raw_text.startswith("2007 年"), f"Got: {result[1].raw_text[:40]}"
        # Normal text untouched
        assert result[3].raw_text.startswith("正常"), f"Got: {result[3].raw_text[:40]}"

        assert report["page_numbers_found"] >= 2
        assert report["stripped_sections"] >= 2

    def test_process_empty_sections(self):
        stripper = PageNumberStripper(llm_client=None)
        result, report = stripper.process([], paper_id="test")
        assert result == []
        assert report["candidates_collected"] == 0

    def test_deduplicate_by_number(self):
        from app.litmesh.repair.page_number_stripper import _deduplicate_by_number
        candidates = [
            PageNumberCandidate(candidate_id="c1", number_str="144",
                              context="144 第一段", section_id="s1"),
            PageNumberCandidate(candidate_id="c2", number_str="144",
                              context="144 第二段", section_id="s2"),
            PageNumberCandidate(candidate_id="c3", number_str="35",
                              context="35 第三段", section_id="s3"),
        ]
        deduped = _deduplicate_by_number(candidates)
        assert len(deduped) == 2
        assert deduped[0].count == 2  # "144" appears twice
        assert deduped[1].count == 1

    def test_multiple_page_numbers_same_section(self):
        """A section that starts with a page number should only have it stripped once."""
        stripper = PageNumberStripper(llm_client=None)
        sections = [
            _make_section("s1", raw_text="144 这个144不是页码，只是数字。"),
        ]
        candidates = stripper.collect_candidates(sections)
        # "144" at start should be a candidate
        assert len(candidates) == 1
        result, report = stripper.process(sections, paper_id="test")
        # Leading "144 " stripped, but "144" in the middle preserved
        assert result[0].raw_text.startswith("这个144")
        assert "144" in result[0].raw_text  # the mid-text 144 preserved
