"""
LitMesh structural repair layer.

Repair pipeline:
  CandidateDetector (rules) -> RerankerClient (cross-encoder)
  -> RepairPolicy (thresholds) -> FallbackLLM (grey zone)
  -> RepairExecutor (orchestrator) -> RepairLog (JSONL audit)
"""

from .candidate_detector import CandidateDetector, RepairCandidate
from .reranker_client import RerankerClient, RerankerScore
from .repair_policy import RepairPolicy, RepairThresholds
from .repair_log import RepairLog, RepairLogEntry
from .fallback_llm import FallbackLLM, RepairLLMDecision
from .repair_executor import RepairExecutor
from .page_number_stripper import PageNumberStripper, PageNumberCandidate
from .heading_classifier import HeadingClassifier, HeadingRole, classify_heading
from .keyword_extractor import KeywordExtractor
from .title_augmenter import TitleAugmenter
