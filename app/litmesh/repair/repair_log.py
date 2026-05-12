"""
JSONL audit log for repair decisions.

Records every repair candidate and its disposition for offline analysis.
"""

import json
import os
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any


@dataclass
class RepairLogEntry:
    timestamp: str = ""
    paper_id: str = ""
    graph_id: str = ""
    phase: str = ""             # "phase2_observe" | "phase3_auto" | "phase4_llm"
    candidate_id: str = ""
    repair_type: str = ""
    section_ids: List[str] = field(default_factory=list)
    rule_priority: float = 0.0
    reranker_score: float = 0.0
    reranker_confidence: float = 0.0
    classification: str = ""    # auto_fix | grey_zone | skip
    llm_triggered: bool = False
    llm_decision: Optional[str] = None
    llm_reasoning: Optional[str] = None
    action_taken: str = ""      # applied | logged_only | skipped
    before_state: Dict[str, Any] = field(default_factory=dict)
    after_state: Optional[Dict[str, Any]] = None


class RepairLog:
    """Appends one JSONL line per repair decision.

    Log file: {log_dir}/{paper_id}_{timestamp}.jsonl
    """

    def __init__(self, log_dir: str = "logs/repair_audit/"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._current_path: Optional[Path] = None

    def _ensure_file(self, paper_id: str) -> Path:
        if self._current_path is not None:
            return self._current_path
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        filename = f"{paper_id or 'unknown'}_{ts}.jsonl"
        self._current_path = self.log_dir / filename
        return self._current_path

    def log(self, entry: RepairLogEntry):
        """Append a single repair decision to the JSONL file."""
        if not entry.timestamp:
            entry.timestamp = datetime.now(timezone.utc).isoformat()
        path = self._ensure_file(entry.paper_id)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")

    def get_entries(self, paper_id: Optional[str] = None) -> List[RepairLogEntry]:
        """Read all entries, optionally filtered by paper_id."""
        entries = []
        pattern = f"{paper_id}_*.jsonl" if paper_id else "*.jsonl"
        for path in sorted(self.log_dir.glob(pattern)):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            entries.append(RepairLogEntry(**json.loads(line)))
                        except TypeError:
                            # Skip malformed lines
                            pass
        return entries

    def summary(self, paper_id: Optional[str] = None) -> dict:
        """Return aggregate statistics for all or filtered entries."""
        entries = self.get_entries(paper_id)
        if not entries:
            return {"total": 0}

        by_type: Dict[str, int] = {}
        by_classification: Dict[str, int] = {}
        by_action: Dict[str, int] = {}
        llm_count = 0

        for e in entries:
            by_type[e.repair_type] = by_type.get(e.repair_type, 0) + 1
            by_classification[e.classification] = by_classification.get(e.classification, 0) + 1
            by_action[e.action_taken] = by_action.get(e.action_taken, 0) + 1
            if e.llm_triggered:
                llm_count += 1

        return {
            "total": len(entries),
            "by_type": by_type,
            "by_classification": by_classification,
            "by_action": by_action,
            "llm_triggered": llm_count,
        }
