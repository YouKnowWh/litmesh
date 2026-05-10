"""
Thread-safe progress tracker for pipeline stages.

Used by synchronous pipeline threads to emit progress events,
and by async SSE endpoints to stream them to clients.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ProgressEvent:
    stage: str
    message: str
    percentage: float
    timestamp: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)
    index: int = 0


class ProgressTracker:
    """Thread-safe pub/sub for pipeline progress events.

    Usage:
        tracker = ProgressTracker()
        tracker.create_task("task_1", "Starting...")
        tracker.emit("task_1", "parsing", "Parsing PDF...", 5.0)
        ...
        tracker.emit("task_1", "done", "Complete", 100.0, paper_id="x")

        # SSE reader:
        events = tracker.get_events_since("task_1", last_index)
    """

    def __init__(self, cleanup_after: float = 600.0):
        self._events: dict[str, list[ProgressEvent]] = {}
        self._locks: dict[str, threading.Lock] = {}
        self._global_lock = threading.Lock()
        self._cleanup_after = cleanup_after

    def create_task(self, task_id: str, initial_message: str = "Starting") -> None:
        with self._global_lock:
            if task_id not in self._events:
                self._events[task_id] = []
                self._locks[task_id] = threading.Lock()
        self.emit(task_id, "created", initial_message, 0.0)

    def emit(self, task_id: str, stage: str, message: str,
             percentage: float, **metadata) -> None:
        lock = self._locks.get(task_id)
        if lock is None:
            with self._global_lock:
                if task_id not in self._events:
                    self._events[task_id] = []
                    self._locks[task_id] = threading.Lock()
                lock = self._locks[task_id]

        event = ProgressEvent(
            stage=stage,
            message=message,
            percentage=percentage,
            metadata=metadata or {},
        )

        with lock:
            if task_id in self._events:
                event.index = len(self._events[task_id])
                self._events[task_id].append(event)

    def get_events_since(self, task_id: str, since_index: int = 0) -> list[ProgressEvent]:
        lock = self._locks.get(task_id)
        if lock is None:
            return []
        with lock:
            events = self._events.get(task_id, [])
            return events[since_index:]

    def get_latest(self, task_id: str) -> Optional[ProgressEvent]:
        lock = self._locks.get(task_id)
        if lock is None:
            return None
        with lock:
            events = self._events.get(task_id, [])
            return events[-1] if events else None

    def is_done(self, task_id: str) -> bool:
        latest = self.get_latest(task_id)
        return latest is not None and latest.stage in ("done", "failed")

    def cleanup(self, task_id: str) -> None:
        """Remove a completed task from memory."""
        with self._global_lock:
            self._events.pop(task_id, None)
            self._locks.pop(task_id, None)

    def cleanup_stale(self) -> int:
        """Remove tasks that have been done for longer than _cleanup_after seconds."""
        now = time.time()
        stale = []
        with self._global_lock:
            for task_id in list(self._events.keys()):
                lock = self._locks.get(task_id)
                if lock is None:
                    continue
                with lock:
                    events = self._events.get(task_id, [])
                    if not events:
                        stale.append(task_id)
                        continue
                    latest = events[-1]
                    if latest.stage in ("done", "failed"):
                        if now - latest.timestamp > self._cleanup_after:
                            stale.append(task_id)
        for task_id in stale:
            self.cleanup(task_id)
        return len(stale)
