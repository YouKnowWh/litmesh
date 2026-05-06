"""Traversal: Typed Pointer Mesh traversal engine (v0.4).

TraversalPresets -> TraversalPlan -> TraversalExecutor -> TraversalResult -> TraversalTrace.
"""
from .traversal_presets import build_preset_plan, get_preset, PRESETS
from .traversal_executor import TraversalExecutor
from .traversal_trace import TraceStore
