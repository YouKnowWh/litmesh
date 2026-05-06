"""
Traversal mode presets.

Each preset defines which pointer types to traverse, in what order,
with what constraints. These are the "travel policies" that the
TraversalExecutor enforces programmatically.

Principle: LLM picks the mode; program enforces the rules.
"""

from ..models.prompt_packet import TraversalMode, PointerType, TraversalPlan


# ============================================================
# Preset definitions
# ============================================================

from dataclasses import dataclass, field


@dataclass
class PresetConfig:
    """Configuration for a traversal mode."""
    pointer_types: list[PointerType]
    max_depth: int
    max_nodes: int
    max_edges_per_type: int
    require_source_span: bool
    must_include_limitations: bool
    allow_cross_graph: bool
    max_cross_graph_jumps: int
    min_confidence: float
    budget: int  # approximate token budget for final PromptPacket


PRESETS: dict[TraversalMode, PresetConfig] = {
    # ----
    # explain_mode: concept exploration
    # Walk: concept -> claim, concept -> child concept,
    #       concept -> related concept, concept -> limitation
    # ----
    TraversalMode.EXPLAIN: PresetConfig(
        pointer_types=[
            PointerType.BELONGS_TO,
            PointerType.DERIVED_FROM,
            PointerType.SUPPORTS,
            PointerType.CONSTRAINS,
        ],
        max_depth=3,
        max_nodes=40,
        max_edges_per_type=15,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.5,
        budget=4000,
    ),

    # ----
    # audit_mode: claim verification
    # Walk: claim -> evidence, claim -> limitation,
    #       claim -> claim (supports/contradicts), claim -> source_span
    # ----
    TraversalMode.AUDIT: PresetConfig(
        pointer_types=[
            PointerType.SUPPORTS,
            PointerType.CONSTRAINS,
            PointerType.CONTRADICTS,
            PointerType.DERIVED_FROM,
            PointerType.REFINES,
        ],
        max_depth=2,
        max_nodes=25,
        max_edges_per_type=10,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.6,
        budget=3000,
    ),

    # ----
    # compare_mode: framework/literature comparison
    # Walk: concept -> claim, claim -> evidence, claim -> limitation,
    #       claim -> claim (refines)
    # ----
    TraversalMode.COMPARE: PresetConfig(
        pointer_types=[
            PointerType.DERIVED_FROM,
            PointerType.SUPPORTS,
            PointerType.CONSTRAINS,
            PointerType.CONTRADICTS,
            PointerType.REFINES,
        ],
        max_depth=3,
        max_nodes=50,
        max_edges_per_type=20,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.5,
        budget=5000,
    ),

    # ----
    # trace_mode: source verification
    # Walk: claim -> source_span, claim -> section, section -> paper
    # ----
    TraversalMode.TRACE: PresetConfig(
        pointer_types=[
            PointerType.BELONGS_TO,
            PointerType.SECTION_PARENT,
            PointerType.SECTION_NEXT,
        ],
        max_depth=2,
        max_nodes=20,
        max_edges_per_type=10,
        require_source_span=True,  # Non-negotiable for trace mode
        must_include_limitations=False,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.3,  # Low bar for structural traversal
        budget=2000,
    ),

    # ----
    # conflict_mode: contradiction analysis
    # Walk: claim -> contradicts claim, claim -> limitation constraining claim,
    #       claim -> evidence (for both sides)
    # ----
    TraversalMode.CONFLICT: PresetConfig(
        pointer_types=[
            PointerType.CONTRADICTS,
            PointerType.CONSTRAINS,
            PointerType.SUPPORTS,
            PointerType.DERIVED_FROM,
        ],
        max_depth=3,
        max_nodes=30,
        max_edges_per_type=15,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.5,
        budget=4000,
    ),

    # ----
    # synthesis_mode: literature review
    # Walk: concept -> claim, claim -> evidence, claim -> limitation,
    #       claim -> supports claim, claim -> refines claim
    # ----
    TraversalMode.SYNTHESIS: PresetConfig(
        pointer_types=[
            PointerType.DERIVED_FROM,
            PointerType.SUPPORTS,
            PointerType.CONSTRAINS,
            PointerType.CONTRADICTS,
            PointerType.REFINES,
            PointerType.EXTENDS,
        ],
        max_depth=4,
        max_nodes=60,
        max_edges_per_type=25,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=False,
        max_cross_graph_jumps=0,
        min_confidence=0.4,
        budget=6000,
    ),

    # ----
    # transfer_mode: cross-graph knowledge transfer
    # Walk: source graph concept -> claim, target graph concept -> claim,
    #       analogous_to bridge, transfers_to bridge,
    #       conflicts_with bridge, claim -> limitation
    # ----
    TraversalMode.TRANSFER: PresetConfig(
        pointer_types=[
            PointerType.DERIVED_FROM,
            PointerType.SUPPORTS,
            PointerType.CONSTRAINS,
            PointerType.ANALOGOUS_TO_BRIDGE,
            PointerType.TRANSFERS_TO_BRIDGE,
            PointerType.CONFLICTS_WITH_BRIDGE,
        ],
        max_depth=4,
        max_nodes=40,
        max_edges_per_type=15,
        require_source_span=True,
        must_include_limitations=True,
        allow_cross_graph=True,
        max_cross_graph_jumps=3,
        min_confidence=0.4,
        budget=5000,
    ),
}


def build_preset_plan(
    mode: TraversalMode,
    start_nodes: list[str],
    graph_scope: list[str],
    task_type: str = "",
) -> TraversalPlan:
    """Build a TraversalPlan from a preset mode.

    Args:
        mode: Which traversal mode to use.
        start_nodes: Starting concept_keys or claim_ids.
        graph_scope: Which graph_ids to include.
        task_type: Short description of the user's task.

    Returns:
        TraversalPlan ready for TraversalExecutor.
    """
    preset = PRESETS[mode]
    return TraversalPlan(
        task_type=task_type or mode.value,
        start_nodes=start_nodes,
        graph_scope=graph_scope,
        pointer_types=preset.pointer_types,
        traversal_mode=mode,
        max_depth=preset.max_depth,
        max_nodes=preset.max_nodes,
        max_edges_per_pointer_type=preset.max_edges_per_type,
        allow_cross_graph=preset.allow_cross_graph,
        max_cross_graph_jumps=preset.max_cross_graph_jumps,
        require_source_span=preset.require_source_span,
        must_include_limitations=preset.must_include_limitations,
        budget=preset.budget,
    )


def get_preset(mode: TraversalMode) -> PresetConfig:
    """Get the preset config for a mode."""
    return PRESETS[mode]
