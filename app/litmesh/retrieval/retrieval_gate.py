"""
RetrievalGate: decides when to trigger memory retrieval.

Not every query needs vector retrieval. The gate checks:
- Is the query too short? (skip)
- Does it match trigger keywords? (force retrieve)
- Is it a follow-up within the context window? (skip)
- Are state/cognitive anchors sufficient? (skip)

Inspired by KokoroMemo's RetrievalGate design.
"""

from dataclasses import dataclass, field


_TRIGGER_KEYWORDS = [
    "记得", "还记得", "上次", "以前", "之前", "曾经",
    "约定", "说过", "提到过", "你提过", "你讲过的",
    "那个人", "那个地方", "那个概念", "叫什么",
    "什么是", "解释一下", "如何定义",
    "文献中", "论文中", "框架", "模型",
    "证据", "限制", "冲突", "矛盾",
    "对比", "区别", "比较", "异同",
]


@dataclass
class GateInput:
    query: str
    context_turn_count: int = 0
    existing_anchor_count: int = 0
    existing_confidence_sum: float = 0.0


@dataclass
class GateDecision:
    should_retrieve: bool
    mode: str  # "vector", "fts_only", "graph_only", "full", "skip"
    reason: str


def decide_retrieval(inp: GateInput,
                     trigger_keywords: list[str] | None = None,
                     skip_short_queries: int = 4,
                     skip_when_anchors_sufficient: bool = True) -> GateDecision:
    """Decide whether and how to retrieve.

    Returns GateDecision with mode:
    - "skip": don't retrieve, existing context is sufficient
    - "fts_only": use full-text search only (keyword match)
    - "vector": use vector search (semantic match)
    - "graph_only": use graph traversal only (structured navigation)
    - "full": all methods
    """
    if trigger_keywords is None:
        trigger_keywords = _TRIGGER_KEYWORDS

    query = inp.query.strip()

    # 1. Skip very short queries
    if len(query) < skip_short_queries:
        return GateDecision(False, "skip", f"Query too short ({len(query)} chars)")

    # 2. Check trigger keywords
    for kw in trigger_keywords:
        if kw in query:
            return GateDecision(True, "full", f"Trigger keyword matched: '{kw}'")

    # 3. If we have strong anchors, skip vector retrieval
    if skip_when_anchors_sufficient and inp.existing_anchor_count >= 3:
        avg_conf = inp.existing_confidence_sum / inp.existing_anchor_count
        if avg_conf >= 0.7:
            return GateDecision(False, "skip", f"Sufficient cognitive anchors ({inp.existing_anchor_count}, avg_conf={avg_conf:.1%})")

    # 4. For long, specific queries: use vector
    if len(query) > 20:
        return GateDecision(True, "vector", f"Long query ({len(query)} chars), semantic retrieval needed")

    # 5. Default: light retrieval
    return GateDecision(True, "fts_only", "Default: keyword search")
