"""Universal block roles — language-agnostic, pattern-driven."""

from enum import Enum


class BlockRole(str, Enum):
    FRONT = "front"             # Front/back matter: TOC, preface, copyright, appendix header
    STRUCTURAL = "structural"   # Chapter/section-level: organizes many subsequent blocks
    CONTEXT = "context"         # Local organizer: discussion, example, activity, note
    CONTENT = "content"         # Ordinary body text block
    NOISE = "noise"             # Header/footer, debris, meaningless fragments
