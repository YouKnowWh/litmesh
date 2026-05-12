"""
StructureGroup builder — organizes classified SectionBlocks into a hierarchy.

structural blocks → structural groups
context blocks    → context groups (attached to nearest structural)
content blocks    → attached to nearest context or structural
front blocks      → merged into a single front group
noise blocks      → skipped
"""

from __future__ import annotations

import logging
from dataclasses import asdict, is_dataclass
from dataclasses import dataclass, field
from uuid import uuid4

from ..models.section import SectionBlock
from .block_role import BlockRole
from .role_classifier import RoleClassifier
from .title_augmenter import TitleAugmenter

logger = logging.getLogger("litmesh.structure")


@dataclass
class StructureGroup:
    """A structural or contextual grouping of blocks."""
    group_id: str
    paper_id: str
    graph_id: str
    role: str                     # BlockRole value
    structure_title: str = ""
    display_title: str = ""
    keyword_summary: str = ""
    heading_path: list[str] = field(default_factory=list)
    parent_group_id: str = ""
    child_section_ids: list[str] = field(default_factory=list)
    order_index: int = 0
    confidence: float = 0.8


class GroupBuilder:
    """Build StructureGroup hierarchy from classified SectionBlocks."""

    def __init__(
        self,
        classifier: RoleClassifier | None = None,
        augmenter: TitleAugmenter | None = None,
    ):
        self.classifier = classifier or RoleClassifier()
        self.augmenter = augmenter or TitleAugmenter()

    def build(
        self,
        sections: list[SectionBlock],
        paper_id: str = "",
        graph_id: str = "",
        outline_nodes: list | None = None,
    ) -> list[StructureGroup]:
        """Build structure groups from sections.

        Args:
            sections: SectionBlocks with heading and raw_text populated.
            paper_id: Paper identifier.
            graph_id: Graph identifier.
            outline_nodes: Optional TOC outline nodes for structure_title.

        Returns:
            List of StructureGroup in document order.
        """
        n = len(sections)
        if n == 0:
            return []

        # 1. Classify each section
        roles: list[BlockRole] = []
        for i, s in enumerate(sections):
            heading_level = 1 if s.heading_path else (3 if s.heading else 0)
            # Classify using heading text if available, else body text
            classify_text = s.heading if s.heading else s.raw_text[:200]
            role = self.classifier.classify(
                text=classify_text,
                order_index=i,
                total_blocks=n,
                heading_level=heading_level,
            )
            roles.append(role)
            # Write role back to section
            s.block_role = role.value

        # 2. Build hierarchy
        groups: list[StructureGroup] = []
        current_structural: StructureGroup | None = None
        current_context: StructureGroup | None = None
        front_sections: list[str] = []

        for i, (s, role) in enumerate(zip(sections, roles)):
            if role == BlockRole.NOISE:
                continue

            toc_title = self._find_toc_title(s, outline_nodes) if outline_nodes else ""

            if role == BlockRole.FRONT:
                front_sections.append(s.section_id)
                continue

            if role == BlockRole.STRUCTURAL:
                s_title, d_title = self.augmenter.generate(
                    s.heading, s.raw_text, role.value, toc_title,
                )
                sg = StructureGroup(
                    group_id=f"sg_{uuid4().hex[:12]}",
                    paper_id=paper_id,
                    graph_id=graph_id,
                    role=role.value,
                    structure_title=s_title,
                    display_title=d_title,
                    heading_path=list(s.heading_path),
                    order_index=len(groups),
                )
                groups.append(sg)
                current_structural = sg
                current_context = None
                sg.child_section_ids.append(s.section_id)
                s.group_id = sg.group_id
                s.structure_title = s_title
                continue

            if role == BlockRole.CONTEXT:
                s_title, d_title = self.augmenter.generate(
                    s.heading, s.raw_text, role.value, toc_title,
                )
                sg = StructureGroup(
                    group_id=f"sg_{uuid4().hex[:12]}",
                    paper_id=paper_id,
                    graph_id=graph_id,
                    role=role.value,
                    structure_title=s_title,
                    display_title=d_title,
                    heading_path=list(s.heading_path),
                    parent_group_id=current_structural.group_id if current_structural else "",
                    order_index=len(groups),
                )
                groups.append(sg)
                current_context = sg
                sg.child_section_ids.append(s.section_id)
                s.group_id = sg.group_id
                s.structure_title = s_title
                continue

            if role == BlockRole.CONTENT:
                # Attach to nearest context or structural group
                target = current_context or current_structural
                if target:
                    target.child_section_ids.append(s.section_id)
                    s.group_id = target.group_id
                    s.structure_title = target.structure_title
                else:
                    # Orphan content — create a minimal structural group
                    s_title, d_title = self.augmenter.generate(
                        "", s.raw_text, BlockRole.STRUCTURAL.value, "",
                    )
                    sg = StructureGroup(
                        group_id=f"sg_{uuid4().hex[:12]}",
                        paper_id=paper_id,
                        graph_id=graph_id,
                        role=BlockRole.STRUCTURAL.value,
                        structure_title=s_title,
                        display_title=d_title,
                        order_index=len(groups),
                    )
                    groups.append(sg)
                    current_structural = sg
                    sg.child_section_ids.append(s.section_id)
                    s.group_id = sg.group_id
                    s.structure_title = s_title

        # 3. Front matter group
        if front_sections:
            fg = StructureGroup(
                group_id=f"sg_{uuid4().hex[:12]}",
                paper_id=paper_id,
                graph_id=graph_id,
                role=BlockRole.FRONT.value,
                structure_title="",
                display_title="前置材料",
                order_index=0,
            )
            fg.child_section_ids = front_sections
            for sid in front_sections:
                for s in sections:
                    if s.section_id == sid:
                        s.group_id = fg.group_id
                        break
            groups.insert(0, fg)

        logger.info(
            "groups_built sections=%d groups=%d front=%d",
            n, len(groups), len(front_sections),
        )
        return groups

    @staticmethod
    def _find_toc_title(section, outline_nodes: list) -> str:
        """Match a section to its TOC title, if any."""
        anchor = getattr(section, "toc_anchor_title", None) or getattr(section, "toc_anchor_id", None)
        if anchor and outline_nodes:
            for node in outline_nodes:
                if isinstance(node, dict):
                    data = node
                elif is_dataclass(node):
                    data = asdict(node)
                else:
                    data = {
                        "outline_id": getattr(node, "outline_id", ""),
                        "title": getattr(node, "title", ""),
                    }
                nid = data.get("outline_id", "")
                ntitle = data.get("title", "")
                if anchor == nid or anchor == ntitle:
                    return ntitle
        return ""
