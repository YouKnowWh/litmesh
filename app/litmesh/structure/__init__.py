"""
LitMesh structure layer — language-agnostic document block classification and grouping.

Replaces the Chinese-textbook-specific repair/heading_classifier with a universal
5-role system: front / structural / context / content / noise.

Pipeline: DocumentBlock → RoleClassifier → TitleAugmenter → GroupBuilder → graph.
"""

from .block_role import BlockRole
from .role_classifier import RoleClassifier
from .keyword_summary import KeywordExtractor
from .title_augmenter import TitleAugmenter
from .group_builder import GroupBuilder, StructureGroup
