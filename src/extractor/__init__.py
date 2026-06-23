"""
src/indexing/extractor
----------------------
Public surface of the extractor sub-package.

Re-exports every class and utility function that downstream modules
(builder, search_index, tests) need, so callers can import from
`src.indexing.extractor` rather than from individual sub-modules.
"""

from .models import (
    ASN1TypeDef,
    EFFile,
    ErrorCode,
    Field,
    Relationship,
    ValidationRule,
)
from .section_extractor import ASN1Block, Section, SectionExtractor
from .table_extractor import SpecTable, TableExtractor, TableRecord
from .requirement_extractor import Requirement, RequirementExtractor, split_sentences
from .entity_extractor import (
    EntityExtractor,
    normalize_asn1_content,
    print_entity_stats,
    print_entity_stats_obj,
)
from .relationship_extractor import RelationshipExtractor, print_relationship_stats

__all__ = [
    # Data models
    "ASN1Block",
    "ASN1TypeDef",
    "EFFile",
    "ErrorCode",
    "Field",
    "Relationship",
    "Requirement",
    "Section",
    "SpecTable",
    "TableRecord",
    "ValidationRule",
    # Extractors
    "EntityExtractor",
    "RelationshipExtractor",
    "RequirementExtractor",
    "SectionExtractor",
    "TableExtractor",
    # Utilities
    "normalize_asn1_content",
    "split_sentences",
    # Stats
    "print_entity_stats",
    "print_entity_stats_obj",
    "print_relationship_stats",
]
