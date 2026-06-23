"""
src/indexing
-------------
Public surface of the indexing package.

All imports below maintain the exact same names and signatures as the
original flat-module layout, so no call sites need to change.
"""

from .parser.pdf_parser import PDFParser

from .extractor.section_extractor import SectionExtractor
from .extractor.table_extractor import TableExtractor
from .extractor.requirement_extractor import RequirementExtractor
from .extractor.entity_extractor import EntityExtractor, print_entity_stats_obj
from .extractor.relationship_extractor import RelationshipExtractor, print_relationship_stats

from .builder.knowledge_object_builder import KnowledgeObjectBuilder, print_ko_stats

from .search_index import HybridIndex, VectorIndex

__all__ = [
    "PDFParser",
    "SectionExtractor",
    "TableExtractor",
    "RequirementExtractor",
    "EntityExtractor",
    "RelationshipExtractor",
    "KnowledgeObjectBuilder",
    "HybridIndex",
    "VectorIndex",
    "print_entity_stats_obj",
    "print_relationship_stats",
    "print_ko_stats",
]
