"""
src/indexing/search_index/models.py
------------------------------------
Shared data models for the search index layer.

SearchResult is used by both HybridIndex and VectorIndex so it lives
here rather than inside either index module.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """
    A single ranked result returned by any search method.

    Attributes:
        ko_id:          Knowledge Object identifier.
        ko_type:        Entity type (PE, Field, ValidationRule, …).
        primary_label:  Human-readable name of the entity.
        score:          Combined/final relevance score.
        bm25_score:     BM25 component score (0 when not applicable).
        vector_score:   Vector similarity component score (0 when not applicable).
        metadata:       Structured attributes for filtering and display.
        text_content:   Full text used for embedding / BM25 matching.
        relationships:  Adjacent graph edges (type, source, target).
    """

    ko_id: str
    ko_type: str
    primary_label: str
    score: float
    bm25_score: float
    vector_score: float
    metadata: dict
    text_content: str
    relationships: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "ko_id":         self.ko_id,
            "ko_type":       self.ko_type,
            "primary_label": self.primary_label,
            "score":         round(self.score, 4),
            "bm25_score":    round(self.bm25_score, 4),
            "vector_score":  round(self.vector_score, 4),
            "metadata":      self.metadata,
            "text_preview":  self.text_content[:300],
        }
