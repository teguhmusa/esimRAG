"""
src/indexing/search_index
--------------------------
Public surface of the search_index sub-package.

Exposes HybridIndex, VectorIndex, and SearchResult so callers can
import from `src.indexing` without knowing the internal layout.
"""

from .models import SearchResult
from .hybrid_index import HybridIndex
from .vector_index import VectorIndex

__all__ = [
    "HybridIndex",
    "VectorIndex",
    "SearchResult",
]
