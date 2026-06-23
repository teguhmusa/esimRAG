"""
src/indexing/search_index/_metadata.py
----------------------------------------
ChromaDB metadata serialisation helper.

ChromaDB only accepts metadata values of type str, int, float, or bool.
This private module contains the single utility function that flattens
arbitrary Python values to those supported types before upsert.
"""

import json


def flatten_metadata(metadata: dict) -> dict:
    """
    Flatten a metadata dict so all values are ChromaDB-compatible.

    - None  → ""
    - list  → JSON string
    - other scalars (str, int, float, bool) → unchanged
    - anything else → str()
    """
    flat: dict = {}
    for k, v in metadata.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, list):
            flat[k] = json.dumps(v)
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat
