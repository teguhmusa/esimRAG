"""
src/error_transformer/retriever.py
--------------------------------------------
Context Retriever for the error transformation pipeline.

Assembles all relevant spec context from the Hybrid/Vector Index for a
single ValidatorError using path lookup, graph traversal, and fallback
semantic search.
"""

from typing import Optional

from .models import ValidatorError, RetrievedContext


class ContextRetriever:
    """
    Retrieves and assembles spec context from the index for a given error.

    Strategy:
      1. Primary: direct path lookup → ValidationRule KO
      2. Fallback: hybrid semantic search on description + element_path tokens
      3. Graph traverse: BELONGS_TO → PE, HAS_TYPE → type,
                         VALIDATED_BY → requirements
      4. Section body text from primary KO's section_id
    """

    def __init__(
        self,
        index,
        sections_data: list[dict],
        requirements_data: list[dict],
    ):
        self._index     = index
        self._sec_index = {s["section_id"]: s for s in sections_data}
        self._req_index = {r["req_id"]: r for r in requirements_data}
        self._ko_index  = {ko["ko_id"]: ko for ko in index._kos}

    def retrieve(self, error: ValidatorError) -> RetrievedContext:
        """Retrieve all relevant context for one validator error."""
        primary_ko = self._index.lookup_path(error.element_path)

        if not primary_ko:
            query   = f"{error.element_path} {error.description} {error.expected_value}"
            results = self._index.search(query, top_k=3, ko_type="ValidationRule")
            if results:
                r          = results[0]
                primary_ko = {
                    "ko_id":         r.ko_id,
                    "ko_type":       r.ko_type,
                    "primary_label": r.primary_label,
                    "text_content":  r.text_content,
                    "metadata":      r.metadata,
                }

        rule_id   = primary_ko["metadata"].get("rule_id",   "") if primary_ko else ""
        section_id= primary_ko["metadata"].get("section_id","") if primary_ko else ""

        pe_ko   = self._resolve_neighbor(rule_id, "BELONGS_TO")
        type_ko = self._resolve_neighbor(rule_id, "HAS_TYPE")
        field_ko= self._find_field_ko(rule_id)

        req_ids = [
            e["target_id"]
            for e in self._index.graph_neighbors(rule_id, rel_type="VALIDATED_BY")
        ]
        requirement_kos = [
            self._ko_index.get(
                f"KO-REQ-{rid}",
                self._req_to_ko(self._req_index.get(rid)),
            )
            for rid in req_ids[:6]
            if self._req_index.get(rid)
        ]
        requirement_kos = [r for r in requirement_kos if r]

        section       = self._sec_index.get(section_id, {})
        section_body  = section.get("body_text", "")
        section_title = section.get("title", "")

        spec_expected = primary_ko["metadata"].get("expected_value") if primary_ko else None

        return RetrievedContext(
            primary_ko=primary_ko,
            field_ko=field_ko,
            pe_ko=pe_ko,
            type_ko=type_ko,
            requirements=requirement_kos,
            section_body=section_body,
            section_id=section_id,
            section_title=section_title,
            found_expected_value=spec_expected or error.expected_value,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Private helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _resolve_neighbor(self, entity_id: str, rel_type: str) -> Optional[dict]:
        """Get the first neighbor of the given rel_type and return its KO."""
        edges = self._index.graph_neighbors(entity_id, rel_type=rel_type)
        if not edges:
            return None
        target_id = edges[0]["target_id"]
        return (
            self._ko_index.get(f"KO-PE-{target_id}")
            or self._ko_index.get(f"KO-FIELD-{target_id}")
        )

    def _find_field_ko(self, rule_id: str) -> Optional[dict]:
        """Find the Field KO corresponding to this ValidationRule."""
        return self._ko_index.get(f"KO-FIELD-{rule_id}")

    def _req_to_ko(self, req: Optional[dict]) -> Optional[dict]:
        """Convert a requirement dict to a minimal KO-compatible dict."""
        if not req:
            return None
        return {
            "ko_id":         req["req_id"],
            "ko_type":       "Requirement",
            "primary_label": req["req_id"],
            "text_content":  req["statement"],
            "metadata": {
                "strength":   req["strength"],
                "keyword":    req["keyword"],
                "section_id": req["section_id"],
            },
        }
