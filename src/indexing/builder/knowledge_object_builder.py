# src/knowledge_object_builder.py
"""
Knowledge Object Builder for eUICC spec.

Aggregates entities + relationships into self-contained KnowledgeObjects
that are ready for hybrid indexing (Vector DB, BM25, Graph DB).

A KnowledgeObject represents one searchable unit:
  - For a Field:          full context (parent PE, type, constraints, requirements)
  - For a PE:             all fields, section body, requirements
  - For an ErrorCode:     description, numeric value, related sections
  - For a ValidationRule: path, expected value, section reference, requirements
  - For an EFFile:        FID, default value, parent PE, section

Each object has:
  text_content   → for BM25 + embedding
  metadata       → for filtering
  relationships  → for graph traversal
"""

import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class KnowledgeObject:
    ko_id: str
    ko_type: str            # Field | PE | ErrorCode | ValidationRule | EFFile | Section
    primary_label: str      # human-readable name
    text_content: str       # full text for embedding/BM25
    metadata: dict          # structured attributes for filtering
    relationships: list[dict] = field(default_factory=list)

    def to_dict(self):
        return {
            "ko_id":        self.ko_id,
            "ko_type":      self.ko_type,
            "primary_label": self.primary_label,
            "text_content": self.text_content,
            "metadata":     self.metadata,
            "relationships": self.relationships,
        }


class KnowledgeObjectBuilder:
    """
    Builds KnowledgeObjects from entities, relationships, sections, requirements.
    """

    def __init__(self):
        # Built indexes for fast lookup
        self._type_index   = {}   # type_id → type_def dict
        self._field_index  = {}   # field_id → field dict
        self._req_index    = {}   # req_id → requirement dict
        self._sec_index    = {}   # section_id → section dict
        self._rel_index    = {}   # source_id → [relationships]
        self._rev_index    = {}   # target_id → [relationships] (reverse)

    def build(
        self,
        entities: dict,
        relationships: list[dict],
        sections: list[dict],
        requirements: list[dict],
    ) -> list[KnowledgeObject]:
        """Main entry point. Returns list of KnowledgeObjects."""

        self._build_indexes(entities, relationships, sections, requirements)
        objects = []

        print("  → Building PE KnowledgeObjects...")
        for tdef in entities["type_defs"]:
            if tdef["is_pe"]:
                objects.append(self._build_pe_ko(tdef))

        print("  → Building Field KnowledgeObjects...")
        for tdef in entities["type_defs"]:
            for f in tdef["fields"]:
                objects.append(self._build_field_ko(f, tdef))

        print("  → Building ValidationRule KnowledgeObjects...")
        seen_paths = set()
        for vr in entities["validation_rules"]:
            path = vr["normalized_path"]
            if path not in seen_paths:
                seen_paths.add(path)
                objects.append(self._build_validation_rule_ko(vr))

        print("  → Building ErrorCode KnowledgeObjects...")
        for ec in entities["error_codes"]:
            objects.append(self._build_error_code_ko(ec))

        print("  → Building EFFile KnowledgeObjects...")
        for ef in entities["ef_files"]:
            objects.append(self._build_ef_ko(ef))

        return objects

    # ─────────────────────────────────────────────────────────────────────────
    # Index builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_indexes(self, entities, relationships, sections, requirements):
        for t in entities["type_defs"]:
            self._type_index[t["type_id"]] = t
            for f in t["fields"]:
                self._field_index[f["field_id"]] = f

        for r in requirements:
            self._req_index[r["req_id"]] = r

        for s in sections:
            self._sec_index[s["section_id"]] = s

        for rel in relationships:
            src = rel["source_id"]
            tgt = rel["target_id"]
            if src not in self._rel_index:
                self._rel_index[src] = []
            self._rel_index[src].append(rel)
            if tgt not in self._rev_index:
                self._rev_index[tgt] = []
            self._rev_index[tgt].append(rel)

    # ─────────────────────────────────────────────────────────────────────────
    # KO builders
    # ─────────────────────────────────────────────────────────────────────────

    def _build_pe_ko(self, tdef: dict) -> KnowledgeObject:
        pe_id = tdef["type_id"]
        section_id = tdef.get("section_id", "")
        section = self._sec_index.get(section_id, {})

        # Gather fields text
        fields_text = "\n".join(
            f"  - {f['name']} ({f['asn1_type']})"
            + (" [OPTIONAL]" if f["is_optional"] else "")
            + (f" DEFAULT {f['default_value']}" if f["default_value"] else "")
            + (f" — {f['comment']}" if f["comment"] else "")
            for f in tdef["fields"]
        )

        # Gather requirements for this section
        req_ids = [
            rel["target_id"] for rel in self._rel_index.get(pe_id, [])
            if rel["rel_type"] == "VALIDATED_BY"
        ]
        req_stmts = [
            self._req_index[rid]["statement"]
            for rid in req_ids if rid in self._req_index
        ]

        section_body = section.get("body_text", "")[:600]

        text = f"""Profile Element: {pe_id}
Section: {section_id} — {section.get('title', '')}

Fields:
{fields_text}

Section description:
{section_body}

Requirements:
{chr(10).join(f'- {s}' for s in req_stmts[:10])}
""".strip()

        rels = self._rel_index.get(pe_id, []) + self._rev_index.get(pe_id, [])

        return KnowledgeObject(
            ko_id=f"KO-PE-{pe_id}",
            ko_type="PE",
            primary_label=pe_id,
            text_content=text,
            metadata={
                "type_id":    pe_id,
                "section_id": section_id,
                "field_count": len(tdef["fields"]),
                "asn1_kind":  tdef["asn1_kind"],
            },
            relationships=[_rel_summary(r) for r in rels],
        )

    def _build_field_ko(self, f: dict, parent: dict) -> KnowledgeObject:
        field_id = f["field_id"]
        parent_id = f["parent_type"]
        section_id = f.get("section_id", "")
        section = self._sec_index.get(section_id, {})

        # Resolve type reference
        type_ref_text = ""
        ref_type = self._type_index.get(f["asn1_type"].split("(")[0].strip())
        if ref_type:
            type_ref_text = f"\nType '{f['asn1_type']}' is defined in section {ref_type.get('section_id', '')}."

        # Requirements that reference this field
        req_rels = [
            r for r in self._rev_index.get(field_id, [])
            if r["rel_type"] == "REFERENCES"
        ]
        req_stmts = [
            self._req_index[r["source_id"]]["statement"]
            for r in req_rels if r["source_id"] in self._req_index
        ]

        text = f"""Field: {f['name']}
Parent type: {parent_id}
Full path: {field_id}
ASN.1 type: {f['asn1_type']}
Optional: {f['is_optional']}
Default value: {f['default_value'] or 'none'}
Size constraint: {f['size_constraint'] or 'none'}
Description: {f['comment'] or 'No comment in spec'}
{type_ref_text}
Section: {section_id} — {section.get('title', '')}

Related requirements:
{chr(10).join(f'- {s}' for s in req_stmts[:5])}
""".strip()

        rels = self._rel_index.get(field_id, []) + self._rev_index.get(field_id, [])

        return KnowledgeObject(
            ko_id=f"KO-FIELD-{field_id}",
            ko_type="Field",
            primary_label=f"{parent_id}.{f['name']}",
            text_content=text,
            metadata={
                "field_id":       field_id,
                "name":           f["name"],
                "parent_type":    parent_id,
                "asn1_type":      f["asn1_type"],
                "is_optional":    f["is_optional"],
                "default_value":  f["default_value"],
                "section_id":     section_id,
            },
            relationships=[_rel_summary(r) for r in rels],
        )

    def _build_validation_rule_ko(self, vr: dict) -> KnowledgeObject:
        rule_id = vr["rule_id"]
        section_id = vr.get("section_id", "")
        section = self._sec_index.get(section_id, {})

        # Requirements linked via VALIDATED_BY
        val_rels = [
            r for r in self._rel_index.get(rule_id, [])
            if r["rel_type"] == "VALIDATED_BY"
        ]
        req_stmts = [
            self._req_index[r["target_id"]]["statement"]
            for r in val_rels if r["target_id"] in self._req_index
        ]

        text = f"""Validation Rule: {rule_id}
Normalized path: {vr['normalized_path']}
Field: {vr.get('field_ref', '')}
ASN.1 type: {vr.get('asn1_type_ref', '')}
Expected/default value: {vr.get('expected_value') or 'not specified'}
Section: {section_id} — {section.get('title', '')}
Description: {vr.get('description', '')}

Requirements governing this rule:
{chr(10).join(f'- {s}' for s in req_stmts[:5])}
""".strip()

        rels = self._rel_index.get(rule_id, [])

        return KnowledgeObject(
            ko_id=f"KO-VR-{vr['normalized_path']}",
            ko_type="ValidationRule",
            primary_label=vr["normalized_path"],
            text_content=text,
            metadata={
                "rule_id":        rule_id,
                "normalized_path": vr["normalized_path"],
                "field_ref":      vr.get("field_ref"),
                "asn1_type_ref":  vr.get("asn1_type_ref"),
                "expected_value": vr.get("expected_value"),
                "section_id":     section_id,
            },
            relationships=[_rel_summary(r) for r in rels],
        )

    def _build_error_code_ko(self, ec: dict) -> KnowledgeObject:
        code_id = ec["code_id"]
        section_id = ec.get("section_id", "")
        section = self._sec_index.get(section_id, {})

        # Sections where this error is raised
        raised_rels = [
            r for r in self._rel_index.get(code_id, [])
            if r["rel_type"] == "RAISED_IN"
        ]
        raised_in = [r["target_id"] for r in raised_rels]

        # Requirements that reference this error
        ref_rels = [
            r for r in self._rev_index.get(code_id, [])
            if r["rel_type"] == "REFERENCES"
        ]
        ref_stmts = [
            self._req_index[r["source_id"]]["statement"]
            for r in ref_rels if r["source_id"] in self._req_index
        ]

        numeric = ec.get("numeric_value")
        text = f"""Error Code: {code_id}
Numeric value: {numeric if numeric is not None else 'N/A'}
Description: {ec.get('description', '')}
Defined in section: {section_id} — {section.get('title', '')}
Also raised in sections: {', '.join(raised_in) or 'none'}

Requirements referencing this error:
{chr(10).join(f'- {s}' for s in ref_stmts[:5])}
""".strip()

        rels = self._rel_index.get(code_id, []) + self._rev_index.get(code_id, [])

        return KnowledgeObject(
            ko_id=f"KO-ERR-{code_id}",
            ko_type="ErrorCode",
            primary_label=code_id,
            text_content=text,
            metadata={
                "code_id":       code_id,
                "numeric_value": numeric,
                "section_id":    section_id,
                "raised_in":     raised_in,
            },
            relationships=[_rel_summary(r) for r in rels],
        )

    def _build_ef_ko(self, ef: dict) -> KnowledgeObject:
        ef_id = ef["ef_id"]
        section_id = ef.get("section_id", "")
        section = self._sec_index.get(section_id, {})

        text = f"""EF File: {ef['name']}
File ID (FID): {ef['fid']}
File type: {ef['file_type']}
Number of records: {ef['nb_records'] or 'N/A'}
Record size: {ef['rec_size'] or 'N/A'}
SFI: {ef['sfi'] or 'N/A'}
Access rules: {ef['access_rules'] or 'N/A'}
Default value: {ef['default_value'] or 'empty'}
Content required: {ef['content_required'] or 'N/A'}
Part of: {ef['parent_template'] or 'unknown PE'}
Section: {section_id} — {section.get('title', '')}
""".strip()

        rels = self._rel_index.get(ef_id, []) + self._rev_index.get(ef_id, [])

        return KnowledgeObject(
            ko_id=f"KO-EF-{ef_id}",
            ko_type="EFFile",
            primary_label=ef["name"],
            text_content=text,
            metadata={
                "ef_id":          ef_id,
                "fid":            ef["fid"],
                "name":           ef["name"],
                "file_type":      ef["file_type"],
                "default_value":  ef["default_value"],
                "section_id":     section_id,
                "parent_template": ef.get("parent_template"),
            },
            relationships=[_rel_summary(r) for r in rels],
        )


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _rel_summary(rel: dict) -> dict:
    return {
        "type":   rel["rel_type"],
        "target": rel["target_id"],
        "source": rel["source_id"],
    }


def print_ko_stats(objects: list[KnowledgeObject]):
    from collections import Counter
    counts = Counter(o.ko_type for o in objects)
    print(f"  Total knowledge objects: {len(objects)}")
    for ko_type, count in counts.most_common():
        print(f"    {ko_type:<20} {count:>5}")