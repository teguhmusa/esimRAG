# src/relationship_extractor.py
"""
Pure-Python Relationship Extractor for eUICC spec entities.

Relationship types produced:
  BELONGS_TO      Field → parent ASN1TypeDef
  HAS_TYPE        Field → referenced ASN1TypeDef (when type is not primitive)
  HAS_FIELD       ASN1TypeDef → its Fields
  DEFINED_IN      ASN1TypeDef/EFFile → Section
  VALIDATED_BY    ValidationRule → Requirement(s)
  REFERENCES      Requirement → Field/ErrorCode
  CHILD_OF        Section → parent Section
  PE_CONTAINS     ProfileElement → Field
  ERROR_RAISED_BY ErrorCode → Section (where it's mentioned)
  HAS_DEFAULT     Field → expected/default value (scalar, not entity)
  PART_OF_PE      EFFile → ProfileElement (via section)
"""

import re
from dataclasses import dataclass
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Relationship:
    rel_id: str
    rel_type: str           # BELONGS_TO, HAS_TYPE, DEFINED_IN, etc.
    source_id: str          # entity that holds the relationship
    source_type: str        # entity type of source
    target_id: str          # entity being pointed at
    target_type: str        # entity type of target
    properties: dict        # additional metadata

    def to_dict(self):
        return {
            "rel_id":      self.rel_id,
            "rel_type":    self.rel_type,
            "source_id":   self.source_id,
            "source_type": self.source_type,
            "target_id":   self.target_id,
            "target_type": self.target_type,
            "properties":  self.properties,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Primitive type set (not navigable as entities)
# ─────────────────────────────────────────────────────────────────────────────

PRIMITIVE_TYPES = {
    "NULL", "BOOLEAN", "INTEGER", "OCTET STRING", "UTF8String",
    "IA5String", "BIT STRING", "OBJECT IDENTIFIER", "ANY",
    "SEQUENCE", "CHOICE", "SET",
}

def is_primitive(asn1_type: str) -> bool:
    base = asn1_type.split("(")[0].split(" OF ")[0].strip()
    return (
        base in PRIMITIVE_TYPES
        or base.startswith("UInt")
        or base.startswith("SEQUENCE")
        or base.startswith("CHOICE")
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class RelationshipExtractor:

    def __init__(self):
        self._counter = 0
        self._seen = set()

    def extract_all(
        self,
        entities: dict,
        sections: list[dict],
        requirements: list[dict],
    ) -> list[Relationship]:
        """
        Build all relationships from entity + section + requirement data.
        Returns flat list of Relationship objects.
        """
        rels: list[Relationship] = []

        # Build lookup indexes
        type_index  = {t["type_id"]: t for t in entities["type_defs"]}
        sec_index   = {s["section_id"]: s for s in sections}
        ef_index    = {e["ef_id"]: e for e in entities["ef_files"]}
        err_index   = {e["code_id"]: e for e in entities["error_codes"]}
        req_index   = {r["req_id"]: r for r in requirements}

        print("  → Field relationships (BELONGS_TO, HAS_TYPE, HAS_DEFAULT)...")
        rels += self._field_relationships(entities["type_defs"], type_index)

        print("  → Type → Section (DEFINED_IN)...")
        rels += self._type_section_relationships(entities["type_defs"], sec_index)

        print("  → EFFile → Section + PE (DEFINED_IN, PART_OF_PE)...")
        rels += self._ef_relationships(entities["ef_files"], sec_index, type_index)

        print("  → Section hierarchy (CHILD_OF)...")
        rels += self._section_hierarchy(sections)

        print("  → Requirement → Field/Error (REFERENCES)...")
        rels += self._requirement_references(requirements, type_index, err_index)

        print("  → ValidationRule → Requirements (VALIDATED_BY)...")
        rels += self._validation_rule_links(
            entities["validation_rules"], requirements, type_index
        )

        print("  → ErrorCode → Section (ERROR_RAISED_BY)...")
        rels += self._error_section_links(entities["error_codes"], requirements)

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Field relationships
    # ─────────────────────────────────────────────────────────────────────────

    def _field_relationships(self, type_defs, type_index):
        rels = []
        for tdef in type_defs:
            parent_id = tdef["type_id"]
            for f in tdef["fields"]:
                field_id = f["field_id"]

                # HAS_FIELD: TypeDef → Field
                rels.append(self._rel(
                    "HAS_FIELD", parent_id, "ASN1TypeDef",
                    field_id, "Field",
                    {"optional": f["is_optional"]},
                ))

                # BELONGS_TO: Field → TypeDef
                rels.append(self._rel(
                    "BELONGS_TO", field_id, "Field",
                    parent_id, "ASN1TypeDef", {},
                ))

                # HAS_TYPE: Field → referenced TypeDef (non-primitive)
                base_type = f["asn1_type"].split("(")[0].split(" OF ")[0].strip()
                if not is_primitive(base_type) and base_type in type_index:
                    rels.append(self._rel(
                        "HAS_TYPE", field_id, "Field",
                        base_type, "ASN1TypeDef",
                        {"declared_as": f["asn1_type"]},
                    ))

                # HAS_DEFAULT: Field → scalar value (stored as property)
                if f["default_value"]:
                    rels.append(self._rel(
                        "HAS_DEFAULT", field_id, "Field",
                        f["default_value"], "Value",
                        {"asn1_type": f["asn1_type"]},
                    ))

                # IS_OPTIONAL marker
                if f["is_optional"]:
                    rels.append(self._rel(
                        "IS_OPTIONAL", field_id, "Field",
                        parent_id, "ASN1TypeDef", {},
                    ))

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Type → Section
    # ─────────────────────────────────────────────────────────────────────────

    def _type_section_relationships(self, type_defs, sec_index):
        rels = []
        for tdef in type_defs:
            sec_id = tdef.get("section_id")
            if sec_id and sec_id in sec_index:
                rels.append(self._rel(
                    "DEFINED_IN", tdef["type_id"], "ASN1TypeDef",
                    sec_id, "Section",
                    {"page_num": tdef.get("page_num")},
                ))
                # PE_CONTAINS: reverse — Section defines this PE
                if tdef["is_pe"]:
                    rels.append(self._rel(
                        "CONTAINS_PE", sec_id, "Section",
                        tdef["type_id"], "ASN1TypeDef", {},
                    ))
        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 3. EFFile relationships
    # ─────────────────────────────────────────────────────────────────────────

    def _ef_relationships(self, ef_files, sec_index, type_index):
        rels = []
        for ef in ef_files:
            # DEFINED_IN: EFFile → Section
            if ef["section_id"] and ef["section_id"] in sec_index:
                rels.append(self._rel(
                    "DEFINED_IN", ef["ef_id"], "EFFile",
                    ef["section_id"], "Section",
                    {"fid": ef["fid"]},
                ))

            # PART_OF_PE: EFFile → ProfileElement
            if ef["parent_template"] and ef["parent_template"] in type_index:
                rels.append(self._rel(
                    "PART_OF_PE", ef["ef_id"], "EFFile",
                    ef["parent_template"], "ASN1TypeDef",
                    {"fid": ef["fid"], "file_type": ef["file_type"]},
                ))

            # HAS_DEFAULT_CONTENT: EFFile → value (when default_value is set)
            if ef["default_value"] and ef["default_value"] not in ("", "eUICC Platform dependant"):
                rels.append(self._rel(
                    "HAS_DEFAULT", ef["ef_id"], "EFFile",
                    ef["default_value"], "Value",
                    {"fid": ef["fid"]},
                ))

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 4. Section hierarchy
    # ─────────────────────────────────────────────────────────────────────────

    def _section_hierarchy(self, sections):
        rels = []
        sec_index = {s["section_id"]: s for s in sections}
        for s in sections:
            parent_id = s.get("parent_id")
            if parent_id and parent_id in sec_index:
                rels.append(self._rel(
                    "CHILD_OF", s["section_id"], "Section",
                    parent_id, "Section",
                    {"level": s["level"]},
                ))
        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 5. Requirement → Field / ErrorCode (REFERENCES)
    # ─────────────────────────────────────────────────────────────────────────

    def _requirement_references(self, requirements, type_index, err_index):
        rels = []

        # Build field name → field_id index for fast lookup
        field_name_index: dict[str, list[str]] = {}
        for tdef in type_index.values():
            for f in tdef["fields"]:
                fname = f["name"]
                if fname not in field_name_index:
                    field_name_index[fname] = []
                field_name_index[fname].append(f["field_id"])

        for req in requirements:
            req_id = req["req_id"]

            # REFERENCES → Field
            for field_name in req.get("field_refs", []):
                # Try direct field name match
                if field_name in field_name_index:
                    for field_id in field_name_index[field_name][:3]:  # cap at 3
                        rels.append(self._rel(
                            "REFERENCES", req_id, "Requirement",
                            field_id, "Field",
                            {"mention_type": "field_ref"},
                        ))
                # Try as type name
                elif field_name in type_index:
                    rels.append(self._rel(
                        "REFERENCES", req_id, "Requirement",
                        field_name, "ASN1TypeDef",
                        {"mention_type": "type_ref"},
                    ))

            # REFERENCES → ErrorCode
            for err_code in req.get("error_refs", []):
                if err_code in err_index:
                    rels.append(self._rel(
                        "REFERENCES", req_id, "Requirement",
                        err_code, "ErrorCode",
                        {"mention_type": "error_ref"},
                    ))

            # DEFINED_IN → Section (requirement lives in a section)
            rels.append(self._rel(
                "DEFINED_IN", req_id, "Requirement",
                req["section_id"], "Section",
                {"strength": req["strength"], "keyword": req["keyword"]},
            ))

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 6. ValidationRule → Requirements (VALIDATED_BY)
    # ─────────────────────────────────────────────────────────────────────────

    def _validation_rule_links(self, val_rules, requirements, type_index):
        rels = []

        # Index: section_id → requirements in that section
        sec_to_reqs: dict[str, list[dict]] = {}
        for req in requirements:
            sid = req["section_id"]
            if sid not in sec_to_reqs:
                sec_to_reqs[sid] = []
            sec_to_reqs[sid].append(req)

        # Build field_ref → req index
        field_to_reqs: dict[str, list[str]] = {}
        for req in requirements:
            for fref in req.get("field_refs", []):
                if fref not in field_to_reqs:
                    field_to_reqs[fref] = []
                field_to_reqs[fref].append(req["req_id"])

        seen_pairs = set()
        for vr in val_rules:
            rule_id = vr["rule_id"]
            field_ref = vr.get("field_ref", "")
            section_id = vr.get("section_id", "")

            # Link via field_ref name match in requirements
            matched_reqs = field_to_reqs.get(field_ref, [])

            # Also link via same section requirements with SHALL/MUST
            if section_id in sec_to_reqs:
                for req in sec_to_reqs[section_id]:
                    if req["strength"] in ("mandatory", "prohibited"):
                        matched_reqs.append(req["req_id"])

            # Deduplicate and create relationships
            for req_id in dict.fromkeys(matched_reqs):  # preserves order, dedupes
                pair = (rule_id, req_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    rels.append(self._rel(
                        "VALIDATED_BY", rule_id, "ValidationRule",
                        req_id, "Requirement",
                        {"section_id": section_id},
                    ))

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # 7. ErrorCode → Section (ERROR_RAISED_BY)
    # ─────────────────────────────────────────────────────────────────────────

    def _error_section_links(self, error_codes, requirements):
        rels = []

        # Build error_code → sections where it's mentioned
        err_to_sections: dict[str, set[str]] = {}
        for req in requirements:
            for code in req.get("error_refs", []):
                if code not in err_to_sections:
                    err_to_sections[code] = set()
                err_to_sections[code].add(req["section_id"])

        for ec in error_codes:
            code_id = ec["code_id"]
            # Primary section (where it's defined)
            if ec["section_id"]:
                rels.append(self._rel(
                    "DEFINED_IN", code_id, "ErrorCode",
                    ec["section_id"], "Section",
                    {"numeric_value": ec.get("numeric_value")},
                ))
            # Sections where it's referenced in requirements
            for sec_id in err_to_sections.get(code_id, set()):
                if sec_id != ec["section_id"]:
                    rels.append(self._rel(
                        "RAISED_IN", code_id, "ErrorCode",
                        sec_id, "Section", {},
                    ))

        return rels

    # ─────────────────────────────────────────────────────────────────────────
    # Helper
    # ─────────────────────────────────────────────────────────────────────────

    def _rel(
        self,
        rel_type: str,
        source_id: str,
        source_type: str,
        target_id: str,
        target_type: str,
        properties: dict,
    ) -> Relationship:
        self._counter += 1
        return Relationship(
            rel_id=f"R{self._counter:05d}",
            rel_type=rel_type,
            source_id=source_id,
            source_type=source_type,
            target_id=target_id,
            target_type=target_type,
            properties=properties,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def print_relationship_stats(relationships: list[Relationship]):
    from collections import Counter
    counts = Counter(r.rel_type for r in relationships)
    print(f"  Total relationships: {len(relationships)}")
    for rel_type, count in counts.most_common():
        print(f"    {rel_type:<25} {count:>5}")