"""
src/indexing/extractor/models.py
---------------------------------
Domain data-model dataclasses for the eUICC spec extraction pipeline.

All entity and relationship dataclasses that were previously scattered
across entity_extractor.py and relationship_extractor.py are consolidated
here so that extractor modules contain only processing logic.

Entity hierarchy:
  Field          — a single field inside an ASN.1 SEQUENCE
  ASN1TypeDef    — an ASN.1 type definition (SEQUENCE, INTEGER, CHOICE, …)
  EFFile         — an Elementary File from an EF/DF template table
  ErrorCode      — an error code extracted from requirements or PEStatus
  ValidationRule — a path → expected-value rule derived from type fields
  Relationship   — a directed edge between two entities in the knowledge graph
"""

from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Entity models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Field:
    """A single field inside an ASN.1 SEQUENCE type definition."""

    field_id: str
    name: str
    parent_type: str
    asn1_type: str
    is_optional: bool
    default_value: Optional[str]
    size_constraint: Optional[str]
    comment: Optional[str]
    section_id: str

    def to_dict(self) -> dict:
        return {
            "entity_type":     "Field",
            "field_id":        self.field_id,
            "name":            self.name,
            "parent_type":     self.parent_type,
            "asn1_type":       self.asn1_type,
            "is_optional":     self.is_optional,
            "default_value":   self.default_value,
            "size_constraint": self.size_constraint,
            "comment":         self.comment,
            "section_id":      self.section_id,
        }


@dataclass
class ASN1TypeDef:
    """An ASN.1 type definition (SEQUENCE, INTEGER, CHOICE, etc.)."""

    type_id: str
    asn1_kind: str
    section_id: str
    page_num: Optional[int]
    fields: list = field(default_factory=list)
    is_pe: bool = False
    enum_values: list = field(default_factory=list)
    comment: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "entity_type": "ASN1TypeDef",
            "type_id":     self.type_id,
            "asn1_kind":   self.asn1_kind,
            "section_id":  self.section_id,
            "page_num":    self.page_num,
            "is_pe":       self.is_pe,
            "fields":      [f.to_dict() for f in self.fields],
            "enum_values": self.enum_values,
            "comment":     self.comment,
        }


@dataclass
class EFFile:
    """An Elementary File entry from an EF/DF template table."""

    ef_id: str
    fid: str
    name: str
    file_type: str
    nb_records: str
    rec_size: str
    access_rules: str
    sfi: str
    default_value: str
    content_required: str
    parameters: str
    section_id: Optional[str]
    parent_template: Optional[str]

    def to_dict(self) -> dict:
        return {
            "entity_type":     "EFFile",
            "ef_id":           self.ef_id,
            "fid":             self.fid,
            "name":            self.name,
            "file_type":       self.file_type,
            "nb_records":      self.nb_records,
            "rec_size":        self.rec_size,
            "access_rules":    self.access_rules,
            "sfi":             self.sfi,
            "default_value":   self.default_value,
            "content_required": self.content_required,
            "parameters":      self.parameters,
            "section_id":      self.section_id,
            "parent_template": self.parent_template,
        }


@dataclass
class ErrorCode:
    """An error code, either from normative requirements or PEStatus enum."""

    code_id: str
    numeric_value: Optional[int]
    description: str
    section_id: str

    def to_dict(self) -> dict:
        return {
            "entity_type":   "ErrorCode",
            "code_id":       self.code_id,
            "numeric_value": self.numeric_value,
            "description":   self.description,
            "section_id":    self.section_id,
        }


@dataclass
class ValidationRule:
    """A rule mapping a dotted field path to an expected/default value."""

    rule_id: str
    normalized_path: str
    section_id: str
    expected_value: Optional[str]
    asn1_type_ref: Optional[str]
    field_ref: Optional[str]
    description: str

    def to_dict(self) -> dict:
        return {
            "entity_type":    "ValidationRule",
            "rule_id":        self.rule_id,
            "normalized_path": self.normalized_path,
            "section_id":     self.section_id,
            "expected_value": self.expected_value,
            "asn1_type_ref":  self.asn1_type_ref,
            "field_ref":      self.field_ref,
            "description":    self.description,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Relationship model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Relationship:
    """
    A directed edge in the knowledge graph between two entities.

    Relationship types:
      BELONGS_TO    Field → parent ASN1TypeDef
      HAS_TYPE      Field → referenced ASN1TypeDef (when type is not primitive)
      HAS_FIELD     ASN1TypeDef → its Fields
      DEFINED_IN    ASN1TypeDef/EFFile → Section
      VALIDATED_BY  ValidationRule → Requirement(s)
      REFERENCES    Requirement → Field/ErrorCode
      CHILD_OF      Section → parent Section
      PE_CONTAINS   ProfileElement → Field
      ERROR_RAISED_BY ErrorCode → Section (where it's mentioned)
      HAS_DEFAULT   Field → expected/default value (scalar, not entity)
      PART_OF_PE    EFFile → ProfileElement (via section)
    """

    rel_id: str
    rel_type: str
    source_id: str
    source_type: str
    target_id: str
    target_type: str
    properties: dict

    def to_dict(self) -> dict:
        return {
            "rel_id":      self.rel_id,
            "rel_type":    self.rel_type,
            "source_id":   self.source_id,
            "source_type": self.source_type,
            "target_id":   self.target_id,
            "target_type": self.target_type,
            "properties":  self.properties,
        }
