"""
src/error_transformer/models.py
-----------------------------------------
Data models for the error transformation pipeline.

Three dataclasses represent the three stages of processing:
  ValidatorError   — one raw row from the validator CSV output
  RetrievedContext — spec context retrieved from the Hybrid/Vector Index
  TransformedError — human-friendly output ready for display / export
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidatorError:
    """One row from the validator output CSV."""

    validation_rule: str
    element_path: str
    description: str
    expected_value: str
    saip_value: str
    standard: str
    severity: str
    index: int = 0
    status: str = "Fail"
    rule_set: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "ValidatorError":
        return cls(
            validation_rule = d.get("validation_rule", d.get("Validation Rule", "")),
            element_path    = d.get("element_path",    d.get("Element Path", "")),
            description     = d.get("description",     d.get("Description", "")),
            expected_value  = d.get("expected_value",  d.get("Expected Value", "")),
            saip_value      = d.get("saip_value",      d.get("SAIP Value", "")),
            standard        = d.get("standard",        d.get("Standard Referenced", "")),
            severity        = d.get("severity",        d.get("Severity", "")),
            index           = int(d.get("index",       d.get("Index", 0))),
            status          = d.get("status",          d.get("Status", "Fail")),
            rule_set        = d.get("rule_set",        d.get("SAIP Rule Set", "")),
        )


@dataclass
class RetrievedContext:
    """Spec context assembled from the Hybrid/Vector Index for one error."""

    primary_ko: Optional[dict]           # ValidationRule KO
    field_ko: Optional[dict]             # Field KO (if distinct from VR)
    pe_ko: Optional[dict]                # Parent PE KO
    type_ko: Optional[dict]              # Referenced type KO (e.g. PEHeader)
    requirements: list[dict]             # linked Requirement KOs
    section_body: str                    # raw section body text
    section_id: str
    section_title: str
    found_expected_value: Optional[str]  # from spec (may differ from validator)

    def to_context_text(self) -> str:
        """Build the context string to inject into the LLM prompt."""
        parts = []

        if self.primary_ko:
            parts.append(f"[Validation Rule]\n{self.primary_ko['text_content']}")

        if self.pe_ko:
            parts.append(f"[Parent Profile Element]\n{self.pe_ko['text_content'][:600]}")

        if self.type_ko:
            parts.append(f"[Referenced Type]\n{self.type_ko['text_content'][:400]}")

        if self.section_body:
            parts.append(
                f"[Section {self.section_id} — {self.section_title}]\n"
                f"{self.section_body[:600]}"
            )

        if self.requirements:
            req_text = "\n".join(
                f"- [{r['metadata'].get('strength', '').upper()}] "
                f"{r['text_content'][:200]}"
                for r in self.requirements[:5]
            )
            parts.append(f"[Normative Requirements]\n{req_text}")

        return "\n\n".join(parts)


@dataclass
class TransformedError:
    """Human-friendly error output, ready for display or JSON export."""

    message: str
    cause: str
    correction: str
    reference: str
    location: str
    severity: str
    section_id: str
    raw_error: dict
    context_used: bool = True

    def to_dict(self) -> dict:
        return {
            "message":    self.message,
            "cause":      self.cause,
            "correction": self.correction,
            "reference":  self.reference,
            "terjadi_di": self.terjadi_di,
            "severity":   self.severity,
            "section_id": self.section_id,
        }

    def format_display(self) -> str:
        return (
            f"Message    : {self.message}\n"
            f"Cause      : {self.cause}\n"
            f"Correction : {self.correction}\n"
            f"Reference  : {self.reference}\n"
            f"Location   : {self.location}\n"
            f"Severity   : {self.severity}"
        )
