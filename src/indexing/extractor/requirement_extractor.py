# src/extractor/requirement_extractor.py
"""
Requirement Extractor for eUICC Profile Package spec.

Extracts structured normative requirements from section body text.
Each requirement has:
  - req_id         : unique identifier
  - section_id     : source section
  - keyword        : SHALL / SHALL NOT / MUST / SHOULD / MAY / etc.
  - strength       : mandatory | prohibited | recommended | optional | conditional
  - subject        : who the requirement applies to (eUICC, Profile Creator, etc.)
  - condition      : if/when clause (for conditional requirements)
  - statement      : the full requirement sentence
  - req_type       : simple | conditional | usage_rule | note
  - asn1_refs      : ASN.1 type names mentioned in the requirement
  - field_refs     : quoted field/parameter names
  - error_refs     : error codes
  - source_text    : original sentence as extracted
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Requirement:
    req_id: str
    section_id: str
    section_title: str
    keyword: str
    strength: str
    subject: str
    condition: Optional[str]
    statement: str
    req_type: str
    asn1_refs: list
    field_refs: list
    error_refs: list
    source_text: str

    def to_dict(self):
        return {
            "req_id": self.req_id,
            "section_id": self.section_id,
            "section_title": self.section_title,
            "keyword": self.keyword,
            "strength": self.strength,
            "subject": self.subject,
            "condition": self.condition,
            "statement": self.statement,
            "req_type": self.req_type,
            "asn1_refs": self.asn1_refs,
            "field_refs": self.field_refs,
            "error_refs": self.error_refs,
            "source_text": self.source_text,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Patterns
# ─────────────────────────────────────────────────────────────────────────────

NORMATIVE_KEYWORDS = [
    "SHALL NOT", "MUST NOT", "SHOULD NOT", "MAY NOT",
    "SHALL", "MUST", "SHOULD", "MAY",
]

STRENGTH_MAP = {
    "SHALL NOT":  "prohibited",
    "MUST NOT":   "prohibited",
    "MAY NOT":    "prohibited",
    "SHOULD NOT": "discouraged",
    "SHALL":      "mandatory",
    "MUST":       "mandatory",
    "SHOULD":     "recommended",
    "MAY":        "optional",
}

ACTORS = [
    ("eUICC",           r"\bthe\s+eUICC\b|\beUICC\b"),
    ("Profile Creator", r"\bthe\s+Profile\s+Creator\b|\bProfile\s+Creator\b"),
    ("Profile Package", r"\bthe\s+Profile\s+Package\b|\bProfile\s+Package\b"),
    ("MNO",             r"\bMNO\b"),
    ("SM-DP+",          r"SM-DP\+"),
    ("Profile",         r"\bthe\s+Profile\b(?!\s+(?:Creator|Package))"),
]

CONDITION_PATTERNS = [
    re.compile(r"^(If\s+[^,;]{5,120}(?:,|then))\s*", re.I),
    re.compile(r"^(When\s+[^,;]{5,120}(?:,|then))\s*", re.I),
    re.compile(r"^(In case\s+[^,;]{5,120},)\s*", re.I),
    re.compile(r"^(Unless\s+[^,;]{5,120},)\s*", re.I),
]

ASN1_TYPE_RE = re.compile(r"\b([A-Z][A-Za-z0-9]+(?:-[A-Z][A-Za-z0-9]+)*)\b")
FIELD_REF_RE = re.compile(r'"\s*([a-zA-Z][a-zA-Z0-9_\-]+)\s*"')
ERROR_REF_RE = re.compile(r'"([a-z][a-z0-9]+(?:-[a-z0-9]+){1,4})"')

USAGE_RULES_RE = re.compile(r"^\s*Usage\s+rules?\s*:\s*", re.I)
NOTE_RE = re.compile(r"^\s*NOTE\s*[:\d]*\s*", re.I)


# ─────────────────────────────────────────────────────────────────────────────
# Sentence splitter
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(text):
    if not text:
        return []
    text = re.sub(r"\s+", " ", text).strip()

    # Protect abbreviations and decimal numbers
    text = text.replace("e.g.", "e_g_")
    text = text.replace("i.e.", "i_e_")
    text = text.replace("etc.", "etc_")
    text = re.sub(r"(V\d+)\.(\d+)", r"\1_DOT_\2", text)
    text = re.sub(r"(\d+)\.(\d+)", r"\1_DOTNUM_\2", text)
    text = re.sub(r"(Annex\s+[A-Z])\.", r"\1_DOT_", text)
    text = re.sub(r"(section\s+\d+)\.(\d+)", r"\1_DOT_\2", text, flags=re.I)

    parts = re.split(r"(?<=[.;])\s+", text)

    result = []
    for part in parts:
        part = part.replace("e_g_", "e.g.").replace("i_e_", "i.e.").replace("etc_", "etc.")
        part = re.sub(r"(\w+)_DOT_(\w+)", r"\1.\2", part)
        part = re.sub(r"(\d+)_DOTNUM_(\d+)", r"\1.\2", part)
        part = part.strip()
        if len(part) > 10:  # skip very short fragments
            result.append(part)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class RequirementExtractor:

    def extract_from_sections(self, sections):
        requirements = []
        req_counter = 1

        for section in sections:
            section_id = section["section_id"]
            section_title = section["title"]
            body = section.get("body_text", "")
            if not body:
                continue

            sentences = split_sentences(body)
            for sentence in sentences:
                req = self._extract_requirement(
                    sentence, section_id, section_title, req_counter
                )
                if req:
                    requirements.append(req)
                    req_counter += 1

        return requirements

    def _extract_requirement(self, sentence, section_id, section_title, req_counter):
        keyword, keyword_pos = self._find_keyword(sentence)
        if not keyword:
            return None

        req_type = self._classify_req_type(sentence)
        strength = STRENGTH_MAP.get(keyword, "unknown")
        if req_type == "conditional":
            strength = "conditional"

        subject = self._extract_subject(sentence, keyword_pos)
        condition = self._extract_condition(sentence)
        asn1_refs = self._extract_asn1_refs(sentence)
        field_refs = self._extract_field_refs(sentence)
        error_refs = self._extract_error_refs(sentence)
        statement = self._clean_statement(sentence)

        req_id = f"REQ-{section_id}-{req_counter:04d}"

        return Requirement(
            req_id=req_id,
            section_id=section_id,
            section_title=section_title,
            keyword=keyword,
            strength=strength,
            subject=subject,
            condition=condition,
            statement=statement,
            req_type=req_type,
            asn1_refs=asn1_refs,
            field_refs=field_refs,
            error_refs=error_refs,
            source_text=sentence,
        )

    def _find_keyword(self, sentence):
        upper = sentence.upper()
        for kw in NORMATIVE_KEYWORDS:
            idx = upper.find(kw)
            if idx != -1:
                return kw, idx
        return None, -1

    def _classify_req_type(self, sentence):
        if USAGE_RULES_RE.match(sentence):
            return "usage_rule"
        if NOTE_RE.match(sentence):
            return "note"
        for pattern in CONDITION_PATTERNS:
            if pattern.match(sentence):
                return "conditional"
        return "simple"

    def _extract_subject(self, sentence, keyword_pos):
        prefix = sentence[:keyword_pos] if keyword_pos > 0 else sentence
        for actor, pattern in ACTORS:
            if re.search(pattern, prefix, re.I):
                return actor
        for actor, pattern in ACTORS:
            if re.search(pattern, sentence, re.I):
                return actor
        return "generic"

    def _extract_condition(self, sentence):
        for pattern in CONDITION_PATTERNS:
            m = pattern.match(sentence)
            if m:
                condition = m.group(1).strip()
                condition = re.sub(r"[,]?\s*(then)?\s*$", "", condition, flags=re.I)
                return condition.strip()
        return None

    def _extract_asn1_refs(self, sentence):
        candidates = ASN1_TYPE_RE.findall(sentence)
        refs = []
        for c in candidates:
            if (
                c.startswith("PE-")
                or c.startswith("PUK")
                or c.startswith("PIN")
                or c.startswith("Profile")
                or c.startswith("UInt")
                or c.startswith("Services")
                or (len(c) >= 6 and c[0].isupper() and any(ch.islower() for ch in c[1:]))
            ):
                if c not in refs and len(c) > 3:
                    refs.append(c)
        return refs[:8]

    def _extract_field_refs(self, sentence):
        matches = FIELD_REF_RE.findall(sentence)
        seen, refs = set(), []
        for m in matches:
            if m not in seen:
                seen.add(m)
                refs.append(m)
        return refs

    def _extract_error_refs(self, sentence):
        return ERROR_REF_RE.findall(sentence)

    def _clean_statement(self, sentence):
        stmt = sentence.strip()
        stmt = USAGE_RULES_RE.sub("", stmt)
        stmt = NOTE_RE.sub("", stmt)
        stmt = re.sub(r'"\s+([^"]+)\s+"', r'"\1"', stmt)
        stmt = re.sub(r"^[•\-\*]\s*", "", stmt)
        return stmt.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Stats helper
# ─────────────────────────────────────────────────────────────────────────────

def print_stats(requirements):
    from collections import Counter

    print(f"Total requirements: {len(requirements)}")
    print()
    print("By keyword:")
    for kw, count in Counter(r.keyword for r in requirements).most_common():
        print(f"  {kw:<12} {count:>4}")
    print()
    print("By strength:")
    for st, count in Counter(r.strength for r in requirements).most_common():
        print(f"  {st:<15} {count:>4}")
    print()
    print("By type:")
    for ty, count in Counter(r.req_type for r in requirements).most_common():
        print(f"  {ty:<15} {count:>4}")
    print()
    print("By subject:")
    for su, count in Counter(r.subject for r in requirements).most_common():
        print(f"  {su:<20} {count:>4}")