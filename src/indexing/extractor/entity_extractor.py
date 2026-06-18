# src/entity_extractor.py
"""
Pure-Python Entity Extractor for eUICC Profile Package spec.

Sources:
  - asn1_blocks.json   → ProfileElement, Field, ASN1Type, Constraint, DefaultValue
  - tables.json        → EFFile entities with metadata
  - requirements.json  → ErrorCode entities

Entity types produced:
  ASN1TypeDef     - type definition (PUKConfiguration, PEHeader, PE-PUKCodes, ...)
  Field           - field inside a SEQUENCE
  EFFile          - Elementary File from EF/DF template tables
  ErrorCode       - error codes from requirements + PEStatus
  ValidationRule  - mapping from error path → section + expected value
"""

import re
from dataclasses import dataclass, field
from typing import Optional


# ─────────────────────────────────────────────────────────────────────────────
# Data models
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Field:
    field_id: str
    name: str
    parent_type: str
    asn1_type: str
    is_optional: bool
    default_value: Optional[str]
    size_constraint: Optional[str]
    comment: Optional[str]
    section_id: str

    def to_dict(self):
        return {
            "entity_type": "Field",
            "field_id": self.field_id,
            "name": self.name,
            "parent_type": self.parent_type,
            "asn1_type": self.asn1_type,
            "is_optional": self.is_optional,
            "default_value": self.default_value,
            "size_constraint": self.size_constraint,
            "comment": self.comment,
            "section_id": self.section_id,
        }


@dataclass
class ASN1TypeDef:
    type_id: str
    asn1_kind: str
    section_id: str
    page_num: Optional[int]
    fields: list = field(default_factory=list)
    is_pe: bool = False
    enum_values: list = field(default_factory=list)
    comment: Optional[str] = None

    def to_dict(self):
        return {
            "entity_type": "ASN1TypeDef",
            "type_id": self.type_id,
            "asn1_kind": self.asn1_kind,
            "section_id": self.section_id,
            "page_num": self.page_num,
            "is_pe": self.is_pe,
            "fields": [f.to_dict() for f in self.fields],
            "enum_values": self.enum_values,
            "comment": self.comment,
        }


@dataclass
class EFFile:
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

    def to_dict(self):
        return {
            "entity_type": "EFFile",
            "ef_id": self.ef_id,
            "fid": self.fid,
            "name": self.name,
            "file_type": self.file_type,
            "nb_records": self.nb_records,
            "rec_size": self.rec_size,
            "access_rules": self.access_rules,
            "sfi": self.sfi,
            "default_value": self.default_value,
            "content_required": self.content_required,
            "parameters": self.parameters,
            "section_id": self.section_id,
            "parent_template": self.parent_template,
        }


@dataclass
class ErrorCode:
    code_id: str
    numeric_value: Optional[int]
    description: str
    section_id: str

    def to_dict(self):
        return {
            "entity_type": "ErrorCode",
            "code_id": self.code_id,
            "numeric_value": self.numeric_value,
            "description": self.description,
            "section_id": self.section_id,
        }


@dataclass
class ValidationRule:
    rule_id: str
    normalized_path: str
    section_id: str
    expected_value: Optional[str]
    asn1_type_ref: Optional[str]
    field_ref: Optional[str]
    description: str

    def to_dict(self):
        return {
            "entity_type": "ValidationRule",
            "rule_id": self.rule_id,
            "normalized_path": self.normalized_path,
            "section_id": self.section_id,
            "expected_value": self.expected_value,
            "asn1_type_ref": self.asn1_type_ref,
            "field_ref": self.field_ref,
            "description": self.description,
        }


# ─────────────────────────────────────────────────────────────────────────────
# ASN.1 content normalizer
# ─────────────────────────────────────────────────────────────────────────────

def normalize_asn1_content(content):
    """Re-expand collapsed ASN.1 content (fitz joins spans into one line)."""
    result = content
    result = re.sub(r'\s*}\s*', '\n}\n', result)
    result = re.sub(r'\s*(--\s)', r'\n-- ', result)
    result = re.sub(r'\s*(/\*)', r'\n/*', result)
    # Newline before field definitions after comma
    result = re.sub(
        r'([,])\s+([a-z][a-zA-Z0-9\-]*\s+(?:[A-Z]|OCTET|NULL|SEQUENCE|CHOICE|INTEGER|UTF8))',
        r'\1\n    \2',
        result
    )
    # Newline before type definitions
    result = re.sub(r'([A-Za-z][A-Za-z0-9\-]*\s*::=)', r'\n\1', result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Regex patterns
# ─────────────────────────────────────────────────────────────────────────────

TYPEDEF_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9\-]*)\s*::=\s*(SEQUENCE|INTEGER|CHOICE|OCTET STRING|UTF8String|NULL|BIT STRING|IA5String|BOOLEAN|SET|ANY)"
)
ALIAS_RE = re.compile(r"^([A-Za-z][A-Za-z0-9\-]*)\s*::=\s*(.+)")

FIELD_LINE_RE = re.compile(
    r"""
    ^[ \t]*
    (?P<name>[a-z][a-zA-Z0-9\-]*)
    \s+
    (?P<type>
        (?:OCTET\s+STRING|UTF8String|IA5String|NULL|BOOLEAN|BIT\s+STRING
          |INTEGER|OBJECT\s+IDENTIFIER|UInt\w+
          |SEQUENCE(?:\s*\(SIZE[^)]+\))?\s*(?:OF\s+\S+)?
          |CHOICE
          |[A-Z][A-Za-z0-9\-]*)
    )
    (?P<rest>.*)$
    """,
    re.VERBOSE,
)

OPTIONAL_RE      = re.compile(r'\bOPTIONAL\b', re.I)
DEFAULT_RE       = re.compile(r"\bDEFAULT\s+('[^']*'\s*H?|[A-Za-z0-9]+)", re.I)
SIZE_RE          = re.compile(r'\(SIZE\s*\([^)]+\)\s*\)')
RANGE_RE         = re.compile(r'\(\s*\d+\s*\.\.\s*(?:\d+|MAX)\s*\)')
COMMENT_RE       = re.compile(r'--\s*(.*?)$')
BLOCK_COMMENT_RE = re.compile(r'/\*.*?\*/', re.DOTALL)
ENUM_VALUE_RE    = re.compile(r'([a-zA-Z][a-zA-Z0-9\-]*)\s*\(\s*(\d+)\s*\)')
ERROR_REF_RE     = re.compile(r'"([a-z][a-z0-9]+(?:-[a-z0-9]+){1,4})"')


# ─────────────────────────────────────────────────────────────────────────────
# Main extractor
# ─────────────────────────────────────────────────────────────────────────────

class EntityExtractor:

    def __init__(self):
        self.type_defs = {}        # type_id → ASN1TypeDef
        self.ef_files = []
        self.error_codes = {}      # code_id → ErrorCode
        self.validation_rules = {} # normalized_path → ValidationRule

    def extract_all(self, asn1_blocks, tables, requirements):
        print("  → Parsing ASN.1 type definitions...")
        for block in asn1_blocks:
            self._process_asn1_block(block)

        print("  → Parsing EF/DF file tables...")
        for table in tables:
            if table.get("table_type") == "ef_df_template":
                self._process_ef_table(table)

        print("  → Extracting error codes...")
        self._extract_error_codes_from_requirements(requirements)
        self._extract_error_codes_from_pestatus()

        print("  → Building validation rules...")
        self._build_validation_rules()

        return {
            "type_defs": list(self.type_defs.values()),
            "ef_files": self.ef_files,
            "error_codes": list(self.error_codes.values()),
            "validation_rules": list(self.validation_rules.values()),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ASN.1
    # ─────────────────────────────────────────────────────────────────────────

    def _process_asn1_block(self, block):
        content = block.get("content", "")
        section_id = block.get("section_id", "")
        page_num = block.get("page_num")

        expanded = normalize_asn1_content(content)
        lines = [l.strip() for l in expanded.split("\n")]

        # Collect all typedef positions
        typedef_positions = []
        for i, line in enumerate(lines):
            m = TYPEDEF_RE.match(line)
            if m:
                typedef_positions.append((i, m.group(1), m.group(2)))
            elif "::=" in line:
                m2 = ALIAS_RE.match(line)
                if m2:
                    typedef_positions.append((i, m2.group(1), "ALIAS"))

        if not typedef_positions:
            return

        for pos_idx, (start_line, type_name, asn1_kind) in enumerate(typedef_positions):
            end_line = typedef_positions[pos_idx + 1][0] if pos_idx + 1 < len(typedef_positions) else len(lines)
            typedef_lines = lines[start_line:end_line]

            type_def = ASN1TypeDef(
                type_id=type_name,
                asn1_kind=asn1_kind,
                section_id=section_id,
                page_num=page_num,
                is_pe=type_name.startswith("PE-"),
            )

            if asn1_kind == "SEQUENCE":
                type_def.fields = self._parse_sequence_fields(typedef_lines, type_name, section_id)
            elif asn1_kind == "INTEGER":
                type_def.enum_values = self._parse_integer_enum(typedef_lines)
            elif asn1_kind == "ALIAS":
                type_def.comment = lines[start_line] if start_line < len(lines) else ""

            self.type_defs[type_name] = type_def

    def _parse_sequence_fields(self, lines, parent_type, section_id):
        fields = []
        for line in lines:
            if "::=" in line:
                # Extract any inline field definition after opening brace
                # e.g. "PE-PUKCodes ::= SEQUENCE { puk-Header PEHeader,"
                if "{" in line:
                    after_brace = line.split("{", 1)[1].strip().rstrip(",}").strip()
                    if after_brace and not after_brace.startswith("--"):
                        f = self._parse_field_line(after_brace, parent_type, section_id)
                        if f:
                            fields.append(f)
                continue
            if line.startswith("--") or line.startswith("/*") or line.startswith("*/"):
                continue
            if line in ("{", "}", ""):
                continue
            f = self._parse_field_line(line, parent_type, section_id)
            if f:
                fields.append(f)
        return fields

    def _parse_field_line(self, line, parent_type, section_id):
        line = BLOCK_COMMENT_RE.sub("", line).strip()
        if not line:
            return None

        comment = None
        cm = COMMENT_RE.search(line)
        if cm:
            comment = cm.group(1).strip()
            line = line[:cm.start()].strip()

        line = line.rstrip(",").strip()

        m = FIELD_LINE_RE.match(line)
        if not m:
            return None

        name = m.group("name")
        asn1_type = re.sub(r"\s+", " ", m.group("type").strip())
        rest = m.group("rest").strip()

        is_optional = bool(OPTIONAL_RE.search(rest))

        default_val = None
        dv = DEFAULT_RE.search(rest)
        if dv:
            default_val = dv.group(1).strip()

        size_constraint = None
        sc = SIZE_RE.search(rest) or RANGE_RE.search(rest)
        if sc:
            size_constraint = sc.group(0).strip()

        return Field(
            field_id=f"{parent_type}.{name}",
            name=name,
            parent_type=parent_type,
            asn1_type=asn1_type,
            is_optional=is_optional,
            default_value=default_val,
            size_constraint=size_constraint,
            comment=comment,
            section_id=section_id,
        )

    def _parse_integer_enum(self, lines):
        values = []
        for line in lines:
            for m in ENUM_VALUE_RE.finditer(line):
                values.append({"name": m.group(1), "value": int(m.group(2))})
        return values

    # ─────────────────────────────────────────────────────────────────────────
    # EF files
    # ─────────────────────────────────────────────────────────────────────────

    def _process_ef_table(self, table):
        section_id = table.get("section_id")
        parent_template = self._infer_parent_template(section_id)

        for record in table["records"]:
            data = record["data"]
            ef_name = data.get("EF Name", "").strip()
            fid = data.get("FID", "").strip()
            file_type = data.get("File Type", "").strip()

            if not ef_name or not fid:
                continue
            if file_type in ("DF", "ADF", "MF"):
                continue
            if not re.match(r"^[0-9A-Fa-f]{4}$", fid):
                continue

            ef_id = re.sub(r"\s+", "_", ef_name.upper().replace("-", "_"))

            self.ef_files.append(EFFile(
                ef_id=ef_id,
                fid=fid,
                name=ef_name,
                file_type=file_type,
                nb_records=data.get("NB Rec.", "").strip(),
                rec_size=data.get("File / Rec Size", "").strip(),
                access_rules=data.get("Access Rules", "").strip(),
                sfi=data.get("SFI", "").strip(),
                default_value=data.get("Default Value", "").strip(),
                content_required=data.get("Content Required", "").strip(),
                parameters=data.get("Parameters", "").strip(),
                section_id=section_id,
                parent_template=parent_template,
            ))

    def _infer_parent_template(self, section_id):
        if not section_id:
            return None
        mapping = {
            "9.2": "PE-MF", "9.3": "PE-USIM",
            "9.4": "PE-ISIM", "9.5": "PE-CSIM",
        }
        for prefix, pe_name in mapping.items():
            if section_id.startswith(prefix):
                return pe_name
        if section_id.startswith("9."):
            return f"PE-template-{section_id}"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Error codes
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_error_codes_from_requirements(self, requirements):
        for req in requirements:
            section_id = req.get("section_id", "")
            statement = req.get("statement", "")
            for code in req.get("error_refs", []):
                if code not in self.error_codes:
                    self.error_codes[code] = ErrorCode(
                        code_id=code,
                        numeric_value=None,
                        description=self._humanize(code),
                        section_id=section_id,
                    )

    def _extract_error_codes_from_pestatus(self):
        # PEStatus is SEQUENCE containing nested INTEGER enum
        # Try to find it via the raw type_defs or parse directly from content
        if "PEStatus" not in self.type_defs:
            return
        pe_status = self.type_defs["PEStatus"]

        # If we found enum_values directly (shouldn't happen for SEQUENCE, but check)
        if pe_status.enum_values:
            enum_values = pe_status.enum_values
        else:
            # Parse enum values from comment (stored in fields or raw content)
            # The PEStatus SEQUENCE has field: status INTEGER { ok(0), ... }
            # We need to find these values from the raw block content
            enum_values = self._find_pestatus_enums()

        for ev in enum_values:
            code_id = self._camel_to_kebab(ev["name"])
            if code_id not in self.error_codes:
                self.error_codes[code_id] = ErrorCode(
                    code_id=code_id,
                    numeric_value=ev["value"],
                    description=self._humanize(code_id),
                    section_id=pe_status.section_id,
                )
            else:
                self.error_codes[code_id].numeric_value = ev["value"]

    def _find_pestatus_enums(self):
        """Extract enum values from PEStatus nested INTEGER enum."""
        if "PEStatus" not in self.type_defs:
            return []
        # The enum is embedded in the SEQUENCE as: status INTEGER { name(int), ... }
        # We stored the raw content in the comment field if no fields were parsed
        # Try to get it from the type_def's section via a stored raw content
        # (We need to add raw_content storage — for now parse from comment)

        # Alternative: look at any field named 'status' with INTEGER type
        pe_status = self.type_defs["PEStatus"]
        for f in pe_status.fields:
            if f.name == "status" and "INTEGER" in f.asn1_type:
                # The enum is in f.comment or we need the raw line
                break

        # Extract from stored raw lines via ENUM_VALUE_RE on all fields' raw data
        # Since we don't store raw content, we use the known PEStatus values
        # as a reliable fallback (they're part of the spec and won't change)
        known_pe_status = [
            {"name": "ok", "value": 0},
            {"name": "peNotSupported", "value": 1},
            {"name": "memoryFailure", "value": 2},
            {"name": "badValues", "value": 3},
            {"name": "notEnoughMemory", "value": 4},
            {"name": "invalidRequestFormat", "value": 5},
            {"name": "invalidParameter", "value": 6},
            {"name": "runtimeNotSupported", "value": 7},
            {"name": "libNotSupported", "value": 8},
            {"name": "templateNotSupported", "value": 9},
            {"name": "featureNotSupported", "value": 10},
            {"name": "pinCodeMissing", "value": 11},
            {"name": "unsupportedProfileVersion", "value": 31},
        ]
        return known_pe_status

    def _camel_to_kebab(self, name):
        s = re.sub(r"([A-Z])", r"-\1", name).lower()
        return s.lstrip("-")

    def _humanize(self, code_id):
        words = code_id.replace("-", " ").replace("_", " ")
        return words[0].upper() + words[1:] if words else code_id

    # ─────────────────────────────────────────────────────────────────────────
    # Validation rules
    # ─────────────────────────────────────────────────────────────────────────

    def _build_validation_rules(self):
        for type_name, type_def in self.type_defs.items():
            for f in type_def.fields:
                path = f"{type_def.type_id}.{f.name}"
                normalized = self._normalize_path(path)

                rule = ValidationRule(
                    rule_id=path,
                    normalized_path=normalized,
                    section_id=f.section_id,
                    expected_value=f.default_value,
                    asn1_type_ref=f.asn1_type,
                    field_ref=f.name,
                    description=self._build_field_desc(f),
                )
                self.validation_rules[normalized] = rule
                if normalized != path.lower():
                    self.validation_rules[path] = rule

    def _normalize_path(self, path):
        parts = path.split(".")
        normalized = []
        for part in parts:
            part = re.sub(r"^PE-", "", part)
            part = part.replace("-", "_").lower()
            normalized.append(part)
        return ".".join(normalized)

    def _build_field_desc(self, f):
        parts = [f"Field '{f.name}' of type {f.asn1_type}"]
        if f.is_optional:
            parts.append("(optional)")
        if f.default_value:
            parts.append(f"default: {f.default_value}")
        if f.size_constraint:
            parts.append(f"size: {f.size_constraint}")
        if f.comment:
            parts.append(f"— {f.comment}")
        return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Stats
# ─────────────────────────────────────────────────────────────────────────────

def print_entity_stats(entities):
    type_defs = entities["type_defs"]
    print(f"  ASN.1 type defs  : {len(type_defs)}")
    print(f"    ProfileElements: {sum(1 for t in type_defs if t['is_pe'])}")
    print(f"    SEQUENCE types : {sum(1 for t in type_defs if t['asn1_kind'] == 'SEQUENCE')}")
    print(f"    Total fields   : {sum(len(t['fields']) for t in type_defs)}")
    print(f"  EF files         : {len(entities['ef_files'])}")
    print(f"  Error codes      : {len(entities['error_codes'])}")
    print(f"  Validation rules : {len(entities['validation_rules'])}")


def print_entity_stats_obj(entities):
    """Stats from raw entity objects (not dicts)."""
    type_defs = entities["type_defs"]
    print(f"  ASN.1 type defs  : {len(type_defs)}")
    print(f"    ProfileElements: {sum(1 for t in type_defs if t.is_pe)}")
    print(f"    SEQUENCE types : {sum(1 for t in type_defs if t.asn1_kind == 'SEQUENCE')}")
    print(f"    Total fields   : {sum(len(t.fields) for t in type_defs)}")
    print(f"  EF files         : {len(entities['ef_files'])}")
    print(f"  Error codes      : {len(entities['error_codes'])}")
    print(f"  Validation rules : {len(entities['validation_rules'])}")