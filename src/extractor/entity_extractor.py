"""
src/indexing/extractor/entity_extractor.py
------------------------------------------
Entity Extractor for the eUICC Profile Package specification.

Sources consumed:
  asn1_blocks   → ASN1TypeDef, Field, Constraint, DefaultValue
  tables        → EFFile entities with metadata
  requirements  → ErrorCode entities

All produced entity dataclasses are defined in .models and imported here.
This module contains only parsing and extraction logic.
"""

import re
from typing import Optional

from .models import ASN1TypeDef, EFFile, ErrorCode, Field, ValidationRule


# ─────────────────────────────────────────────────────────────────────────────
# ASN.1 content normaliser
# ─────────────────────────────────────────────────────────────────────────────

def normalize_asn1_content(content: str) -> str:
    """Re-expand collapsed ASN.1 content (fitz joins spans into one line)."""
    result = content
    result = re.sub(r'\s*}\s*', '\n}\n', result)
    result = re.sub(r'\s*(--\s)', r'\n-- ', result)
    result = re.sub(r'\s*(/\*)', r'\n/*', result)
    # Newline before field definitions after comma
    result = re.sub(
        r'([,])\s+([a-z][a-zA-Z0-9\-]*\s+(?:[A-Z]|OCTET|NULL|SEQUENCE|CHOICE|INTEGER|UTF8))',
        r'\1\n    \2',
        result,
    )
    # Newline before type definitions
    result = re.sub(r'([A-Za-z][A-Za-z0-9\-]*\s*::=)', r'\n\1', result)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Compiled regex patterns
# ─────────────────────────────────────────────────────────────────────────────

TYPEDEF_RE = re.compile(
    r"^([A-Za-z][A-Za-z0-9\-]*)\s*::=\s*"
    r"(SEQUENCE|INTEGER|CHOICE|OCTET STRING|UTF8String|NULL|BIT STRING"
    r"|IA5String|BOOLEAN|SET|ANY)"
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
    """
    Extracts structured entity objects from ASN.1 blocks, EF tables,
    and requirements data.  Returns a dict of entity lists suitable for
    downstream relationship extraction and knowledge-object building.
    """

    def __init__(self):
        self.type_defs: dict[str, ASN1TypeDef] = {}
        self.ef_files: list[EFFile] = []
        self.error_codes: dict[str, ErrorCode] = {}
        self.validation_rules: dict[str, ValidationRule] = {}

    def extract_all(
        self,
        asn1_blocks: list[dict],
        tables: list[dict],
        requirements: list[dict],
    ) -> dict:
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
            "type_defs":        list(self.type_defs.values()),
            "ef_files":         self.ef_files,
            "error_codes":      list(self.error_codes.values()),
            "validation_rules": list(self.validation_rules.values()),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # ASN.1 block processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_asn1_block(self, block: dict) -> None:
        content    = block.get("content", "")
        section_id = block.get("section_id", "")
        page_num   = block.get("page_num")

        expanded = normalize_asn1_content(content)
        lines = [l.strip() for l in expanded.split("\n")]

        typedef_positions: list[tuple[int, str, str]] = []
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
            end_line = (
                typedef_positions[pos_idx + 1][0]
                if pos_idx + 1 < len(typedef_positions)
                else len(lines)
            )
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

    def _parse_sequence_fields(
        self,
        lines: list[str],
        parent_type: str,
        section_id: str,
    ) -> list[Field]:
        fields = []
        for line in lines:
            if "::=" in line:
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

    def _parse_field_line(
        self,
        line: str,
        parent_type: str,
        section_id: str,
    ) -> Optional[Field]:
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

        name      = m.group("name")
        asn1_type = re.sub(r"\s+", " ", m.group("type").strip())
        rest      = m.group("rest").strip()

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

    def _parse_integer_enum(self, lines: list[str]) -> list[dict]:
        values = []
        for line in lines:
            for m in ENUM_VALUE_RE.finditer(line):
                values.append({"name": m.group(1), "value": int(m.group(2))})
        return values

    # ─────────────────────────────────────────────────────────────────────────
    # EF file processing
    # ─────────────────────────────────────────────────────────────────────────

    def _process_ef_table(self, table: dict) -> None:
        section_id      = table.get("section_id")
        parent_template = self._infer_parent_template(section_id)

        for record in table["records"]:
            data      = record["data"]
            ef_name   = data.get("EF Name", "").strip()
            fid       = data.get("FID", "").strip()
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

    def _infer_parent_template(self, section_id: Optional[str]) -> Optional[str]:
        if not section_id:
            return None
        mapping = {
            "9.2": "PE-MF",
            "9.3": "PE-USIM",
            "9.4": "PE-ISIM",
            "9.5": "PE-CSIM",
        }
        for prefix, pe_name in mapping.items():
            if section_id.startswith(prefix):
                return pe_name
        if section_id.startswith("9."):
            return f"PE-template-{section_id}"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Error code extraction
    # ─────────────────────────────────────────────────────────────────────────

    def _extract_error_codes_from_requirements(self, requirements: list[dict]) -> None:
        for req in requirements:
            section_id = req.get("section_id", "")
            for code in req.get("error_refs", []):
                if code not in self.error_codes:
                    self.error_codes[code] = ErrorCode(
                        code_id=code,
                        numeric_value=None,
                        description=self._humanize(code),
                        section_id=section_id,
                    )

    def _extract_error_codes_from_pestatus(self) -> None:
        if "PEStatus" not in self.type_defs:
            return
        pe_status   = self.type_defs["PEStatus"]
        enum_values = pe_status.enum_values if pe_status.enum_values else self._find_pestatus_enums()

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

    def _find_pestatus_enums(self) -> list[dict]:
        """
        Extract enum values from PEStatus nested INTEGER enum.
        Falls back to spec-accurate hardcoded values (Section 8.11) when the
        raw ASN.1 block cannot be parsed structurally.
        """
        if "PEStatus" not in self.type_defs:
            return []
        pe_status = self.type_defs["PEStatus"]
        for f in pe_status.fields:
            if f.name == "status" and "INTEGER" in f.asn1_type:
                break

        # Hardcoded fallback — values from TCA spec V3.3.1 Section 8.11
        return [
            {"name": "ok",                       "value": 0},
            {"name": "peNotSupported",           "value": 1},
            {"name": "memoryFailure",            "value": 2},
            {"name": "badValues",                "value": 3},
            {"name": "notEnoughMemory",          "value": 4},
            {"name": "invalidRequestFormat",     "value": 5},
            {"name": "invalidParameter",         "value": 6},
            {"name": "runtimeNotSupported",      "value": 7},
            {"name": "libNotSupported",          "value": 8},
            {"name": "templateNotSupported",     "value": 9},
            {"name": "featureNotSupported",      "value": 10},
            {"name": "pinCodeMissing",           "value": 11},
            {"name": "unsupportedProfileVersion","value": 31},
        ]

    # ─────────────────────────────────────────────────────────────────────────
    # Validation rule construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_validation_rules(self) -> None:
        for type_def in self.type_defs.values():
            for f in type_def.fields:
                path       = f"{type_def.type_id}.{f.name}"
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

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _normalize_path(self, path: str) -> str:
        parts = path.split(".")
        normalized = []
        for part in parts:
            part = re.sub(r"^PE-", "", part)
            part = part.replace("-", "_").lower()
            normalized.append(part)
        return ".".join(normalized)

    def _build_field_desc(self, f: Field) -> str:
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

    def _camel_to_kebab(self, name: str) -> str:
        s = re.sub(r"([A-Z])", r"-\1", name).lower()
        return s.lstrip("-")

    def _humanize(self, code_id: str) -> str:
        words = code_id.replace("-", " ").replace("_", " ")
        return words[0].upper() + words[1:] if words else code_id


# ─────────────────────────────────────────────────────────────────────────────
# Stats helpers
# ─────────────────────────────────────────────────────────────────────────────

def print_entity_stats(entities: dict) -> None:
    """Print stats from serialised entity dicts (post to_dict())."""
    type_defs = entities["type_defs"]
    print(f"  ASN.1 type defs  : {len(type_defs)}")
    print(f"    ProfileElements: {sum(1 for t in type_defs if t['is_pe'])}")
    print(f"    SEQUENCE types : {sum(1 for t in type_defs if t['asn1_kind'] == 'SEQUENCE')}")
    print(f"    Total fields   : {sum(len(t['fields']) for t in type_defs)}")
    print(f"  EF files         : {len(entities['ef_files'])}")
    print(f"  Error codes      : {len(entities['error_codes'])}")
    print(f"  Validation rules : {len(entities['validation_rules'])}")


def print_entity_stats_obj(entities: dict) -> None:
    """Print stats from raw entity objects (not yet serialised to dicts)."""
    type_defs = entities["type_defs"]
    print(f"  ASN.1 type defs  : {len(type_defs)}")
    print(f"    ProfileElements: {sum(1 for t in type_defs if t.is_pe)}")
    print(f"    SEQUENCE types : {sum(1 for t in type_defs if t.asn1_kind == 'SEQUENCE')}")
    print(f"    Total fields   : {sum(len(t.fields) for t in type_defs)}")
    print(f"  EF files         : {len(entities['ef_files'])}")
    print(f"  Error codes      : {len(entities['error_codes'])}")
    print(f"  Validation rules : {len(entities['validation_rules'])}")
