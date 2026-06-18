# src/section_extractor.py
"""
Section Extractor for eUICC spec.
Builds a hierarchical section tree from parsed pages.
Each section captures: id, title, page range, body text, ASN.1 blocks, subsections.
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.indexing.parser.pdf_parser import ParsedPage, TextBlock


@dataclass
class ASN1Block:
    """A single ASN.1 definition block."""
    content: str
    type_name: Optional[str] = None   # e.g. "PE-PUKCodes", "PUKConfiguration"
    asn1_type: Optional[str] = None   # SEQUENCE, INTEGER, CHOICE, etc.
    section_id: Optional[str] = None
    page_num: Optional[int] = None


@dataclass
class Section:
    """A single section/subsection of the spec."""
    section_id: str          # e.g. "8.5.2"
    title: str               # e.g. "PUK Code PE"
    level: int               # depth: 1=top, 2=sub, 3=subsub
    page_start: int
    page_end: Optional[int] = None
    body_text: str = ""
    asn1_blocks: list[ASN1Block] = field(default_factory=list)
    parent_id: Optional[str] = None
    children: list["Section"] = field(default_factory=list)

    @property
    def full_title(self) -> str:
        return f"{self.section_id} {self.title}"

    @property
    def has_asn1(self) -> bool:
        return len(self.asn1_blocks) > 0

    def to_dict(self) -> dict:
        return {
            "section_id": self.section_id,
            "title": self.title,
            "level": self.level,
            "page_start": self.page_start,
            "page_end": self.page_end,
            "body_text": self.body_text,
            "asn1_blocks": [
                {
                    "type_name": b.type_name,
                    "asn1_type": b.asn1_type,
                    "content": b.content,
                    "page_num": b.page_num,
                }
                for b in self.asn1_blocks
            ],
            "parent_id": self.parent_id,
            "children": [c.section_id for c in self.children],
        }


# Regex for section headings like "8.5.2" or "12." or "3.1"
HEADING_RE = re.compile(r"^(\d+(?:\.\d+)*)\s+(.*)")

# ASN.1 type name: e.g. "PE-PUKCodes ::= SEQUENCE {"
ASN1_TYPEDEF_RE = re.compile(r"^([A-Za-z][A-Za-z0-9_\-]*)\s*::=\s*(SEQUENCE|INTEGER|CHOICE|ENUMERATED|OCTET STRING|UTF8String|NULL|BIT STRING|IA5String|BOOLEAN|SET|ANY)")


class SectionExtractor:
    """
    Extracts a hierarchical section tree from a list of ParsedPages.

    Algorithm:
    1. Walk all blocks across pages in order
    2. When a heading block is found → open a new Section
    3. Accumulate body text and ASN.1 blocks into current section
    4. Close & store section when next heading found
    5. Build parent-child tree based on section number depth
    """

    def __init__(self, skip_toc_pages: int = 6):
        """
        skip_toc_pages: pages to skip at the start (cover + TOC).
        Default 6 matches this spec (cover, copyright, TOC pages).
        """
        self.skip_toc_pages = skip_toc_pages

    def extract(self, pages: list[ParsedPage]) -> list[Section]:
        """
        Main entry point. Returns flat list of sections.
        Call build_tree() after to get parent-child relationships.
        """
        sections: list[Section] = []
        current_section: Optional[Section] = None
        current_body: list[str] = []
        current_asn1_lines: list[str] = []
        in_asn1 = False

        for page in pages:
            if page.page_num <= self.skip_toc_pages:
                continue

            for block in page.text_blocks:
                text = block.text.strip()
                if not text:
                    continue

                # ── ASN.1 block handling ──────────────────────────────────
                if "-- ASN1START" in text:
                    in_asn1 = True
                    current_asn1_lines = []
                    continue

                if "-- ASN1STOP" in text:
                    in_asn1 = False
                    if current_section and current_asn1_lines:
                        asn1_block = self._parse_asn1_block(
                            "\n".join(current_asn1_lines),
                            page.page_num,
                            current_section.section_id,
                        )
                        current_section.asn1_blocks.append(asn1_block)
                    current_asn1_lines = []
                    continue

                if in_asn1:
                    current_asn1_lines.append(text)
                    continue

                # ── Heading detection ─────────────────────────────────────
                if block.block_type == "heading":
                    match = HEADING_RE.match(text)
                    if match:
                        # Close current section
                        if current_section:
                            current_section.body_text = " ".join(current_body).strip()
                            current_section.page_end = page.page_num
                            sections.append(current_section)

                        section_id = match.group(1)
                        title = match.group(2).strip()
                        level = section_id.count(".") + 1

                        current_section = Section(
                            section_id=section_id,
                            title=title,
                            level=level,
                            page_start=page.page_num,
                        )
                        current_body = []
                        current_asn1_lines = []
                        continue

                # ── Body text accumulation ────────────────────────────────
                if current_section and block.block_type == "text":
                    current_body.append(text)

        # Close final section
        if current_section:
            current_section.body_text = " ".join(current_body).strip()
            sections.append(current_section)

        # Build hierarchy
        self._build_tree(sections)
        return sections

    def build_index(self, sections: list[Section]) -> dict[str, Section]:
        """Build a lookup dict: section_id → Section."""
        return {s.section_id: s for s in sections}

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _parse_asn1_block(self, content: str, page_num: int, section_id: str) -> ASN1Block:
        """
        Parse an ASN.1 block to extract the primary type name and kind.
        Handles multi-definition blocks by taking the first typedef.
        """
        type_name = None
        asn1_type = None

        for line in content.split("\n"):
            line = line.strip()
            m = ASN1_TYPEDEF_RE.match(line)
            if m:
                type_name = m.group(1)
                asn1_type = m.group(2)
                break  # take first definition as primary

        return ASN1Block(
            content=content,
            type_name=type_name,
            asn1_type=asn1_type,
            section_id=section_id,
            page_num=page_num,
        )

    def _build_tree(self, sections: list[Section]) -> None:
        """
        Assign parent_id and children to each section in-place.
        Uses section numbering to determine hierarchy.
        E.g. 8.5 is parent of 8.5.2; 8 is parent of 8.5.
        """
        index = {s.section_id: s for s in sections}

        for section in sections:
            parts = section.section_id.split(".")
            if len(parts) > 1:
                parent_id = ".".join(parts[:-1])
                section.parent_id = parent_id
                if parent_id in index:
                    index[parent_id].children.append(section)