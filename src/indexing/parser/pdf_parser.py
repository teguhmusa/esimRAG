# src/pdf_parser.py
"""
PDF Parser for eUICC Profile Package Technical Specification.
Extracts raw structured content: pages, blocks, font metadata.
"""

import re
import fitz  # PyMuPDF
import pdfplumber
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


@dataclass
class TextSpan:
    text: str
    size: float
    bold: bool
    italic: bool
    font: str
    color: int


@dataclass
class TextBlock:
    """A logical block of text with consistent formatting."""
    spans: list[TextSpan]
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1
    block_type: str  # 'text' | 'asn1' | 'heading' | 'page_header'

    @property
    def text(self) -> str:
        return " ".join(s.text.strip() for s in self.spans if s.text.strip())

    @property
    def is_bold(self) -> bool:
        return any(s.bold for s in self.spans)

    @property
    def font_size(self) -> float:
        sizes = [s.size for s in self.spans if s.text.strip()]
        return max(sizes) if sizes else 0.0


@dataclass
class RawTable:
    """Raw table as extracted by pdfplumber."""
    rows: list[list[Optional[str]]]
    bbox: tuple[float, float, float, float]
    page_num: int


@dataclass
class ParsedPage:
    """All content extracted from a single PDF page."""
    page_num: int          # 1-indexed
    text_blocks: list[TextBlock]
    tables: list[RawTable]
    raw_text: str


class PDFParser:
    """
    Low-level PDF parser for eUICC spec documents.
    Produces ParsedPage objects with text blocks and tables.
    """

    # Font size thresholds observed in this spec
    HEADING_SIZE = 10.0
    BODY_SIZE = 10.0
    FOOTER_SIZE = 8.0
    TITLE_SIZE = 12.0

    # Page header/footer text to strip
    BOILERPLATE = {"Enabling trust in a connected future"}

    def __init__(self, pdf_path: str | Path):
        self.pdf_path = Path(pdf_path)
        self._fitz_doc: Optional[fitz.Document] = None
        self._plumber_doc = None

    def __enter__(self):
        self._fitz_doc = fitz.open(str(self.pdf_path))
        self._plumber_doc = pdfplumber.open(str(self.pdf_path))
        return self

    def __exit__(self, *args):
        if self._fitz_doc:
            self._fitz_doc.close()
        if self._plumber_doc:
            self._plumber_doc.close()

    @property
    def page_count(self) -> int:
        return len(self._fitz_doc)

    def parse_page(self, page_num: int) -> ParsedPage:
        """
        Parse a single page (1-indexed).
        Returns structured text blocks and tables.
        """
        fitz_page = self._fitz_doc[page_num - 1]
        plumber_page = self._plumber_doc.pages[page_num - 1]

        text_blocks = self._extract_text_blocks(fitz_page)
        tables = self._extract_tables(plumber_page, page_num)
        raw_text = plumber_page.extract_text() or ""

        return ParsedPage(
            page_num=page_num,
            text_blocks=text_blocks,
            tables=tables,
            raw_text=raw_text,
        )

    def parse_all(self, start: int = 1, end: Optional[int] = None) -> list[ParsedPage]:
        """Parse a range of pages (1-indexed, inclusive)."""
        end = end or self.page_count
        pages = []
        for page_num in range(start, end + 1):
            pages.append(self.parse_page(page_num))
        return pages

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _extract_text_blocks(self, fitz_page: fitz.Page) -> list[TextBlock]:
        """
        Extract text blocks with font metadata using PyMuPDF.

        Key challenge: fitz groups heading + body into one raw block when they
        are visually adjacent. We split each raw block at formatting transitions
        (bold→non-bold and heading-pattern detection) to produce clean blocks.

        Skips boilerplate (page headers/footers).
        """
        blocks = []
        raw_blocks = fitz_page.get_text("dict")["blocks"]

        for raw_block in raw_blocks:
            if raw_block["type"] != 0:  # skip image blocks
                continue

            block_text = self._get_block_text(raw_block)
            if not block_text.strip():
                continue
            if self._is_boilerplate(block_text):
                continue

            # Split raw block into logical sub-blocks at formatting transitions
            sub_blocks = self._split_block_by_format(raw_block)
            for spans, sub_text, bbox in sub_blocks:
                if not spans or not sub_text.strip():
                    continue
                block = TextBlock(spans=spans, bbox=bbox, block_type="text")
                block.block_type = self._classify_block(block, sub_text)
                blocks.append(block)

        return blocks

    def _split_block_by_format(
        self, raw_block: dict
    ) -> list[tuple[list[TextSpan], str, tuple]]:
        """
        Split a raw fitz block into sub-blocks at:
        1. bold↔non-bold transitions (heading vs body)
        2. ASN.1 boundary markers (-- ASN1START / -- ASN1STOP)

        Returns list of (spans, text, bbox).
        """
        lines = raw_block.get("lines", [])
        if not lines:
            return []

        groups = []          # list of (line_group, group_type)
        current_group = []
        current_is_bold = None

        for line in lines:
            line_spans = line.get("spans", [])
            line_text = " ".join(s.get("text", "") for s in line_spans).strip()
            if not line_text:
                continue

            # Check for ASN.1 boundary markers → force a split
            is_asn1_marker = ("-- ASN1START" in line_text or "-- ASN1STOP" in line_text)
            if is_asn1_marker:
                if current_group:
                    groups.append(current_group)
                groups.append([line])   # ASN1 marker gets its own group
                current_group = []
                current_is_bold = None
                continue

            # Determine dominant boldness for this line
            bold_chars = sum(
                len(s.get("text", "")) for s in line_spans
                if "Bold" in s.get("font", "") or "bold" in s.get("font", "")
            )
            total_chars = sum(len(s.get("text", "")) for s in line_spans) or 1
            line_is_bold = (bold_chars / total_chars) > 0.4

            # Detect bold↔non-bold transition → start new group
            if current_is_bold is not None and line_is_bold != current_is_bold:
                if current_group:
                    groups.append(current_group)
                current_group = []

            current_group.append(line)
            current_is_bold = line_is_bold

        if current_group:
            groups.append(current_group)

        # Convert each group of lines → (spans, text, bbox)
        result = []
        block_bbox = raw_block["bbox"]
        for group in groups:
            spans = []
            texts = []
            y_coords = []
            for line in group:
                y_coords.append(line["bbox"][1])
                y_coords.append(line["bbox"][3])
                for span in line.get("spans", []):
                    t = span.get("text", "").strip()
                    if not t:
                        continue
                    font = span.get("font", "")
                    spans.append(TextSpan(
                        text=span.get("text", ""),
                        size=span.get("size", 10.0),
                        bold="Bold" in font or "bold" in font,
                        italic="Italic" in font or "italic" in font,
                        font=font,
                        color=span.get("color", 0),
                    ))
                    texts.append(t)
            if spans:
                y_min = min(y_coords) if y_coords else block_bbox[1]
                y_max = max(y_coords) if y_coords else block_bbox[3]
                bbox = (block_bbox[0], y_min, block_bbox[2], y_max)
                result.append((spans, " ".join(texts), bbox))

        return result

    def _extract_spans(self, raw_block: dict) -> list[TextSpan]:
        """Flatten all spans from a raw fitz block."""
        spans = []
        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text:
                    continue
                font = span.get("font", "")
                spans.append(TextSpan(
                    text=span.get("text", ""),
                    size=span.get("size", 10.0),
                    bold="Bold" in font or "bold" in font,
                    italic="Italic" in font or "italic" in font,
                    font=font,
                    color=span.get("color", 0),
                ))
        return spans

    def _get_block_text(self, raw_block: dict) -> str:
        """Get plain text from a raw fitz block."""
        parts = []
        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                parts.append(span.get("text", ""))
        return " ".join(parts)

    def _is_boilerplate(self, text: str) -> bool:
        """Detect page header/footer boilerplate."""
        stripped = text.strip()
        # Exact boilerplate match
        if stripped in self.BOILERPLATE:
            return True
        # Page numbers: standalone integers (≤ 4 digits)
        if re.match(r"^\d{1,4}$", stripped):
            return True
        # Combined header+page: "Enabling trust in a connected future 46"
        for bp in self.BOILERPLATE:
            if stripped.startswith(bp):
                return True
        return False

    def _classify_block(self, block: TextBlock, raw_text: str) -> str:
        """
        Classify block type based on content and formatting.
        Types: heading | asn1 | page_header | text

        Note: In this spec, headings occupy their own fitz block (bold spans),
        separate from body text. Classification is based on the FIRST bold span.
        """
        stripped = raw_text.strip()

        # ASN.1 markers (checked before heading - some ASN.1 lines start with bold)
        if "-- ASN1START" in stripped or "-- ASN1STOP" in stripped:
            return "asn1"
        if "::=" in stripped:
            return "asn1"
        # Pure ASN.1 comment lines
        if re.match(r"^--\s+", stripped):
            return "asn1"

        # Headings: bold span + starts with section number pattern
        # Only classify as heading if the BOLD part is the section number+title
        bold_text = " ".join(s.text.strip() for s in block.spans if s.bold and s.text.strip())
        if bold_text:
            if re.match(r"^\d+(\.\d+)*\s+\w", bold_text):
                return "heading"
            # Standalone section number without title (e.g. "8." alone)
            if re.match(r"^\d+(\.\d+)*\.?\s*$", bold_text):
                return "heading"

        return "text"

    def _extract_tables(self, plumber_page, page_num: int) -> list[RawTable]:
        """
        Extract tables using pdfplumber.
        Filters out phantom tables (split header cells) based on column count.
        """
        tables = []
        raw_tables = plumber_page.find_tables()

        for raw_table in raw_tables:
            rows = raw_table.extract()
            if not rows:
                continue

            # Filter out single-column "phantom" tables (pdfplumber splits
            # multi-line header cells into separate 1-col tables)
            col_count = max(len(row) for row in rows)
            if col_count <= 1:
                continue

            # Filter tables with too few rows (likely noise)
            if len(rows) < 2:
                continue

            bbox = raw_table.bbox
            tables.append(RawTable(
                rows=rows,
                bbox=bbox,
                page_num=page_num,
            ))

        return tables