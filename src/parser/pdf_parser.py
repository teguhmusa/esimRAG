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
        end = end or self.page_count
        pages = []
        for page_num in range(start, end + 1):
            pages.append(self.parse_page(page_num))
        return pages

    def _extract_text_blocks(self, fitz_page: fitz.Page) -> list[TextBlock]:
        blocks = []
        raw_blocks = fitz_page.get_text("dict")["blocks"]

        for raw_block in raw_blocks:
            if raw_block["type"] != 0:
                continue

            block_text = self._get_block_text(raw_block)
            if not block_text.strip():
                continue
            if self._is_boilerplate(block_text):
                continue

            sub_blocks = self._split_block_by_format(raw_block)
            for spans, sub_text, bbox in sub_blocks:
                if not spans or not sub_text.strip():
                    continue
                block = TextBlock(spans=spans, bbox=bbox, block_type="text")
                block.block_type = self._classify_block(block, sub_text)
                blocks.append(block)

        return blocks

    def _split_block_by_format(self, raw_block: dict) -> list[tuple[list[TextSpan], str, tuple]]:
        lines = raw_block.get("lines", [])
        if not lines:
            return []

        groups = []
        current_group = []
        current_is_bold = None

        for line in lines:
            line_spans = line.get("spans", [])
            line_text = " ".join(s.get("text", "") for s in line_spans).strip()
            if not line_text:
                continue

            is_asn1_marker = ("-- ASN1START" in line_text or "-- ASN1STOP" in line_text)
            if is_asn1_marker:
                if current_group:
                    groups.append(current_group)
                groups.append([line])
                current_group = []
                current_is_bold = None
                continue

            bold_chars = sum(
                len(s.get("text", "")) for s in line_spans
                if "Bold" in s.get("font", "") or "bold" in s.get("font", "")
            )
            total_chars = sum(len(s.get("text", "")) for s in line_spans) or 1
            line_is_bold = (bold_chars / total_chars) > 0.4

            if current_is_bold is not None and line_is_bold != current_is_bold:
                if current_group:
                    groups.append(current_group)
                current_group = []

            current_group.append(line)
            current_is_bold = line_is_bold

        if current_group:
            groups.append(current_group)

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
        parts = []
        for line in raw_block.get("lines", []):
            for span in line.get("spans", []):
                parts.append(span.get("text", ""))
        return " ".join(parts)

    def _is_boilerplate(self, text: str) -> bool:
        stripped = text.strip()
        if stripped in self.BOILERPLATE:
            return True
        if re.match(r"^\d{1,4}$", stripped):
            return True
        for bp in self.BOILERPLATE:
            if stripped.startswith(bp):
                return True
        return False

    def _classify_block(self, block: TextBlock, raw_text: str) -> str:
        stripped = raw_text.strip()

        if "-- ASN1START" in stripped or "-- ASN1STOP" in stripped:
            return "asn1"
        if "::=" in stripped:
            return "asn1"
        if re.match(r"^--\s+", stripped):
            return "asn1"

        bold_text = " ".join(s.text.strip() for s in block.spans if s.bold and s.text.strip())
        if bold_text:
            if re.match(r"^\d+(\.\d+)*\s+\w", bold_text):
                return "heading"
            if re.match(r"^\d+(\.\d+)*\.?\s*$", bold_text):
                return "heading"

        return "text"

    def _extract_tables(self, plumber_page, page_num: int) -> list[RawTable]:
        tables = []
        raw_tables = plumber_page.find_tables()

        for raw_table in raw_tables:
            rows = raw_table.extract()
            if not rows:
                continue

            col_count = max(len(row) for row in rows)
            if col_count <= 1:
                continue
            if len(rows) < 2:
                continue

            bbox = raw_table.bbox
            tables.append(RawTable(
                rows=rows,
                bbox=bbox,
                page_num=page_num,
            ))

        return tables
