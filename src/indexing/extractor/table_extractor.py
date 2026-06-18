# src/table_extractor.py
"""
Table Extractor for eUICC spec.
Normalizes raw pdfplumber tables into clean, structured records.

eUICC spec tables have these challenges:
  1. Multi-line header cells (e.g. "File /\nRec\nSize")
  2. Merged cells leave None in adjacent cells
  3. Split tables across pages
  4. Known table types: EF/DF templates, encoding tables, bit-field tables
"""

import re
from dataclasses import dataclass, field
from typing import Optional
from src.indexing.parser.pdf_parser import RawTable, ParsedPage


@dataclass
class TableRecord:
    """A single normalized row from a spec table."""
    row_index: int
    data: dict[str, str]    # column_name → value
    raw_row: list[Optional[str]]


@dataclass
class SpecTable:
    """A normalized spec table with header and records."""
    table_id: str                       # e.g. "p70_t0"
    page_num: int
    section_id: Optional[str]          # assigned later by SectionExtractor
    table_type: str                     # 'ef_df_template' | 'encoding' | 'bit_field' | 'generic'
    headers: list[str]
    records: list[TableRecord]
    raw: RawTable

    def to_dict(self) -> dict:
        return {
            "table_id": self.table_id,
            "page_num": self.page_num,
            "section_id": self.section_id,
            "table_type": self.table_type,
            "headers": self.headers,
            "records": [
                {"row_index": r.row_index, "data": r.data}
                for r in self.records
            ],
        }


# Column names that identify EF/DF template tables (eUICC file system tables)
EF_DF_COLUMNS = {"FID", "EF Name", "File Type", "Default Value", "Access Rules"}

# Column names that identify bit-field tables
BIT_FIELD_COLUMNS = {"Bit", "Meaning"}

# Column names that identify encoding/parameter tables
ENCODING_COLUMNS = {"Value", "Meaning", "Description"}


class TableExtractor:
    """
    Extracts and normalizes tables from ParsedPages.
    Assigns section context based on page position relative to sections.
    """

    def extract(self, pages: list[ParsedPage]) -> list[SpecTable]:
        """Extract all tables from all pages."""
        tables = []
        table_counter = 0

        for page in pages:
            for raw_table in page.tables:
                normalized = self._normalize_table(raw_table, table_counter)
                if normalized:
                    tables.append(normalized)
                    table_counter += 1

        return tables

    def assign_sections(
        self,
        tables: list[SpecTable],
        section_index: dict,  # section_id → Section
    ) -> None:
        """
        Assign section_id to each table based on page proximity.
        Modifies tables in-place.
        """
        # Build page → section_id mapping (last section starting on or before page)
        page_to_section: dict[int, str] = {}
        for sid, section in section_index.items():
            for p in range(section.page_start, (section.page_end or section.page_start) + 1):
                page_to_section[p] = sid

        for table in tables:
            table.section_id = page_to_section.get(table.page_num)

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    def _normalize_table(self, raw: RawTable, idx: int) -> Optional[SpecTable]:
        """
        Convert a RawTable into a SpecTable.
        Handles:
          - Multi-row headers (joined with space)
          - None cells (forward-fill from previous row)
          - Phantom columns from PDF merged-cell rendering
          - Empty/noise tables
        """
        rows = raw.rows
        if not rows or len(rows) < 2:
            return None

        # Step 1: Find header rows
        header_rows, data_start = self._detect_header_rows(rows)
        if not header_rows:
            return None

        # Step 2: Merge multi-row headers, get clean headers + phantom map
        headers, real_col_indices = self._merge_header_rows_with_map(header_rows)
        if not headers:
            return None

        raw_col_count = max(len(r) for r in rows)

        # Step 3: Build records using only real column indices
        records = []
        prev_row = {h: "" for h in headers}
        for row_idx, row in enumerate(rows[data_start:]):
            # Pad row to raw column count
            padded = [row[i] if i < len(row) else None for i in range(raw_col_count)]

            # Extract only real columns
            extracted = {}
            for header, col_idx in zip(headers, real_col_indices):
                cell = padded[col_idx] if col_idx < len(padded) else None
                extracted[header] = " ".join(cell.split()) if cell else ""

            # Skip completely empty rows
            if all(not v.strip() for v in extracted.values()):
                continue

            # Forward-fill: if entire row is empty → continuation
            all_none = all(
                padded[i] is None
                for i in real_col_indices
                if i < len(padded)
            )
            if all_none:
                extracted = dict(prev_row)

            records.append(TableRecord(
                row_index=row_idx,
                data=extracted,
                raw_row=row,
            ))
            prev_row = extracted

        if not records:
            return None

        table_type = self._classify_table(headers)

        return SpecTable(
            table_id=f"p{raw.page_num}_t{idx}",
            page_num=raw.page_num,
            section_id=None,
            table_type=table_type,
            headers=headers,
            records=records,
            raw=raw,
        )

    def _detect_header_rows(
        self, rows: list[list[Optional[str]]]
    ) -> tuple[list[list[Optional[str]]], int]:
        """
        Detect which rows are headers vs data.
        Headers typically contain column labels (text) without numeric-only cells.
        Returns (header_rows, data_start_index).
        """
        # Strategy: first non-empty row(s) that look like labels
        header_rows = []
        data_start = 0

        for i, row in enumerate(rows[:4]):  # headers are in first 4 rows max
            cells = [c for c in row if c is not None and c.strip()]
            if not cells:
                continue

            # Check if this looks like a header row:
            # - Contains known column label keywords
            # - OR is the first row
            is_header = i == 0 or self._looks_like_header(row)

            if is_header:
                header_rows.append(row)
                data_start = i + 1
            else:
                break

        return header_rows, data_start

    def _looks_like_header(self, row: list[Optional[str]]) -> bool:
        """Heuristic: row is a header if it contains mostly text, no hex values."""
        cells = [c.strip() for c in row if c and c.strip()]
        if not cells:
            return False
        # If any cell is a known header keyword → header
        header_keywords = {
            "FID", "EF Name", "File Type", "NB Rec", "SFI", "Default Value",
            "Access Rules", "Parameters", "Version", "Content Required",
            "Bit", "Meaning", "Value", "Description", "Condition",
            "Ass.", "File", "Size", "Name", "Type", "Serv",
        }
        for cell in cells:
            for kw in header_keywords:
                if kw.lower() in cell.lower():
                    return True
        return False

    def _merge_header_rows_with_map(
        self, header_rows: list[list[Optional[str]]]
    ) -> tuple[list[str], list[int]]:
        """
        Merge multi-row headers into single list of column names.
        Returns (headers, real_col_indices) where real_col_indices maps
        each header to its position in the raw row.

        Collapses phantom columns (None cells from merged cells in PDF).
        """
        if not header_rows:
            return [], []

        col_count = max(len(r) for r in header_rows)
        padded = [
            [row[i] if i < len(row) else None for i in range(col_count)]
            for row in header_rows
        ]

        # Merge multi-line cell text per column
        merged_raw = []
        for col_idx in range(col_count):
            parts = []
            for row in padded:
                cell = row[col_idx]
                if cell and cell.strip():
                    parts.append(" ".join(cell.split()))
            merged_raw.append((" ".join(parts), col_idx))

        # Keep only columns with a real header name
        real = [(name, idx) for name, idx in merged_raw if name]

        headers = [name for name, _ in real]
        indices = [idx for _, idx in real]
        return headers, indices

    def _merge_header_rows(self, header_rows: list[list[Optional[str]]]) -> list[str]:
        """Convenience wrapper — returns only header names."""
        headers, _ = self._merge_header_rows_with_map(header_rows)
        return headers

    def _normalize_row(
        self,
        row: list[Optional[str]],
        col_count: int,
        prev_row: list[str],
    ) -> list[str]:
        """
        Normalize a data row:
        - Pad/trim to col_count
        - Replace None with empty string (not forward-fill for data rows,
          forward-fill only for continuation rows with ALL None)
        """
        padded = [row[i] if i < len(row) else None for i in range(col_count)]

        # If entire row is None → continuation of previous (forward-fill)
        if all(c is None for c in padded):
            return list(prev_row)

        result = []
        for i, cell in enumerate(padded):
            if cell is None:
                result.append("")
            else:
                # Clean multi-line within cell
                result.append(" ".join(cell.split()))

        return result

    def _classify_table(self, headers: list[str]) -> str:
        """Classify table type based on column headers."""
        header_set = set(h.strip() for h in headers)

        if EF_DF_COLUMNS & header_set:
            return "ef_df_template"

        # Check partial matches
        header_str = " ".join(headers).lower()
        if "bit" in header_str and "meaning" in header_str:
            return "bit_field"
        if "value" in header_str and ("meaning" in header_str or "description" in header_str):
            return "encoding"

        return "generic"