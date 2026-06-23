"""
src/indexing/parser
--------------------
Public surface of the parser sub-package.
"""

from .pdf_parser import (
    PDFParser,
    ParsedPage,
    TextBlock,
    TextSpan,
    RawTable,
)

__all__ = [
    "PDFParser",
    "ParsedPage",
    "TextBlock",
    "TextSpan",
    "RawTable",
]
