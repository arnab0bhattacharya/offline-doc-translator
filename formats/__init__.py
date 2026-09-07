"""
formats package initialization.
"""

from .base import (
    BaseFormatHandler,
    MAX_EXTRACTED_BYTES,
    MAX_ARCHIVE_ENTRIES,
    MAX_SINGLE_ENTRY_BYTES,
)
from .pptx_handler import PPTXHandler
from .xlsx_handler import XLSXHandler
from .docx_handler import DOCXHandler
from .pdf_handler import PDFHandler
from .registry import get_handler, HANDLER_REGISTRY, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseFormatHandler",
    "MAX_EXTRACTED_BYTES",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_SINGLE_ENTRY_BYTES",
    "PPTXHandler",
    "XLSXHandler",
    "DOCXHandler",
    "PDFHandler",
    "get_handler",
    "HANDLER_REGISTRY",
    "SUPPORTED_EXTENSIONS",
]

