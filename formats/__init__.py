"""
formats package initialization.
"""

from .base import (
    MAX_ARCHIVE_ENTRIES,
    MAX_EXTRACTED_BYTES,
    MAX_SINGLE_ENTRY_BYTES,
    BaseFormatHandler,
)
from .docx_handler import DOCXHandler
from .pdf_handler import PDFHandler
from .pptx_handler import PPTXHandler
from .registry import HANDLER_REGISTRY, SUPPORTED_EXTENSIONS, _ensure_registry_loaded, get_handler
from .xlsx_handler import XLSXHandler

_ensure_registry_loaded()

__all__ = [
    "HANDLER_REGISTRY",
    "MAX_ARCHIVE_ENTRIES",
    "MAX_EXTRACTED_BYTES",
    "MAX_SINGLE_ENTRY_BYTES",
    "SUPPORTED_EXTENSIONS",
    "BaseFormatHandler",
    "DOCXHandler",
    "PDFHandler",
    "PPTXHandler",
    "XLSXHandler",
    "get_handler",
]
