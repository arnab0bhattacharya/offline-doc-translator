"""
formats package initialization.
"""

from .base import BaseFormatHandler
from .pptx_handler import PPTXHandler
from .xlsx_handler import XLSXHandler
from .docx_handler import DOCXHandler
from .pdf_handler import PDFHandler
from .registry import get_handler, HANDLER_REGISTRY, SUPPORTED_EXTENSIONS

__all__ = [
    "BaseFormatHandler",
    "PPTXHandler",
    "XLSXHandler",
    "DOCXHandler",
    "PDFHandler",
    "get_handler",
    "HANDLER_REGISTRY",
    "SUPPORTED_EXTENSIONS",
]
