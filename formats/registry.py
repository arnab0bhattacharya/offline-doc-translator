"""
formats/registry.py
===================
Central format handler registry. Adding a new format means adding one line here.
"""
from typing import Dict, Type, Set
from .base import BaseFormatHandler
from .pptx_handler import PPTXHandler
from .xlsx_handler import XLSXHandler
from .docx_handler import DOCXHandler
from .pdf_handler import PDFHandler

HANDLER_REGISTRY: Dict[str, Type[BaseFormatHandler]] = {
    ".pptx": PPTXHandler,
    ".xlsx": XLSXHandler,
    ".docx": DOCXHandler,
    ".pdf": PDFHandler,
}

SUPPORTED_EXTENSIONS: Set[str] = set(HANDLER_REGISTRY.keys())


def get_handler(ext: str, engine) -> BaseFormatHandler:
    """Returns the handler instance for the given file extension."""
    handler_cls = HANDLER_REGISTRY.get(ext.lower())
    if handler_cls is None:
        from engine.errors import ErrorCode, TranslatorError
        raise TranslatorError(
            ErrorCode.E04,
            detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return handler_cls(engine)
