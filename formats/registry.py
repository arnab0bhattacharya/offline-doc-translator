"""
formats/registry.py
===================
Central format handler registry. Adding a new format means adding one line here.
"""

from engine.security_policy import DocumentSecurityPolicy

from .base import BaseFormatHandler

HANDLER_REGISTRY: dict[str, type[BaseFormatHandler]] = {}
SUPPORTED_EXTENSIONS: set[str] = {".pptx", ".xlsx", ".docx", ".pdf"}


def _ensure_registry_loaded() -> None:
    if not HANDLER_REGISTRY:
        from .docx_handler import DOCXHandler
        from .pdf_handler import PDFHandler
        from .pptx_handler import PPTXHandler
        from .xlsx_handler import XLSXHandler

        HANDLER_REGISTRY.update(
            {
                ".pptx": PPTXHandler,
                ".xlsx": XLSXHandler,
                ".docx": DOCXHandler,
                ".pdf": PDFHandler,
            }
        )


def get_handler(
    ext: str,
    engine,
    policy: DocumentSecurityPolicy | None = None,
) -> BaseFormatHandler:
    """Returns the handler instance for the given file extension."""
    _ensure_registry_loaded()
    handler_cls = HANDLER_REGISTRY.get(ext.lower())
    if handler_cls is None:
        from engine.errors import ErrorCode, TranslatorError

        raise TranslatorError(
            ErrorCode.E04, detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return handler_cls(engine, policy=policy)
