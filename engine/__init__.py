"""
engine package initialization.
"""

from .errors import ErrorCode, TranslatorError, ERROR_MESSAGES
from .preflight import (
    check_ollama_status,
    list_installed_models,
    check_model_installed,
    check_ram,
    check_disk_space,
    check_nmt_ready,
    run_nmt_preflight,
    run_preflight,
)
from .backend_nmt import NMTBackend
from .backend_llm import LLMBackend
from .core import (
    DIRECTIONS,
    TranslationMode,
    contains_japanese,
    contains_latin,
    should_translate,
    mask_numbers,
    verify_placeholders,
    unmask_numbers,
    unmask_protected_text,
    escape_xml,
    unescape_xml,
    mask_glossary_terms,
    hash_text,
    TranslationEngine,
)
from .run_job import execute_translation

__all__ = [
    "ErrorCode",
    "TranslatorError",
    "ERROR_MESSAGES",
    "check_ollama_status",
    "list_installed_models",
    "check_model_installed",
    "check_ram",
    "check_disk_space",
    "check_nmt_ready",
    "run_nmt_preflight",
    "run_preflight",
    "NMTBackend",
    "LLMBackend",
    "DIRECTIONS",
    "TranslationMode",
    "contains_japanese",
    "contains_latin",
    "should_translate",
    "mask_numbers",
    "verify_placeholders",
    "unmask_numbers",
    "unmask_protected_text",
    "escape_xml",
    "unescape_xml",
    "mask_glossary_terms",
    "hash_text",
    "TranslationEngine",
    "execute_translation",
]
