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
from .backend_nmt import (
    NMTBackend,
    load_trusted_packages,
    verify_package_archive,
    compute_file_sha256,
)
from .backend_llm import LLMBackend
from .backend_base import TranslationBackend
from .cache import (
    TranslationCache,
    JSONFileCache,
    NullCache,
    EncryptedFileCache,
    derive_machine_key,
    derive_fernet_key,
)
from .logging import TranslationLogger, TranslationLogEvent, get_logger
from .core import (
    DIRECTIONS,
    CACHE_TTL_DAYS,
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
    TranslationResult,
)


def __getattr__(name: str):
    if name == "execute_translation":
        from .run_job import execute_translation
        return execute_translation
    raise AttributeError(f"module 'engine' has no attribute '{name}'")

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
    "load_trusted_packages",
    "verify_package_archive",
    "compute_file_sha256",
    "LLMBackend",
    "TranslationBackend",
    "TranslationCache",
    "JSONFileCache",
    "NullCache",
    "EncryptedFileCache",
    "derive_machine_key",
    "derive_fernet_key",
    "TranslationLogger",
    "TranslationLogEvent",
    "get_logger",
    "DIRECTIONS",
    "CACHE_TTL_DAYS",
    "TranslationMode",
    "TranslationResult",
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

