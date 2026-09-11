"""
engine package initialization.
"""

from .backend_base import TranslationBackend
from .backend_llm import LLMBackend
from .backend_nmt import (
    NMTBackend,
    compute_file_sha256,
    load_trusted_packages,
    verify_package_archive,
)
from .cache import (
    EncryptedFileCache,
    JSONFileCache,
    NullCache,
    TranslationCache,
    derive_fernet_key,
    derive_machine_key,
)
from .core import (
    CACHE_TTL_DAYS,
    DIRECTIONS,
    TranslationEngine,
    TranslationMode,
    TranslationResult,
    contains_japanese,
    contains_latin,
    escape_xml,
    hash_text,
    mask_glossary_terms,
    mask_numbers,
    should_translate,
    unescape_xml,
    unmask_numbers,
    unmask_protected_text,
    verify_placeholders,
)
from .errors import ERROR_MESSAGES, ErrorCode, TranslatorError
from .logging import TranslationLogEvent, TranslationLogger, get_logger
from .preflight import (
    check_disk_space,
    check_model_installed,
    check_nmt_ready,
    check_ollama_status,
    check_ram,
    list_installed_models,
    run_nmt_preflight,
    run_preflight,
)


def __getattr__(name: str):
    if name == "execute_translation":
        from .run_job import execute_translation

        return execute_translation
    raise AttributeError(f"module 'engine' has no attribute '{name}'")


__all__ = [
    "CACHE_TTL_DAYS",
    "DIRECTIONS",
    "ERROR_MESSAGES",
    "EncryptedFileCache",
    "ErrorCode",
    "JSONFileCache",
    "LLMBackend",
    "NMTBackend",
    "NullCache",
    "TranslationBackend",
    "TranslationCache",
    "TranslationEngine",
    "TranslationLogEvent",
    "TranslationLogger",
    "TranslationMode",
    "TranslationResult",
    "TranslatorError",
    "check_disk_space",
    "check_model_installed",
    "check_nmt_ready",
    "check_ollama_status",
    "check_ram",
    "compute_file_sha256",
    "contains_japanese",
    "contains_latin",
    "derive_fernet_key",
    "derive_machine_key",
    "escape_xml",
    "execute_translation",
    "get_logger",
    "hash_text",
    "list_installed_models",
    "load_trusted_packages",
    "mask_glossary_terms",
    "mask_numbers",
    "run_nmt_preflight",
    "run_preflight",
    "should_translate",
    "unescape_xml",
    "unmask_numbers",
    "unmask_protected_text",
    "verify_package_archive",
    "verify_placeholders",
]
