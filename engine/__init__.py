"""
engine package initialization.
Streamlined for 2-engine enterprise architecture:
- Machine Translation (Google MADLAD-400 3B via CTranslate2 INT8)
- AI Translation (Google Gemma 4 E2B via Ollama)
"""

from .backend_base import TranslationBackend
from .backend_llm import LLMBackend
from .backend_madlad import MADLADBackend
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
    NumericAuditResult,
    TranslationEngine,
    TranslationMode,
    TranslationResult,
    contains_japanese,
    contains_latin,
    escape_xml,
    extract_numeric_tokens,
    hash_text,
    mask_glossary_terms,
    mask_numbers,
    should_translate,
    unescape_xml,
    unmask_numbers,
    unmask_protected_text,
    verify_nmt_numbers,
    verify_placeholders,
)
from .errors import ERROR_MESSAGES, ErrorCode, TranslatorError
from .logging import TranslationLogEvent, TranslationLogger, get_logger
from .madlad_manager import (
    check_madlad_installed,
    download_madlad_model,
    get_madlad_model_dir,
    get_madlad_model_info,
    import_local_madlad_folder,
)
from .ollama_manager import (
    OllamaManager,
    find_ollama_binary,
    get_ollama_manager,
)
from .preflight import (
    check_disk_space,
    check_madlad_ready,
    check_model_installed,
    check_mt_ready,
    check_ollama_status,
    check_ram,
    list_installed_models,
    run_madlad_preflight,
    run_mt_preflight,
    run_preflight,
)
from .system_specs import HardwareSpecs, get_hardware_specs


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
    "HardwareSpecs",
    "JSONFileCache",
    "LLMBackend",
    "MADLADBackend",
    "NullCache",
    "NumericAuditResult",
    "OllamaManager",
    "TranslationBackend",
    "TranslationCache",
    "TranslationEngine",
    "TranslationLogEvent",
    "TranslationLogger",
    "TranslationMode",
    "TranslationResult",
    "TranslatorError",
    "check_disk_space",
    "check_madlad_installed",
    "check_madlad_ready",
    "check_model_installed",
    "check_mt_ready",
    "check_ollama_status",
    "check_ram",
    "contains_japanese",
    "contains_latin",
    "derive_fernet_key",
    "derive_machine_key",
    "download_madlad_model",
    "escape_xml",
    "execute_translation",
    "extract_numeric_tokens",
    "find_ollama_binary",
    "get_hardware_specs",
    "get_logger",
    "get_madlad_model_dir",
    "get_madlad_model_info",
    "get_ollama_manager",
    "hash_text",
    "import_local_madlad_folder",
    "list_installed_models",
    "mask_glossary_terms",
    "mask_numbers",
    "run_madlad_preflight",
    "run_mt_preflight",
    "run_preflight",
    "should_translate",
    "unescape_xml",
    "unmask_numbers",
    "unmask_protected_text",
    "verify_nmt_numbers",
    "verify_placeholders",
]
