"""
engine package initialization.
"""

from .backend_base import TranslationBackend
from .backend_llm import LLMBackend
from .backend_madlad import MADLADBackend
from .backend_nllb import NLLBBackend
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
from .nllb_manager import (
    check_nllb_installed,
    download_nllb_model,
    get_nllb_model_dir,
    get_nllb_model_info,
    import_local_model_folder,
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
    check_nllb_ready,
    check_nmt_ready,
    check_ollama_status,
    check_ram,
    list_installed_models,
    run_madlad_preflight,
    run_nllb_preflight,
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
    "MADLADBackend",
    "NLLBBackend",
    "NMTBackend",
    "NullCache",
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
    "check_nllb_installed",
    "check_nllb_ready",
    "check_nmt_ready",
    "check_ollama_status",
    "check_ram",
    "compute_file_sha256",
    "contains_japanese",
    "contains_latin",
    "derive_fernet_key",
    "derive_machine_key",
    "download_madlad_model",
    "download_nllb_model",
    "escape_xml",
    "execute_translation",
    "extract_numeric_tokens",
    "find_ollama_binary",
    "get_logger",
    "get_madlad_model_dir",
    "get_madlad_model_info",
    "get_nllb_model_dir",
    "get_nllb_model_info",
    "get_ollama_manager",
    "hash_text",
    "import_local_madlad_folder",
    "import_local_model_folder",
    "list_installed_models",
    "load_trusted_packages",
    "mask_glossary_terms",
    "mask_numbers",
    "run_madlad_preflight",
    "run_nllb_preflight",
    "run_nmt_preflight",
    "run_preflight",
    "should_translate",
    "unescape_xml",
    "unmask_numbers",
    "unmask_protected_text",
    "verify_nmt_numbers",
    "verify_package_archive",
    "verify_placeholders",
]
