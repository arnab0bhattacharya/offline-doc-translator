"""
engine/core.py
==============
Core format-agnostic translation engine.
Orchestrates two translation modes:
  - Fast NMT Mode (ArgosTranslate / CTranslate2) for high-speed offline translation
  - Pure LLM Mode (Ollama) with isomorphic retry & custom glossary enforcement
  - Unified atomic caching, single-pass number masking, and bidirectional gates.
"""

import hashlib
import html
import json
import logging
import re
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import psutil
import requests

from .backend_base import TranslationBackend
from .cache import CACHE_TTL_DAYS, CachePolicy, EncryptedFileCache, JSONFileCache, NullCache, TranslationCache
from .errors import ErrorCode, TranslatorError
from .logging import TranslationLogger, get_logger

# Supported translation directions
DIRECTIONS = ("ja2en", "en2ja")
PLACEHOLDER_PATTERN = re.compile(r"\[\[[A-Z][A-Z_0-9]*\]\]")


@dataclass
class TranslationResult:
    """Encapsulates the result and telemetry of translating a text chunk."""

    text: str
    was_translated: bool
    was_reverted: bool
    elapsed: float = 0.0
    source_backend: str = ""  # "nmt", "llm", "cache"

    def __iter__(self):
        """Supports backward-compatible unpacking: (text, was_translated, was_reverted) = result."""
        return iter((self.text, self.was_translated, self.was_reverted))

    def __getitem__(self, index):
        """Supports index-based tuple access for backward compatibility."""
        return (self.text, self.was_translated, self.was_reverted)[index]


class TranslationMode(str, Enum):
    FAST_NMT = "fast_nmt"  # 100% CTranslate2 / Argos (blazing fast, 0.02s/chunk)
    NLLB_3B = "nllb_3b"  # 100% CTranslate2 / NLLB-200 3.3B INT8 (~3.4 GB)
    MADLAD_3B = "madlad_3b"  # 100% CTranslate2 / Google MADLAD-400 3B INT8 (~3.0 GB)
    PURE_LLM = "pure_llm"  # 100% Local LLM via Ollama
    QUALITY_NMT = "quality_nmt"  # Backward-compatible alias (routes to NLLB 3.3B)


def contains_japanese(text: str) -> bool:
    """Detects Hiragana, Katakana, and CJK unified ideographs."""
    return bool(re.search(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]", text))


def contains_latin(text: str) -> bool:
    """Detects 2+ consecutive alphabetic characters on a word boundary."""
    return bool(re.search(r"\b[a-zA-Z]{2,}\b", text))


def should_translate(text: str, direction: str) -> bool:
    """
    Symmetric translation gate:
      ja2en: translate only if Japanese script is present.
      en2ja: translate only if real Latin prose is present AND Japanese is NOT present.
    """
    if not text or not text.strip():
        return False
    if direction == "ja2en":
        return contains_japanese(text)
    elif direction == "en2ja":
        return contains_latin(text) and not contains_japanese(text)
    raise ValueError(f"Unknown translation direction: {direction}")


def mask_numbers(text: str) -> tuple[str, dict[str, str]]:
    """
    Masks numbers, floats, percentages, and basic formulas into placeholders [[N0]], [[N1]].
    Uses a single-pass regex replacement to guarantee placeholders are never recursively nested.
    Identical numbers in the same string share the same placeholder to preserve structure.
    """
    pattern = re.compile(r"(\d+(?:\.\d+)?(?:[,\s]*[+\-xX*/%][\s]*\d+(?:\.\d+)?)*%?)")
    number_map: dict[str, str] = {}
    val_to_placeholder: dict[str, str] = {}
    placeholder_counter = 0

    def repl(match):
        nonlocal placeholder_counter
        val = match.group(1)
        if not val:
            return match.group(0)
        if val in val_to_placeholder:
            return val_to_placeholder[val]
        placeholder = f"[[N{placeholder_counter}]]"
        placeholder_counter += 1
        number_map[placeholder] = val
        val_to_placeholder[val] = placeholder
        return placeholder

    masked_text = pattern.sub(repl, text)
    return masked_text, number_map


def _alphabetic_id(index: int) -> str:
    """Returns A, B, ..., Z, AA, AB, ... without digits that number masking could alter."""
    result = ""
    while True:
        index, remainder = divmod(index, 26)
        result = chr(ord("A") + remainder) + result
        if index == 0:
            return result
        index -= 1


def mask_glossary_terms(text: str, glossary: dict[str, str]) -> tuple[str, dict[str, str]]:
    """Replaces source glossary terms with stable placeholders before translation."""
    terms = [term for term in glossary if term]
    if not terms:
        return text, {}

    # Longest-first matching prevents a shorter glossary term from consuming a longer one.
    pattern = re.compile("|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)))
    term_placeholders: dict[str, str] = {}
    glossary_map: dict[str, str] = {}

    def repl(match):
        term = match.group(0)
        if term not in term_placeholders:
            placeholder = f"[[GLOSSARY_{_alphabetic_id(len(term_placeholders))}]]"
            term_placeholders[term] = placeholder
            glossary_map[placeholder] = glossary[term]
        return term_placeholders[term]

    return pattern.sub(repl, text), glossary_map


def verify_placeholders(
    translated_text: str, placeholder_map: dict[str, str], masked_source: str | None = None
) -> bool:
    """Checks that the exact placeholder multiset survived translation uncorrupted."""
    expected_text = masked_source if masked_source is not None else " ".join(placeholder_map.keys())
    expected = Counter(PLACEHOLDER_PATTERN.findall(expected_text))
    actual = Counter(PLACEHOLDER_PATTERN.findall(translated_text))
    return actual == expected


_MONTH_NAMES_EN = {
    "1": "january",
    "2": "february",
    "3": "march",
    "4": "april",
    "5": "may",
    "6": "june",
    "7": "july",
    "8": "august",
    "9": "september",
    "10": "october",
    "11": "november",
    "12": "december",
}


def extract_numeric_tokens(text: str) -> list[str]:
    """Extracts numeric values without commas or percentages for consistency checks."""
    matches = re.findall(r"(?<![A-Za-z0-9])\d+(?:,\d{3})*(?:\.\d+)?%?(?![A-Za-z0-9])", text)
    cleaned = []
    for m in matches:
        val = m.replace(",", "").rstrip("%").strip()
        if val:
            cleaned.append(val)
    return cleaned


def verify_nmt_numbers(source_text: str, target_text: str) -> tuple[bool, list[str]]:
    """
    Checks that numbers present in the source text appear in the target text.
    Accounts for localized month names (e.g. '10月' -> 'October').
    Returns (all_present: bool, missing_numbers: list[str]).
    """
    src_nums = extract_numeric_tokens(source_text)
    if not src_nums:
        return True, []

    target_lower = target_text.lower()
    tgt_nums = extract_numeric_tokens(target_text)
    tgt_counter = Counter(tgt_nums)

    missing = []
    for num in src_nums:
        if tgt_counter.get(num, 0) > 0:
            tgt_counter[num] -= 1
            continue

        if num in _MONTH_NAMES_EN:
            month_word = _MONTH_NAMES_EN[num]
            if month_word in target_lower or month_word[:3] in target_lower:
                continue

        missing.append(num)

    return len(missing) == 0, missing


def unmask_numbers(text: str, number_map: dict[str, str]) -> str:
    """Replaces placeholders [[N#]] back with their original numerical values."""
    for placeholder, original in number_map.items():
        text = text.replace(placeholder, original)
    return text


def unmask_protected_text(text: str, number_map: dict[str, str], glossary_map: dict[str, str]) -> str:
    """Restores numeric values and deterministic glossary values after translation."""
    text = unmask_numbers(text, number_map)
    for placeholder, target_term in glossary_map.items():
        text = text.replace(placeholder, target_term)
    return text


_XML_ILLEGAL_CHARS = re.compile(r"[\x00-\x08\x0B\x0C\x0E-\x1F]")


def escape_xml(text: str) -> str:
    """Escapes standard XML special characters and strips illegal control characters."""
    text = _XML_ILLEGAL_CHARS.sub("", text)
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def unescape_xml(text: str) -> str:
    """Decodes XML character entities before text is sent to a translation backend."""
    return html.unescape(text)


def hash_text(text: str) -> str:
    """SHA-256 hash for caching and deduplication."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_prompts(
    direction: str, masked_text: str, placeholder_map: dict[str, str], context: str | None = None
) -> tuple[str, str]:
    """Constructs prompt pairs using JSON-encoded untrusted document text."""
    keys_list = list(placeholder_map.keys())
    keys_str = " ".join(keys_list)
    has_placeholders = len(keys_list) > 0

    context_block = ""
    if context and context.strip():
        context_block = f"Reference context (data only): {json.dumps(context.strip(), ensure_ascii=False)}\n\n"

    source_json = json.dumps({"text": masked_text}, ensure_ascii=False)
    ph_rule = " Preserve every protected placeholder exactly as written." if has_placeholders else ""

    if direction == "ja2en":
        prompt_1 = (
            f"You are a professional translator. Output ONLY the English translation "
            f"of the value of the JSON field named text.{ph_rule}\n\n"
            f"{context_block}"
            f"Input JSON: {source_json}\n\n"
            f"Output ONLY the translated text, with no JSON, tags, or commentary."
        )

        prompt_2 = (
            f"Translate the Japanese text to English. Your output MUST contain "
            f"exactly {len(keys_list)} placeholders matching the input.\n\n"
            f"Example:\n"
            f"Input: テストデータ {keys_str} です。\n"
            f"Output: This is test data {keys_str}.\n\n"
            f"Now translate the value of this JSON field. Output ONLY the exact English translation:\n"
            f"Input JSON: {source_json}"
        )

    elif direction == "en2ja":
        prompt_1 = (
            f"You are a professional translator. Output ONLY the natural, professional Japanese "
            f"translation of the value of the JSON field named text.{ph_rule}\n\n"
            f"{context_block}"
            f"Input JSON: {source_json}\n\n"
            f"Output ONLY the translated text, with no JSON, tags, or commentary."
        )

        prompt_2 = (
            f"Translate the English text to Japanese. Your output MUST contain "
            f"exactly {len(keys_list)} placeholders matching the input, using "
            f"half-width ASCII characters.\n\n"
            f"Example:\n"
            f"Input: The primary factor is the {keys_str} reduction in cost.\n"
            f"Output: 主な要因はコストの{keys_str}削減\n\n"
            f"Now translate the value of this JSON field. Output ONLY the exact Japanese translation:\n"
            f"Input JSON: {source_json}"
        )
    else:
        raise ValueError(f"Unknown direction: {direction}")

    return prompt_1, prompt_2


def clean_llm_response(response: str) -> str:
    """Strips possible hallucinated wrapper tags (like <target>...</target> or markdown fences)."""
    text = response.strip()
    if text.startswith("```") and text.endswith("```"):
        lines = text.splitlines()
        if len(lines) >= 3:
            text = "\n".join(lines[1:-1]).strip()
    target_match = re.search(r"<target>(.*?)</target>", text, re.DOTALL | re.IGNORECASE)
    if target_match:
        text = target_match.group(1).strip()
    return text


class TranslationEngine:
    """
    Unified Translation Engine orchestrating the Hybrid Architecture.
    """

    def __init__(
        self,
        model_name: str = "gemma4:e2b-it-qat",
        ollama_url: str = "http://localhost:11434",
        mode: TranslationMode = TranslationMode.FAST_NMT,
        glossary: dict[str, str] | None = None,
        min_free_ram_mb: int = 150,
        context_window: int = 2048,
        cache_file: str | None = None,
        allow_llm: bool | None = None,
        include_source_text: bool = False,
        cache_ttl_days: int = CACHE_TTL_DAYS,
        backend: TranslationBackend | None = None,
        cache: TranslationCache | None = None,
        logger: TranslationLogger | None = None,
        cache_policy: CachePolicy | str = CachePolicy.ENCRYPTED_PERSISTENT,
        encrypted_cache: bool | None = None,
        cache_key: str | bytes | None = None,
        legacy_cache_candidates: list[str] | None = None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.mode = mode
        self.glossary = glossary or {}
        self.min_free_ram_mb = min_free_ram_mb
        self.context_window = context_window
        self.allow_llm = allow_llm
        self.include_source_text = include_source_text
        self.cache_ttl_days = cache_ttl_days
        self.logger = logger if logger is not None else get_logger()

        self.failed_this_run: set = set()
        self.logged_failed_keys: set = set()
        self._glossary_translation_cache: dict[tuple[str, str], str] = {}

        # Resolve CachePolicy (handling legacy encrypted_cache boolean if passed)
        if encrypted_cache is not None:
            effective_policy = CachePolicy.ENCRYPTED_PERSISTENT if encrypted_cache else CachePolicy.PLAINTEXT_PERSISTENT
        elif isinstance(cache_policy, CachePolicy):
            effective_policy = cache_policy
        else:
            try:
                effective_policy = CachePolicy(cache_policy)
            except (ValueError, TypeError):
                effective_policy = CachePolicy.ENCRYPTED_PERSISTENT
        self.cache_policy = effective_policy

        if cache_file is None:
            if self.cache_policy == CachePolicy.PLAINTEXT_PERSISTENT:
                self.cache_file = "translation_cache.json"
            else:
                self.cache_file = "translation_cache.enc"
        else:
            self.cache_file = cache_file

        if cache is not None:
            self._cache_mgr = cache
        elif self.cache_policy == CachePolicy.MEMORY_ONLY:
            self._cache_mgr = NullCache()
        elif self.cache_policy == CachePolicy.PLAINTEXT_PERSISTENT:
            self._cache_mgr = JSONFileCache(
                cache_file=self.cache_file,
                ttl_days=cache_ttl_days,
                legacy_candidates=legacy_cache_candidates,
            )
        else:  # CachePolicy.ENCRYPTED_PERSISTENT
            try:
                self._cache_mgr = EncryptedFileCache(
                    cache_file=self.cache_file,
                    key=cache_key,
                    ttl_days=cache_ttl_days,
                    fallback_to_plain=False,
                    legacy_candidates=legacy_cache_candidates,
                )
            except Exception as err:
                warning_msg = (
                    f"[!] Warning: Cache encryption unavailable ({err}). "
                    "Disabling cache persistence (operating in memory-only mode for privacy)."
                )
                if self.logger:
                    self.logger.warning(warning_msg, category="cache")
                logging.warning(warning_msg)
                self._cache_mgr = NullCache()

        # Lazy-loaded backend instances
        self._nmt_backend = None
        self._nllb_backend = None
        self._madlad_backend = None
        self._llm_backend = None
        self._custom_backend = backend

    @property
    def cache_mgr(self) -> TranslationCache:
        """Returns the underlying TranslationCache instance."""
        return self._cache_mgr

    @property
    def cache(self) -> dict[str, Any]:
        """Provides access to raw cache dict for backward compatibility."""
        if hasattr(self._cache_mgr, "data"):
            return self._cache_mgr.data
        return {}

    @cache.setter
    def cache(self, value: dict[str, Any]) -> None:
        if hasattr(self._cache_mgr, "data"):
            self._cache_mgr.data = value

    @property
    def nmt_backend(self):
        if self._nmt_backend is None:
            from engine.backend_nmt import NMTBackend

            self._nmt_backend = NMTBackend()
        return self._nmt_backend

    @property
    def nllb_backend(self):
        if self._nllb_backend is None:
            from engine.backend_nllb import NLLBBackend

            self._nllb_backend = NLLBBackend()
        return self._nllb_backend

    @property
    def madlad_backend(self):
        if self._madlad_backend is None:
            from engine.backend_madlad import MADLADBackend

            self._madlad_backend = MADLADBackend()
        return self._madlad_backend

    @property
    def llm_backend(self):
        if self._llm_backend is None:
            from engine.backend_llm import LLMBackend

            self._llm_backend = LLMBackend(
                model_name=self.model_name, ollama_url=self.ollama_url, context_window=self.context_window
            )
        return self._llm_backend

    def get_backend(self, mode: TranslationMode | None = None) -> TranslationBackend:
        """Returns the active or requested translation backend instance."""
        if mode is None and self._custom_backend is not None:
            return self._custom_backend
        effective_mode = mode if mode is not None else self.mode
        if effective_mode == TranslationMode.FAST_NMT:
            return self.nmt_backend
        if effective_mode in (TranslationMode.NLLB_3B, TranslationMode.QUALITY_NMT):
            if self._madlad_backend is not None and getattr(self._madlad_backend, "is_model_loaded", lambda: False)():
                self._madlad_backend.unload()
            return self.nllb_backend
        if effective_mode == TranslationMode.MADLAD_3B:
            if self._nllb_backend is not None and getattr(self._nllb_backend, "is_model_loaded", lambda: False)():
                self._nllb_backend.unload()
            return self.madlad_backend
        return self.llm_backend

    def set_backend(self, backend: TranslationBackend, mode: TranslationMode | None = None) -> None:
        """Registers a custom or replacement translation backend."""
        if mode is None:
            self._custom_backend = backend
        elif mode == TranslationMode.FAST_NMT:
            self._nmt_backend = backend
        elif mode in (TranslationMode.NLLB_3B, TranslationMode.QUALITY_NMT):
            self._nllb_backend = backend
        elif mode == TranslationMode.MADLAD_3B:
            self._madlad_backend = backend
        else:
            self._llm_backend = backend

    def unload_backends(self) -> None:
        """Unloads in-memory backends (such as NLLB and MADLAD) and frees model RAM."""
        if self._nllb_backend is not None and hasattr(self._nllb_backend, "unload"):
            self._nllb_backend.unload()
        if self._madlad_backend is not None and hasattr(self._madlad_backend, "unload"):
            self._madlad_backend.unload()
        if self._custom_backend is not None and hasattr(self._custom_backend, "unload"):
            self._custom_backend.unload()

    def load_cache(self, direction: str) -> None:
        """Loads cache namespaced by mode -> direction -> configuration fingerprint -> hash."""
        self._cache_mgr.load(direction)
        self._get_direction_cache(direction)

    def _record_cache_access(
        self, direction: str, context: str | None, key: str, timestamp: float | None = None
    ) -> None:
        """Records last accessed timestamp for a cache entry."""
        mode_str = self.mode.value if isinstance(self.mode, TranslationMode) else str(self.mode)
        fp = self._cache_fingerprint(context)
        self._cache_mgr.record_access(key=key, direction=direction, fingerprint=fp, mode=mode_str, timestamp=timestamp)

    def _prune_cache(self, now: float | None = None) -> int:
        """Removes cache entries older than cache_ttl_days. Returns number of pruned entries."""
        return self._cache_mgr.prune(now)

    def clear_cache(self, log_cb: Callable[[str], None] | None = None) -> None:
        """Wipes the entire cache and saves an empty cache file."""
        self._cache_mgr.clear(log_cb=log_cb)

    def _cache_fingerprint(self, context: str | None = None) -> str:
        """Prevents reuse across models, glossaries, and context-sensitive translations."""
        payload = {
            "schema": 4,
            "model": self.model_name,
            "glossary": sorted(self.glossary.items()),
            "context": context or "",
            "llm_enabled": self.allow_llm,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def _get_direction_cache(self, direction: str, context: str | None = None) -> dict[str, str]:
        """Returns the cache bucket for one fully specified translation configuration."""
        mode_str = self.mode.value if isinstance(self.mode, TranslationMode) else str(self.mode)
        fp = self._cache_fingerprint(context)
        if hasattr(self._cache_mgr, "get_bucket"):
            return self._cache_mgr.get_bucket(direction=direction, fingerprint=fp, mode=mode_str)
        return {}

    def save_cache_atomically(self, log_cb: Callable[[str], None] | None = None) -> None:
        """Saves cache via .tmp file replacement to prevent corruption on crash."""
        self._cache_mgr.save(log_cb=log_cb)

    def check_and_clear_memory(self, log_cb: Callable[[str], None] | None = None) -> None:
        """Flushes Ollama KV/model if available system RAM falls below threshold."""
        try:
            free_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
            if free_ram_mb < self.min_free_ram_mb:
                if log_cb:
                    log_cb(f"[!] Low RAM ({free_ram_mb:.0f} MB). Flushing memory...")
                requests.post(
                    f"{self.ollama_url}/api/generate", json={"model": self.model_name, "keep_alive": 0}, timeout=5
                )
                time.sleep(3)
        except Exception:
            pass

    def flush_model(self) -> None:
        """Explicitly unloads model from VRAM/RAM after a document completes."""
        if self.mode != TranslationMode.PURE_LLM:
            return
        try:
            requests.post(
                f"{self.ollama_url}/api/generate", json={"model": self.model_name, "keep_alive": 0}, timeout=5
            )
        except Exception:
            pass

    def log_needs_review(
        self,
        review_log_path: str,
        location_id: str,
        chunk_id: str,
        original_text: str,
        key: str,
        include_source_text: bool | None = None,
    ) -> None:
        """Logs translation failures to audit log. Source text omitted by default for privacy."""
        if not review_log_path:
            return

        if key in self.logged_failed_keys:
            try:
                with open(review_log_path, "a", encoding="utf-8") as f:
                    f.write(f"  [Additional occurrence in {location_id}]\n")
            except Exception:
                pass
            return

        self.logged_failed_keys.add(key)
        text_length = len(original_text)
        text_hash = hashlib.sha256(original_text.encode("utf-8")).hexdigest()[:16]

        should_include = (
            include_source_text if include_source_text is not None else getattr(self, "include_source_text", False)
        )

        try:
            with open(review_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Location: {location_id} | Chunk ID: {chunk_id}\n"
                    f"Text Length: {text_length} chars | SHA-256: {text_hash}\n"
                )
                if should_include:
                    f.write(f"Original Text: {original_text}\n")
                f.write(f"{'-' * 50}\n")
        except Exception:
            pass

    def _get_term_translation(self, term: str, direction: str, backend: TranslationBackend) -> str:
        """Retrieves or queries the default isolated translation of a glossary term."""
        cache_key = (term, direction)
        if cache_key in self._glossary_translation_cache:
            return self._glossary_translation_cache[cache_key]

        val = ""
        try:
            if hasattr(backend, "translate_single"):
                res = backend.translate_single(term, direction)
                val = (res or "").strip()
            elif hasattr(backend, "translate"):
                res_tuple = backend.translate(text=term, direction=direction)
                res = res_tuple[0] if isinstance(res_tuple, tuple) else res_tuple
                val = (res or "").strip()
        except Exception:
            val = ""

        self._glossary_translation_cache[cache_key] = val
        return val

    def _apply_post_translation_glossary(
        self,
        translated_text: str,
        source_text: str,
        direction: str,
        backend: TranslationBackend,
    ) -> str:
        """
        Enforces user glossary rules on clean NMT output using dynamic pivot alignment.
        Replaces the default machine-translated term with the approved glossary term.
        """
        if not self.glossary:
            return translated_text

        matching_terms = [(src, tgt) for src, tgt in self.glossary.items() if src and tgt and src in source_text]
        matching_terms.sort(key=lambda pair: len(pair[0]), reverse=True)

        for src_term, tgt_term in matching_terms:
            if direction == "ja2en":
                if re.search(rf"\b{re.escape(tgt_term)}\b", translated_text, re.IGNORECASE):
                    continue
            else:
                if tgt_term in translated_text:
                    continue

            default_trans = self._get_term_translation(src_term, direction, backend)
            if not default_trans or default_trans.lower() == tgt_term.lower():
                continue

            if direction == "ja2en":
                pattern = re.compile(rf"\b{re.escape(default_trans)}(?:s|es)?\b", re.IGNORECASE)
                translated_text = pattern.sub(tgt_term, translated_text)
            else:
                translated_text = translated_text.replace(default_trans, tgt_term)

        return translated_text

    def _translate_chunk_nmt(
        self,
        text: str,
        direction: str,
        context: str | None = None,
        location_id: str = "doc",
        chunk_id: str = "0",
        review_log_path: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        """
        Fast NMT translation pipeline operating strictly on clean, natural text.
        Bypasses synthetic placeholder brackets to prevent subword tokenizer fragmentation.
        Enforces post-translation glossary substitution and numeric audit verification.
        """
        key = hash_text(text)
        mode_str = self.mode.value if isinstance(self.mode, TranslationMode) else str(self.mode)
        fp = self._cache_fingerprint(context)
        dir_cache = self._get_direction_cache(direction, context)

        # 1. Check persistent cache
        cached_trans = self._cache_mgr.get(key=key, direction=direction, fingerprint=fp, mode=mode_str)
        if cached_trans is None and key in dir_cache:
            cached_trans = dir_cache[key]

        if cached_trans is not None:
            self._record_cache_access(direction, context, key)
            preview_src = (text[:24] + "..") if len(text) > 26 else text
            preview_res = (cached_trans[:24] + "..") if len(cached_trans) > 26 else cached_trans
            if self.logger:
                self.logger.info(
                    message=f'"{preview_src}" => "{preview_res}"',
                    category="cache",
                    location=location_id,
                    elapsed=0.0,
                )
            if log_cb:
                log_cb(f'  [⚡ Cache] {location_id}: "{preview_src}" => "{preview_res}"')
            return TranslationResult(
                text=cached_trans,
                was_translated=True,
                was_reverted=False,
                elapsed=0.0,
                source_backend="cache",
            )

        # 2. Check runtime failure tracker
        if key in self.failed_this_run:
            if self.logger:
                self.logger.warning(
                    message=f"Skipped {location_id} (previously failed this run)",
                    category="translation",
                    location=location_id,
                )
            if log_cb:
                log_cb(f"  [⏩ Skipped] {location_id}")
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(text=text, was_translated=False, was_reverted=True)

        preview_src = (text[:30] + "..") if len(text) > 32 else text
        backend = self.get_backend()
        backend_name = getattr(backend, "name", "nmt")

        # 3. Check backend readiness
        if not backend.is_ready(direction):
            if self.mode == TranslationMode.MADLAD_3B:
                engine_label = "MADLAD-400 3B"
            elif self.mode in (TranslationMode.NLLB_3B, TranslationMode.QUALITY_NMT):
                engine_label = "NLLB-200 3.3B"
            else:
                engine_label = "Fast NMT (Argos)"
            raise TranslatorError(
                ErrorCode.E08,
                detail=f"{engine_label} cannot translate {direction}: the language model package is not installed.",
            )

        # 4. Dispatch clean translation via backend
        elapsed = 0.0
        try:
            if hasattr(backend, "translate"):
                translated_raw, elapsed = backend.translate(
                    text=text,
                    direction=direction,
                    placeholder_map=None,
                    context=None,
                    log_cb=log_cb,
                )
            else:
                t0 = time.time()
                translated_raw = backend.translate_single(text, direction)
                elapsed = time.time() - t0
        except Exception as e:
            err_str = str(e).lower()
            if "mkl_malloc" in err_str or "out of memory" in err_str or isinstance(e, MemoryError):
                raise TranslatorError(
                    ErrorCode.E03,
                    detail=(
                        f"Out of memory during {backend_name.upper()} translation: {e}. "
                        "The system ran out of RAM for this 3B model. Please close other applications "
                        "or switch to ⚡ Fast (Argos) mode."
                    ),
                ) from e
            if self.logger:
                self.logger.error(
                    message=f"{backend_name.upper()} translation error for {location_id}: {e}. Original kept.",
                    category="translation",
                    location=location_id,
                    details={"error": str(e)},
                )
            if log_cb:
                log_cb(f"  [-] {backend_name.upper()} translation error for {location_id}: {e}. Original kept.")
            self.failed_this_run.add(key)
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(
                text=text,
                was_translated=False,
                was_reverted=True,
                source_backend=backend_name,
            )

        if not translated_raw:
            self.failed_this_run.add(key)
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(
                text=text,
                was_translated=False,
                was_reverted=True,
                source_backend=backend_name,
            )

        # 5. Post-translation numeric audit
        nums_ok, missing_nums = verify_nmt_numbers(text, translated_raw)
        if not nums_ok:
            warn_msg = f"NMT numeric check: missing {missing_nums} in '{location_id}'"
            if self.logger:
                self.logger.warning(message=warn_msg, category="translation", location=location_id)
            if log_cb:
                log_cb(f"  [⚠ Number Audit] {warn_msg}")

        # 6. Post-translation glossary substitution
        final_trans = self._apply_post_translation_glossary(translated_raw, text, direction, backend)

        # 7. Store in cache & return
        self._cache_mgr.put(key=key, direction=direction, fingerprint=fp, value=final_trans, mode=mode_str)
        dir_cache[key] = final_trans
        self._record_cache_access(direction, context, key)

        preview_res = (final_trans[:30] + "..") if len(final_trans) > 32 else final_trans
        if self.logger:
            self.logger.info(
                message=f'"{preview_src}" => "{preview_res}"',
                category="translation",
                location=location_id,
                elapsed=elapsed,
                details={"source_backend": backend_name},
            )
        if log_cb:
            if self.mode == TranslationMode.FAST_NMT:
                tag = "⚡ Fast NMT"
            elif self.mode == TranslationMode.MADLAD_3B:
                tag = "🎯 MADLAD 3B"
            else:
                tag = "🌐 NLLB 3.3B"
            log_cb(f'  [{tag} in {elapsed:.2f}s] {location_id}: "{preview_src}" => "{preview_res}"')

        return TranslationResult(
            text=final_trans,
            was_translated=True,
            was_reverted=False,
            elapsed=elapsed,
            source_backend=backend_name,
        )

    def _translate_chunk_llm(
        self,
        text: str,
        direction: str,
        context: str | None = None,
        location_id: str = "doc",
        chunk_id: str = "0",
        review_log_path: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        """
        Pure LLM translation pipeline with placeholder masking and strict verification.
        """
        glossary_masked_text, glossary_map = mask_glossary_terms(text, self.glossary)
        masked_text, number_map = mask_numbers(glossary_masked_text)
        placeholder_map = {**number_map, **glossary_map}
        key = hash_text(masked_text)

        mode_str = self.mode.value if isinstance(self.mode, TranslationMode) else str(self.mode)
        fp = self._cache_fingerprint(context)
        dir_cache = self._get_direction_cache(direction, context)

        # 1. Check persistent cache
        cached_trans = self._cache_mgr.get(key=key, direction=direction, fingerprint=fp, mode=mode_str)
        if cached_trans is None and key in dir_cache:
            cached_trans = dir_cache[key]

        if cached_trans is not None:
            if verify_placeholders(cached_trans, placeholder_map, masked_text):
                self._record_cache_access(direction, context, key)
                final_cached = unmask_protected_text(cached_trans, number_map, glossary_map)
                preview_src = (text[:24] + "..") if len(text) > 26 else text
                preview_res = (final_cached[:24] + "..") if len(final_cached) > 26 else final_cached
                if self.logger:
                    self.logger.info(
                        message=f'"{preview_src}" => "{preview_res}"',
                        category="cache",
                        location=location_id,
                        elapsed=0.0,
                    )
                if log_cb:
                    log_cb(f'  [⚡ Cache] {location_id}: "{preview_src}" => "{preview_res}"')
                return TranslationResult(
                    text=final_cached,
                    was_translated=True,
                    was_reverted=False,
                    elapsed=0.0,
                    source_backend="cache",
                )
            self._cache_mgr.delete(key=key, direction=direction, fingerprint=fp, mode=mode_str)
            if key in dir_cache:
                del dir_cache[key]

        # 2. Check runtime failure tracker
        if key in self.failed_this_run:
            if self.logger:
                self.logger.warning(
                    message=f"Skipped {location_id} (previously failed this run)",
                    category="translation",
                    location=location_id,
                )
            if log_cb:
                log_cb(f"  [⏩ Skipped] {location_id}")
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(text=text, was_translated=False, was_reverted=True)

        preview_src = (text[:30] + "..") if len(text) > 32 else text
        backend = self.get_backend()
        backend_name = getattr(backend, "name", "llm")

        # 3. Resource gating
        if self.allow_llm is not False:
            self.check_and_clear_memory(log_cb)

        if log_cb:
            path_label = "🤖 Pure LLM"
            log_cb(f'  [{path_label}] {location_id}: "{preview_src}"...')

        if self.allow_llm is False:
            if self.logger:
                self.logger.error(
                    message=f"No LLM backend is available for {location_id}.",
                    category="backend",
                    location=location_id,
                )
            if log_cb:
                log_cb(f"  [❌ Fallback] No LLM backend is available for {location_id}.")
            self.failed_this_run.add(key)
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(
                text=text,
                was_translated=False,
                was_reverted=True,
                source_backend=backend_name,
            )

        # 4. Dispatch translation via backend
        elapsed = 0.0
        try:
            if hasattr(backend, "translate"):
                translated_masked, elapsed = backend.translate(
                    text=masked_text,
                    direction=direction,
                    placeholder_map=placeholder_map,
                    context=context,
                    log_cb=log_cb,
                )
            else:
                t0 = time.time()
                res_tuple = backend.translate_single(
                    masked_text=masked_text,
                    number_map=placeholder_map,
                    direction=direction,
                    context=context,
                    log_cb=log_cb,
                )
                if isinstance(res_tuple, tuple):
                    translated_masked, elapsed = res_tuple
                else:
                    translated_masked = res_tuple
                if elapsed == 0.0:
                    elapsed = time.time() - t0
        except Exception as e:
            if self.logger:
                self.logger.error(
                    message=f"{backend_name.upper()} translation error for {location_id}: {e}. Original kept.",
                    category="translation",
                    location=location_id,
                    details={"error": str(e)},
                )
            if log_cb:
                log_cb(f"  [-] {backend_name.upper()} translation error for {location_id}: {e}. Original kept.")
            self.failed_this_run.add(key)
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return TranslationResult(
                text=text,
                was_translated=False,
                was_reverted=True,
                source_backend=backend_name,
            )

        # 5. Output validation & placeholder verification
        if translated_masked:
            if verify_placeholders(translated_masked, placeholder_map, masked_text):
                self._cache_mgr.put(
                    key=key, direction=direction, fingerprint=fp, value=translated_masked, mode=mode_str
                )
                dir_cache[key] = translated_masked
                self._record_cache_access(direction, context, key)
                final_trans = unmask_protected_text(translated_masked, number_map, glossary_map)
                preview_res = (final_trans[:30] + "..") if len(final_trans) > 32 else final_trans
                if self.logger:
                    self.logger.info(
                        message=f'"{preview_src}" => "{preview_res}"',
                        category="translation",
                        location=location_id,
                        elapsed=elapsed,
                        details={"source_backend": backend_name},
                    )
                if log_cb:
                    log_cb(f'  [✓ Done in {elapsed:.1f}s] "{preview_src}" => "{preview_res}"')
                return TranslationResult(
                    text=final_trans,
                    was_translated=True,
                    was_reverted=False,
                    elapsed=elapsed,
                    source_backend=backend_name,
                )
            else:
                self.failed_this_run.add(key)
                if self.logger:
                    self.logger.warning(
                        message=f"{backend_name.upper()} output dropped placeholder(s). Original kept.",
                        category="translation",
                        location=location_id,
                        elapsed=elapsed,
                    )
                if log_cb:
                    log_cb(
                        f"  [⚠ Skipped] {location_id}: {backend_name.upper()} output dropped placeholder(s). Original kept."
                    )
                if review_log_path:
                    self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
                return TranslationResult(
                    text=text,
                    was_translated=False,
                    was_reverted=True,
                    elapsed=elapsed,
                    source_backend=backend_name,
                )

        # 6. Fallback if backend returned None
        if self.logger:
            self.logger.warning(
                message=f"Reverting {location_id} to original text & logging.",
                category="translation",
                location=location_id,
                elapsed=elapsed,
            )
        if log_cb:
            log_cb(f"  [❌ Fallback] Reverting {location_id} to original text & logging.")
        self.failed_this_run.add(key)
        if review_log_path:
            self.log_needs_review(review_log_path, location_id, chunk_id, text, key)

        return TranslationResult(
            text=text,
            was_translated=False,
            was_reverted=True,
            elapsed=elapsed,
            source_backend=backend_name,
        )

    def translate_chunk(
        self,
        text: str,
        direction: str,
        context: str | None = None,
        location_id: str = "doc",
        chunk_id: str = "0",
        review_log_path: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> TranslationResult:
        """
        Translates a single piece of text according to the selected TranslationMode.
        Returns: TranslationResult(text, was_translated, was_reverted, elapsed, source_backend)
        """
        if not should_translate(text, direction):
            return TranslationResult(text=text, was_translated=False, was_reverted=False)

        if self.mode in (
            TranslationMode.FAST_NMT,
            TranslationMode.QUALITY_NMT,
            TranslationMode.NLLB_3B,
            TranslationMode.MADLAD_3B,
        ):
            return self._translate_chunk_nmt(
                text=text,
                direction=direction,
                context=context,
                location_id=location_id,
                chunk_id=chunk_id,
                review_log_path=review_log_path,
                log_cb=log_cb,
            )
        return self._translate_chunk_llm(
            text=text,
            direction=direction,
            context=context,
            location_id=location_id,
            chunk_id=chunk_id,
            review_log_path=review_log_path,
            log_cb=log_cb,
        )
