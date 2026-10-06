"""
engine/core.py
==============
Core format-agnostic translation engine.
Orchestrates two translation modes:
  - Machine Translation Mode (CTranslate2 / Google MADLAD-400 3B, Apache 2.0)
  - AI Translation Mode (Local LLM via Ollama, Google Gemma 4 E2B IT QAT, Apache 2.0)
  - Unified atomic caching, single-pass number masking, and bidirectional gates.
"""

import hashlib
import html
import json
import logging
import re
import time
import unicodedata
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


@dataclass(frozen=True)
class NumericAuditResult:
    """Encapsulates the structured evaluation of numeric token preservation in NMT."""

    passed: bool
    missing: list[str]
    added: list[str]
    src_tokens: list[str]
    tgt_tokens: list[str]

    def __iter__(self):
        """Supports backward-compatible unpacking: (passed, missing) = result."""
        return iter((self.passed, self.missing))

    def __getitem__(self, index):
        """Supports index-based tuple access for backward compatibility: (passed, missing)[index]."""
        return (self.passed, self.missing)[index]

    def summary(self) -> str:
        parts = []
        if self.missing:
            parts.append(f"missing {self.missing}")
        if self.added:
            parts.append(f"added {self.added}")
        return "; ".join(parts) if parts else "OK"



class TranslationMode(str, Enum):
    MACHINE_TRANSLATION = "machine_translation"  # 100% CTranslate2 / Google MADLAD-400 3B (Apache 2.0)
    AI_TRANSLATION = "ai_translation"  # 100% Local LLM via Ollama (Gemma 4, Apache 2.0)

    # Backward-compatible aliases for existing caches
    FAST_NMT = "fast_nmt"
    QUALITY_NMT = "quality_nmt"
    NLLB_3B = "nllb_3b"
    MADLAD_3B = "madlad_3b"
    PURE_LLM = "pure_llm"


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


# Case-sensitive units (distinguish bytes B vs bits b, etc.)
_CASE_SENSITIVE_UNITS = {
    # Data units
    "PB", "TB", "GB", "MB", "KB", "B",
    "Pb", "Tb", "Gb", "Mb", "Kb", "b",
    "Gbps", "Mbps", "Kbps", "bps",
    # Power / Frequency / Electrical / Pressure
    "GHz", "MHz", "kHz", "Hz",
    "GW", "MW", "kW", "W",
    "kV", "mV", "V",
    "mA", "A",
    "kWh", "MPa", "kPa", "Pa",
}

# Case-insensitive units (normalized to lowercase)
_CASE_INSENSITIVE_UNITS = {
    "kg", "mg", "g",
    "km", "cm", "mm", "nm", "m",
    "ml", "l",
    "bar", "psi", "rpm", "db", "deg",
    "ms", "ns", "min", "hr", "hrs",
}

# Attached single-letter scale suffixes (e.g. 42M, 10k, 1.5B, 1T)
_ATTACHED_SCALE_SUFFIXES = {
    "M": "M",
    "k": "k",
    "K": "k",
    "B": "B",
    "T": "T",
}

# Scale words mapped to canonical unit (e.g. 42 million -> 42M)
_SCALE_WORDS_MAP = {
    "million": "M",
    "thousand": "k",
    "billion": "B",
    "trillion": "T",
}

# Spelled-out ordinals (first through tenth)
_ORDINAL_WORDS_EN = {
    "1": "first", "2": "second", "3": "third", "4": "fourth", "5": "fifth",
    "6": "sixth", "7": "seventh", "8": "eighth", "9": "ninth", "10": "tenth",
}
_ORDINAL_WORD_TO_NUM = {v: k for k, v in _ORDINAL_WORDS_EN.items()}

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
_MONTH_ABBRS_EN = {
    "1": "jan", "2": "feb", "3": "mar", "4": "apr",
    "6": "jun", "7": "jul", "8": "aug",
    "9": "sep", "10": "oct", "11": "nov", "12": "dec",
}


def extract_numeric_tokens(text: str) -> list[str]:
    """
    Extracts numeric tokens, preserving:
    - Pure numbers: '2024', '1250000', '15.5'
    - Units (case-sensitive or normalized): '500kg', '32GB', '10Gbps'
    - Attached scales: '42M', '42k', '1.5B'
    - Ordinals: '1', '2' (from '1st', '2nd')
    - Percentage: '15.5' (with '%' stripped to match existing behavior)
    Excludes identifier-like strings containing underscores (e.g. '500kg_v2').
    """
    normalized = unicodedata.normalize("NFKC", text)

    pattern = re.compile(
        r"(?<![A-Za-z0-9_])(\d+(?:,\d{3})*(?:\.\d+)?)(%|\s*[A-Za-z]+)?(?![A-Za-z0-9_])"
    )
    tokens = []
    for match in pattern.finditer(normalized):
        raw_num = match.group(1).replace(",", "")
        suffix = match.group(2)

        if not suffix:
            tokens.append(raw_num)
            continue

        has_space = suffix.startswith(" ")
        s_clean = suffix.strip()

        if s_clean == "%":
            tokens.append(raw_num)
        elif s_clean.lower() in ("st", "nd", "rd", "th"):
            tokens.append(raw_num)
        elif not has_space and s_clean in _ATTACHED_SCALE_SUFFIXES:
            tokens.append(f"{raw_num}{_ATTACHED_SCALE_SUFFIXES[s_clean]}")
        elif s_clean in _CASE_SENSITIVE_UNITS:
            tokens.append(f"{raw_num}{s_clean}")
        elif s_clean.lower() in _CASE_INSENSITIVE_UNITS:
            tokens.append(f"{raw_num}{s_clean.lower()}")
        else:
            if has_space:
                tokens.append(raw_num)
            else:
                # Attached unrecognized alphanumeric suffix -> identifier, ignore
                pass
    return tokens


def _extract_scale_word_equivalences(text: str) -> Counter:
    """Extracts phrases like '42 million' -> '42M', '10 thousand' -> '10k'."""
    norm = unicodedata.normalize("NFKC", text)
    scales: Counter[str] = Counter()
    for m in re.finditer(
        r"(?<![A-Za-z0-9_])(\d+(?:,\d{3})*(?:\.\d+)?)\s+(million|billion|thousand|trillion)(?![A-Za-z0-9_])",
        norm,
        re.IGNORECASE,
    ):
        raw_num = m.group(1).replace(",", "")
        unit = _SCALE_WORDS_MAP[m.group(2).lower()]
        scales[f"{raw_num}{unit}"] += 1
    return scales


def _extract_source_month_numbers(text: str) -> Counter:
    """Detects numbers in source text explicitly used in month/date context."""
    norm = unicodedata.normalize("NFKC", text)
    months: Counter[str] = Counter()
    # Japanese 1月 through 12月
    for m in re.finditer(r"(?<!\d)(1[0-2]|[1-9])\s*月", norm):
        months[m.group(1)] += 1
    # English month words (for en2ja)
    for num, name in _MONTH_NAMES_EN.items():
        if re.search(rf"\b{name}\b", norm, re.IGNORECASE):
            months[num] += 1
    return months


def _extract_target_month_counts(text: str) -> Counter:
    """Extracts word-bounded English month names and abbreviations from target text."""
    norm = unicodedata.normalize("NFKC", text)
    months: Counter[str] = Counter()
    for num, name in _MONTH_NAMES_EN.items():
        count = len(re.findall(rf"\b{name}\b", norm, re.IGNORECASE))
        if count > 0:
            months[num] += count
    for num, abbr in _MONTH_ABBRS_EN.items():
        count = len(re.findall(rf"\b{abbr}\.?(?![A-Za-z0-9_])", norm, re.IGNORECASE))
        if count > 0:
            months[num] += count
    return months


def _extract_target_ordinal_counts(text: str) -> Counter:
    """Extracts word-bounded spelled-out English ordinals (first -> 1, second -> 2)."""
    norm = unicodedata.normalize("NFKC", text)
    ordinals: Counter[str] = Counter()
    for word, num in _ORDINAL_WORD_TO_NUM.items():
        count = len(re.findall(rf"\b{word}\b", norm, re.IGNORECASE))
        if count > 0:
            ordinals[num] += count
    return ordinals


def verify_nmt_numbers(source_text: str, target_text: str) -> NumericAuditResult:
    """
    Checks that numbers present in the source text appear in the target text.
    Preserves units and scale multipliers (e.g. '500kg', '42M').
    Accounts for localized month names when source has month context (e.g. '10月' -> 'October').
    Supports spelled-out English ordinals (e.g. 第1 -> 'first').
    Compares token counts (multiset semantics).
    Returns NumericAuditResult(passed, missing, added, src_tokens, tgt_tokens).
    """
    src_nums = extract_numeric_tokens(source_text)
    tgt_nums = extract_numeric_tokens(target_text)

    if not src_nums:
        return NumericAuditResult(
            passed=True, missing=[], added=tgt_nums, src_tokens=[], tgt_tokens=tgt_nums
        )

    src_month_nums = _extract_source_month_numbers(source_text)
    tgt_month_counts = _extract_target_month_counts(target_text)
    tgt_ordinal_counts = _extract_target_ordinal_counts(target_text)
    tgt_scale_words = _extract_scale_word_equivalences(target_text)

    tgt_counter = Counter(tgt_nums)
    missing: list[str] = []

    for num in src_nums:
        # 1. Direct match in target numeric tokens
        if tgt_counter.get(num, 0) > 0:
            tgt_counter[num] -= 1
            continue

        # 2. Scale word match (e.g. source '42M' matched by target '42 million')
        if num in tgt_scale_words and tgt_scale_words[num] > 0:
            tgt_scale_words[num] -= 1
            base_num = re.sub(r"[A-Za-z]+$", "", num)
            if tgt_counter.get(base_num, 0) > 0:
                tgt_counter[base_num] -= 1
            continue

        # 3. Spelled-out ordinal word match (e.g. 第1 -> 'first')
        if num in tgt_ordinal_counts and tgt_ordinal_counts[num] > 0:
            tgt_ordinal_counts[num] -= 1
            continue

        # 4. Month name match: ONLY if source number was in explicit month context
        if num in src_month_nums and src_month_nums[num] > 0 and tgt_month_counts.get(num, 0) > 0:
            src_month_nums[num] -= 1
            tgt_month_counts[num] -= 1
            continue

        missing.append(num)

    added = [k for k, count in tgt_counter.items() if count > 0 for _ in range(count)]
    passed = len(missing) == 0 and len(added) == 0

    return NumericAuditResult(
        passed=passed,
        missing=missing,
        added=added,
        src_tokens=src_nums,
        tgt_tokens=tgt_nums,
    )



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
        mode: TranslationMode = TranslationMode.MACHINE_TRANSLATION,
        glossary: dict[str, str] | None = None,
        min_free_ram_mb: int = 150,
        context_window: int = 4096,
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
    def madlad_backend(self):
        if self._madlad_backend is None:
            from engine.backend_madlad import MADLADBackend

            self._madlad_backend = MADLADBackend()
        return self._madlad_backend

    @property
    def _nmt_backend(self):
        return self.madlad_backend

    @_nmt_backend.setter
    def _nmt_backend(self, backend):
        self._madlad_backend = backend

    @property
    def _nllb_backend(self):
        return self.madlad_backend

    @_nllb_backend.setter
    def _nllb_backend(self, backend):
        self._madlad_backend = backend

    @property
    def llm_backend(self):
        if self._llm_backend is None:
            from engine.backend_llm import LLMBackend

            self._llm_backend = LLMBackend(
                model_name=self.model_name, ollama_url=self.ollama_url, context_window=self.context_window
            )
        return self._llm_backend

    def ensure_backend_exclusive(self, mode: TranslationMode | str | None = None) -> None:
        """
        Enforces strict mutual exclusivity between Machine Translation (MADLAD) and AI Translation (Ollama).
        Guarantees that invoking one engine cleanly evicts the other from system RAM before inference starts.
        """
        effective_mode = mode if mode is not None else self.mode
        is_mt = effective_mode in (
            TranslationMode.MACHINE_TRANSLATION,
            TranslationMode.FAST_NMT,
            TranslationMode.QUALITY_NMT,
            TranslationMode.NLLB_3B,
            TranslationMode.MADLAD_3B,
            "machine_translation",
            "fast_nmt",
            "quality_nmt",
            "nllb_3b",
            "madlad_3b",
        )
        if is_mt:
            # We are using MADLAD: evict all models from Ollama's memory
            try:
                from engine.ollama_manager import get_ollama_manager

                get_ollama_manager().unload_all_models()
            except Exception:
                pass
        else:
            # We are using AI Translation: evict MADLAD from memory
            if self._madlad_backend is not None and hasattr(self._madlad_backend, "unload"):
                self._madlad_backend.unload()

    def get_backend(self, mode: TranslationMode | str | None = None) -> TranslationBackend:
        """Returns the active or requested translation backend instance."""
        if mode is None and self._custom_backend is not None:
            return self._custom_backend
        effective_mode = mode if mode is not None else self.mode
        self.ensure_backend_exclusive(effective_mode)
        if effective_mode in (
            TranslationMode.MACHINE_TRANSLATION,
            TranslationMode.FAST_NMT,
            TranslationMode.QUALITY_NMT,
            TranslationMode.NLLB_3B,
            TranslationMode.MADLAD_3B,
            "machine_translation",
            "fast_nmt",
            "quality_nmt",
            "nllb_3b",
            "madlad_3b",
        ):
            return self.madlad_backend
        return self.llm_backend

    def set_backend(self, backend: TranslationBackend, mode: TranslationMode | str | None = None) -> None:
        """Registers a custom or replacement translation backend."""
        if mode is None:
            self._custom_backend = backend
        elif mode in (
            TranslationMode.MACHINE_TRANSLATION,
            TranslationMode.FAST_NMT,
            TranslationMode.QUALITY_NMT,
            TranslationMode.NLLB_3B,
            TranslationMode.MADLAD_3B,
            "machine_translation",
            "fast_nmt",
            "quality_nmt",
            "nllb_3b",
            "madlad_3b",
        ):
            self._madlad_backend = backend
        else:
            self._llm_backend = backend

    def unload_backends(self) -> None:
        """Unloads in-memory backends (both MADLAD and Ollama) and frees model memory."""
        if self._madlad_backend is not None and hasattr(self._madlad_backend, "unload"):
            self._madlad_backend.unload()
        if self._llm_backend is not None and hasattr(self._llm_backend, "unload"):
            self._llm_backend.unload()
        try:
            from engine.ollama_manager import get_ollama_manager

            get_ollama_manager().unload_all_models()
        except Exception:
            pass
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
        if self.mode not in (TranslationMode.PURE_LLM, TranslationMode.AI_TRANSLATION, "pure_llm", "ai_translation"):
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
                val = str(res).strip() if isinstance(res, str) else ""
            elif hasattr(backend, "translate"):
                res_tuple = backend.translate(text=term, direction=direction)
                res = res_tuple[0] if isinstance(res_tuple, tuple) else res_tuple
                val = str(res).strip() if isinstance(res, str) else ""
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
            cache_audit = verify_nmt_numbers(text, cached_trans)
            if cache_audit.passed:
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
            else:
                if self.logger:
                    self.logger.warning(
                        message=f"Bypassing invalid cached translation for '{location_id}': {cache_audit.summary()}",
                        category="cache",
                        location=location_id,
                    )
                self._cache_mgr.delete(key=key, direction=direction, fingerprint=fp, mode=mode_str)
                if key in dir_cache:
                    del dir_cache[key]
                cached_trans = None

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
            engine_label = "Machine Translation (MADLAD-400 3B)"
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
                        "The system ran out of RAM for this model. Please close other applications "
                        "or ensure enough system memory is free."
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

        # 5. Post-translation glossary substitution
        final_trans = self._apply_post_translation_glossary(translated_raw, text, direction, backend)

        # 6. Post-translation numeric audit gate
        audit = verify_nmt_numbers(text, final_trans)
        if not audit.passed:
            warn_msg = f"NMT numeric check failed ({audit.summary()}) in '{location_id}'. Original kept."
            if self.logger:
                self.logger.warning(message=warn_msg, category="translation", location=location_id)
            if log_cb:
                log_cb(f"  [⚠ Number Audit] {warn_msg}")
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

        # 7. Store in cache & return (Only reached if audit passed)
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
            tag = "⚡ Machine Translation"
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
            TranslationMode.MACHINE_TRANSLATION,
            TranslationMode.FAST_NMT,
            TranslationMode.QUALITY_NMT,
            TranslationMode.NLLB_3B,
            TranslationMode.MADLAD_3B,
            "machine_translation",
            "fast_nmt",
            "quality_nmt",
            "nllb_3b",
            "madlad_3b",
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
