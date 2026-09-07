"""
engine/core.py
==============
Core format-agnostic translation engine.
Orchestrates two translation modes:
  - Fast NMT Mode (ArgosTranslate / CTranslate2) for high-speed offline translation
  - Pure LLM Mode (Ollama) with isomorphic retry & custom glossary enforcement
  - Unified atomic caching, single-pass number masking, and bidirectional gates.
"""

import os
import re
import json
import time
import hashlib
import tempfile
import html
from collections import Counter
import psutil
import requests
from enum import Enum
from typing import Dict, Tuple, Optional, Any, List, Callable

from .errors import ErrorCode, TranslatorError

# Supported translation directions
DIRECTIONS = ("ja2en", "en2ja")
PLACEHOLDER_PATTERN = re.compile(r"\[\[[A-Z][A-Z_0-9]*\]\]")


class TranslationMode(str, Enum):
    FAST_NMT = "fast_nmt"  # 100% CTranslate2 / Argos (blazing fast, 0.02s/chunk)
    PURE_LLM = "pure_llm"  # 100% Local LLM via Ollama


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


def mask_numbers(text: str) -> Tuple[str, Dict[str, str]]:
    """
    Masks numbers, floats, percentages, and basic formulas into placeholders [[N0]], [[N1]].
    Uses a single-pass regex replacement to guarantee placeholders are never recursively nested.
    Identical numbers in the same string share the same placeholder to preserve structure.
    """
    pattern = re.compile(r"(\d+(?:\.\d+)?(?:[,\s]*[+\-xX*/%][\s]*\d+(?:\.\d+)?)*%?)")
    number_map: Dict[str, str] = {}
    val_to_placeholder: Dict[str, str] = {}
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


def mask_glossary_terms(text: str, glossary: Dict[str, str]) -> Tuple[str, Dict[str, str]]:
    """Replaces source glossary terms with stable placeholders before translation."""
    terms = [term for term in glossary if term]
    if not terms:
        return text, {}

    # Longest-first matching prevents a shorter glossary term from consuming a longer one.
    pattern = re.compile("|".join(re.escape(term) for term in sorted(terms, key=len, reverse=True)))
    term_placeholders: Dict[str, str] = {}
    glossary_map: Dict[str, str] = {}

    def repl(match):
        term = match.group(0)
        if term not in term_placeholders:
            placeholder = f"[[GLOSSARY_{_alphabetic_id(len(term_placeholders))}]]"
            term_placeholders[term] = placeholder
            glossary_map[placeholder] = glossary[term]
        return term_placeholders[term]

    return pattern.sub(repl, text), glossary_map


def verify_placeholders(
    translated_text: str,
    placeholder_map: Dict[str, str],
    masked_source: Optional[str] = None
) -> bool:
    """Checks that the exact placeholder multiset survived translation uncorrupted."""
    expected_text = masked_source if masked_source is not None else " ".join(placeholder_map.keys())
    expected = Counter(PLACEHOLDER_PATTERN.findall(expected_text))
    actual = Counter(PLACEHOLDER_PATTERN.findall(translated_text))
    return actual == expected


def unmask_numbers(text: str, number_map: Dict[str, str]) -> str:
    """Replaces placeholders [[N#]] back with their original numerical values."""
    for placeholder, original in number_map.items():
        text = text.replace(placeholder, original)
    return text


def unmask_protected_text(
    text: str,
    number_map: Dict[str, str],
    glossary_map: Dict[str, str]
) -> str:
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
    direction: str,
    masked_text: str,
    placeholder_map: Dict[str, str],
    context: Optional[str] = None
) -> Tuple[str, str]:
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
        glossary: Optional[Dict[str, str]] = None,
        min_free_ram_mb: int = 150,
        context_window: int = 2048,
        cache_file: str = "translation_cache.json",
        allow_llm: Optional[bool] = None,
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.mode = mode
        self.glossary = glossary or {}
        self.min_free_ram_mb = min_free_ram_mb
        self.context_window = context_window
        self.cache_file = cache_file
        self.allow_llm = allow_llm

        self.cache: Dict[str, Any] = {}
        self.failed_this_run: set = set()
        self.logged_failed_keys: set = set()

        # Lazy-loaded backend instances
        self._nmt_backend = None
        self._llm_backend = None

    @property
    def nmt_backend(self):
        if self._nmt_backend is None:
            try:
                from .backend_nmt import NMTBackend
            except (ImportError, ValueError):
                from engine.backend_nmt import NMTBackend
            self._nmt_backend = NMTBackend()
        return self._nmt_backend

    @property
    def llm_backend(self):
        if self._llm_backend is None:
            try:
                from .backend_llm import LLMBackend
            except (ImportError, ValueError):
                from engine.backend_llm import LLMBackend
            self._llm_backend = LLMBackend(
                model_name=self.model_name,
                ollama_url=self.ollama_url,
                context_window=self.context_window
            )
        return self._llm_backend

    def load_cache(self, direction: str) -> None:
        """Loads cache namespaced by mode -> direction -> configuration fingerprint -> hash."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.cache = json.load(f)
            except Exception:
                self.cache = {}
        else:
            self.cache = {}

        self._get_direction_cache(direction)

    def _cache_fingerprint(self, context: Optional[str] = None) -> str:
        """Prevents reuse across models, glossaries, and context-sensitive translations."""
        payload = {
            "schema": 3,
            "model": self.model_name,
            "glossary": sorted(self.glossary.items()),
            "context": context or "",
            "llm_enabled": self.allow_llm,
        }
        encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:20]

    def _get_direction_cache(self, direction: str, context: Optional[str] = None) -> Dict[str, str]:
        """Returns the cache bucket for one fully specified translation configuration."""
        mode_str = self.mode.value if isinstance(self.mode, TranslationMode) else str(self.mode)
        mode_cache = self.cache.setdefault(mode_str, {})
        direction_cache = mode_cache.setdefault(direction, {})
        return direction_cache.setdefault(self._cache_fingerprint(context), {})

    def save_cache_atomically(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Saves cache via .tmp file replacement to prevent corruption on crash."""
        cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
        temp_file = None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            fd, temp_file = tempfile.mkstemp(prefix=".translation_cache_", suffix=".tmp", dir=cache_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, ensure_ascii=False, indent=2)
            os.replace(temp_file, self.cache_file)
        except Exception as e:
            if log_cb:
                log_cb(f"[!] Warning: Failed to save translation cache: {e}")
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass

    def check_and_clear_memory(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Flushes Ollama KV/model if available system RAM falls below threshold."""
        try:
            free_ram_mb = psutil.virtual_memory().available / (1024 * 1024)
            if free_ram_mb < self.min_free_ram_mb:
                if log_cb:
                    log_cb(f"[!] Low RAM ({free_ram_mb:.0f} MB). Flushing memory...")
                requests.post(
                    f"{self.ollama_url}/api/generate",
                    json={"model": self.model_name, "keep_alive": 0},
                    timeout=5
                )
                time.sleep(3)
        except Exception:
            pass

    def flush_model(self) -> None:
        """Explicitly unloads model from VRAM/RAM after a document completes."""
        if self.mode == TranslationMode.FAST_NMT:
            return
        try:
            requests.post(
                f"{self.ollama_url}/api/generate",
                json={"model": self.model_name, "keep_alive": 0},
                timeout=5
            )
        except Exception:
            pass

    def log_needs_review(
        self,
        review_log_path: str,
        location_id: str,
        chunk_id: str,
        original_text: str,
        key: str
    ) -> None:
        """Logs unrecoverable translation failures to audit log."""
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
        try:
            with open(review_log_path, "a", encoding="utf-8") as f:
                f.write(
                    f"Location: {location_id} | Chunk ID: {chunk_id}\n"
                    f"Original Text: {original_text}\n"
                    f"{'-' * 50}\n"
                )
        except Exception:
            pass

    def translate_chunk(
        self,
        text: str,
        direction: str,
        context: Optional[str] = None,
        location_id: str = "doc",
        chunk_id: str = "0",
        review_log_path: Optional[str] = None,
        log_cb: Optional[Callable[[str], None]] = None
    ) -> Tuple[str, bool, bool]:
        """
        Translates a single piece of text according to the selected TranslationMode.
        Returns: (result_text, was_translated, was_reverted)
        """
        if not should_translate(text, direction):
            return text, False, False

        glossary_masked_text, glossary_map = mask_glossary_terms(text, self.glossary)
        masked_text, number_map = mask_numbers(glossary_masked_text)
        placeholder_map = {**number_map, **glossary_map}
        key = hash_text(masked_text)

        dir_cache = self._get_direction_cache(direction, context)

        # 1. Check persistent cache
        if key in dir_cache:
            cached_trans = dir_cache[key]
            if verify_placeholders(cached_trans, placeholder_map, masked_text):
                final_cached = unmask_protected_text(cached_trans, number_map, glossary_map)
                preview_src = (text[:24] + "..") if len(text) > 26 else text
                preview_res = (final_cached[:24] + "..") if len(final_cached) > 26 else final_cached
                if log_cb:
                    log_cb(f"  [⚡ Cache] {location_id}: \"{preview_src}\" => \"{preview_res}\"")
                return final_cached, True, False
            del dir_cache[key]

        # 2. Check runtime failure tracker
        if key in self.failed_this_run:
            if log_cb:
                log_cb(f"  [⏩ Skipped] {location_id}")
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return text, False, True

        preview_src = (text[:30] + "..") if len(text) > 32 else text

        # ================= PATH A: FAST NMT MODE =================
        if self.mode == TranslationMode.FAST_NMT:
            nmt_ready = self.nmt_backend.is_ready(direction)
            if nmt_ready:
                t0 = time.time()
                try:
                    nmt_res = self.nmt_backend.translate_single(masked_text, direction)
                    elapsed = time.time() - t0

                    if verify_placeholders(nmt_res, placeholder_map, masked_text):
                        dir_cache[key] = nmt_res
                        final_trans = unmask_protected_text(nmt_res, number_map, glossary_map)
                        preview_res = (final_trans[:30] + "..") if len(final_trans) > 32 else final_trans
                        path_tag = "⚡ Fast NMT"
                        if log_cb:
                            log_cb(f"  [{path_tag} in {elapsed:.2f}s] {location_id}: \"{preview_src}\" => \"{preview_res}\"")
                        return final_trans, True, False
                    else:
                        self.failed_this_run.add(key)
                        if log_cb:
                            log_cb(f"  [⚠ Skipped] {location_id}: NMT output dropped placeholder(s). Original kept.")
                        if review_log_path:
                            self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
                        return text, False, True
                except Exception as e:
                    if log_cb:
                        log_cb(f"  [-] NMT translation error for {location_id}: {e}. Original kept.")
                    self.failed_this_run.add(key)
                    if review_log_path:
                        self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
                    return text, False, True

            if not nmt_ready:
                raise TranslatorError(ErrorCode.E08, detail=f"Fast NMT cannot translate {direction}: the Argos language package is not installed.")

        # ================= PATH B: PURE LLM =================
        if self.allow_llm is not False:
            self.check_and_clear_memory(log_cb)

        if log_cb:
            path_label = "🤖 Pure LLM"
            log_cb(f"  [{path_label}] {location_id}: \"{preview_src}\"...")

        if self.allow_llm is False:
            if log_cb:
                log_cb(f"  [❌ Fallback] No LLM backend is available for {location_id}.")
            self.failed_this_run.add(key)
            if review_log_path:
                self.log_needs_review(review_log_path, location_id, chunk_id, text, key)
            return text, False, True

        # Standard LLM translate with isomorphic retry
        llm_result, elapsed = self.llm_backend.translate_single(
            masked_text=masked_text,
            number_map=placeholder_map,
            direction=direction,
            context=context,
            log_cb=log_cb
        )

        if llm_result:
            dir_cache[key] = llm_result
            final_trans = unmask_protected_text(llm_result, number_map, glossary_map)
            preview_res = (final_trans[:30] + "..") if len(final_trans) > 32 else final_trans
            if log_cb:
                log_cb(f"  [✓ Done in {elapsed:.1f}s] \"{preview_src}\" => \"{preview_res}\"")
            return final_trans, True, False

        # ================= Fallback: Revert & Log =================
        if log_cb:
            log_cb(f"  [❌ Fallback] Reverting {location_id} to original text & logging.")
        self.failed_this_run.add(key)
        if review_log_path:
            self.log_needs_review(review_log_path, location_id, chunk_id, text, key)

        return text, False, True
