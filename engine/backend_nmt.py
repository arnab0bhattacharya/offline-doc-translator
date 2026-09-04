"""
engine/backend_nmt.py
=====================
Fast Neural Machine Translation (NMT) backend powered by Argos Translate / CTranslate2.
Executes batch translations in 10ms - 50ms per item on CPU with zero external server dependencies.
"""

import re
import os
from typing import List, Optional, Tuple, Dict, Any, Callable

# Language code mapping
LANG_MAP = {
    "ja2en": ("ja", "en"),
    "en2ja": ("en", "ja"),
}


def normalize_nmt_placeholders(text: str) -> str:
    """
    Cleans up any potential whitespace separation that subword tokenizers (SentencePiece/BPE)
    may introduce around [[N0]] placeholders (e.g., '[ [ N0 ] ]' -> '[[N0]]').
    """
    def repl(match):
        prefix = match.group(1)
        digits = match.group(2) or ""
        return f"[[{prefix}{digits}]]"

    return re.sub(
        r"\[\s*\[\s*([A-Z][A-Z_]*)(?:\s*(\d+))?\s*\]\s*\]",
        repl,
        text
    )


class NMTBackend:
    """
    Manages CTranslate2 / Argos Translate models, translation lifecycle,
    and automated offline package loading.
    """

    def __init__(self):
        self._argos_available = False
        self._installed_pairs = set()
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import argostranslate.translate
            import argostranslate.package
            self._argos_available = True
            self._scan_installed_packages()
        except ImportError:
            self._argos_available = False

    def is_available(self) -> bool:
        """Returns True if the underlying CTranslate2/Argos library is installed."""
        return self._argos_available

    def _scan_installed_packages(self) -> None:
        if not self._argos_available:
            return
        try:
            import argostranslate.package
            installed = argostranslate.package.get_installed_packages()
            for pkg in installed:
                self._installed_pairs.add((pkg.from_code, pkg.to_code))
        except Exception:
            pass

    def has_language_pair(self, from_code: str, to_code: str) -> bool:
        """Checks if language pair models are locally installed and ready."""
        return (from_code, to_code) in self._installed_pairs

    def is_ready(self, direction: str) -> bool:
        """Returns whether the local backend and requested language package are ready."""
        pair = LANG_MAP.get(direction)
        return bool(pair and self._argos_available and self.has_language_pair(*pair))

    def install_language_pair(
        self,
        from_code: str,
        to_code: str,
        log_cb: Optional[Callable[[str], None]] = None
    ) -> bool:
        """
        Downloads and installs an offline Argos language package if connected.
        """
        if not self._argos_available:
            return False

        if self.has_language_pair(from_code, to_code):
            return True

        try:
            import argostranslate.package
            if log_cb:
                log_cb(f"[*] Updating Argos package index for {from_code} -> {to_code}...")
            argostranslate.package.update_package_index()
            available = argostranslate.package.get_available_packages()

            target_pkg = None
            for pkg in available:
                if pkg.from_code == from_code and pkg.to_code == to_code:
                    target_pkg = pkg
                    break

            if target_pkg:
                if log_cb:
                    log_cb(f"[*] Downloading offline model: {target_pkg} (~100MB)...")
                download_path = target_pkg.download()
                argostranslate.package.install_from_path(download_path)
                self._installed_pairs.add((from_code, to_code))
                if log_cb:
                    log_cb(f"[✓] Successfully installed {from_code} -> {to_code} translation package.")
                return True
        except Exception as e:
            if log_cb:
                log_cb(f"[!] Failed to auto-install Argos package: {e}")
        return False

    def translate_single(self, text: str, direction: str) -> str:
        """Translates a single string using CTranslate2/Argos."""
        if not self._argos_available:
            raise RuntimeError("Argos Translate / CTranslate2 is not installed.")

        from_code, to_code = LANG_MAP.get(direction, ("ja", "en"))
        import argostranslate.translate

        translated = argostranslate.translate.translate(text, from_code, to_code)
        return normalize_nmt_placeholders(translated.strip())

    def translate_batch(self, texts: List[str], direction: str) -> List[str]:
        """Translates a batch of strings."""
        if not self._argos_available:
            raise RuntimeError("Argos Translate / CTranslate2 is not installed.")

        from_code, to_code = LANG_MAP.get(direction, ("ja", "en"))
        import argostranslate.translate

        results = []
        for t in texts:
            if not t.strip():
                results.append(t)
            else:
                res = argostranslate.translate.translate(t, from_code, to_code)
                results.append(normalize_nmt_placeholders(res.strip()))
        return results
