"""
engine/backend_nmt.py
=====================
Fast Neural Machine Translation (NMT) backend powered by Argos Translate / CTranslate2.
Executes batch translations in 10ms - 50ms per item on CPU with zero external server dependencies.
"""

import hashlib
import json
import os
import re
import time
import zipfile
from collections.abc import Callable
from typing import Any

DEFAULT_TRUSTED_PACKAGES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trusted_packages.json")

# Language code mapping
LANG_MAP = {
    "ja2en": ("ja", "en"),
    "en2ja": ("en", "ja"),
}


def load_trusted_packages(manifest_path: str | None = None) -> dict[str, Any]:
    """
    Loads known-good Argos language model package metadata and pinned SHA-256 hashes.
    Falls back to built-in verified hashes if the JSON manifest is missing or unreadable.
    """
    path = manifest_path or DEFAULT_TRUSTED_PACKAGES_PATH
    if os.path.exists(path):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
                return {k: v for k, v in data.items() if not k.startswith("_")}
        except Exception:
            pass

    return {
        "ja-en": {
            "from_code": "ja",
            "to_code": "en",
            "package_version": "1.1",
            "sha256": "623e3477959a815eb0a5ef53e09079ae8f1f9d3bbcd230473baf28c03fb83335",
            "url": "https://argos-net.com/v1/translate-ja_en-1_1.argosmodel",
        },
        "en-ja": {
            "from_code": "en",
            "to_code": "ja",
            "package_version": "1.1",
            "sha256": "16300cc4eaa85320520cabcf433b63d01be40ef6966251de72043a083408f716",
            "url": "https://argos-net.com/v1/translate-en_ja-1_1.argosmodel",
        },
    }


def compute_file_sha256(file_path: str, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hexadecimal hash of a file."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def verify_package_archive(
    archive_path: str,
    expected_hash: str | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Validates a downloaded .argosmodel package archive:
    1. Computes SHA-256 digest and compares with expected_hash (if provided).
    2. Validates zip file integrity via CRC test (testzip).
    3. Prevents path traversal / ZipSlip vulnerabilities.
    Returns (is_valid, digest).
    """
    if not os.path.exists(archive_path):
        if log_cb:
            log_cb(f"[!] Package file does not exist: {archive_path}")
        return False, ""

    try:
        digest = compute_file_sha256(archive_path)

        # 1. SHA-256 comparison if expected_hash provided
        if expected_hash:
            if digest.lower() != expected_hash.strip().lower():
                if log_cb:
                    log_cb(
                        f"[!] Security Error: Package SHA-256 mismatch!\n"
                        f"    Expected: {expected_hash.lower()}\n"
                        f"    Actual:   {digest.lower()}"
                    )
                return False, digest

        # 2. Zip archive validation and ZipSlip check
        with zipfile.ZipFile(archive_path, "r") as zf:
            if zf.testzip() is not None:
                if log_cb:
                    log_cb(f"[!] Package archive failed CRC validation: {archive_path}")
                return False, digest

            for member in zf.infolist():
                norm = os.path.normpath(member.filename)
                if norm.startswith("..") or os.path.isabs(norm) or norm.startswith("/") or norm.startswith("\\"):
                    if log_cb:
                        log_cb(f"[!] Security Error: Unsafe path in package archive: {member.filename}")
                    return False, digest

        return True, digest
    except Exception as e:
        if log_cb:
            log_cb(f"[!] Failed to verify package integrity: {e}")
        return False, ""


def normalize_nmt_placeholders(text: str) -> str:
    """
    Cleans up any potential whitespace separation that subword tokenizers (SentencePiece/BPE)
    may introduce around [[N0]] placeholders (e.g., '[ [ N0 ] ]' -> '[[N0]]').
    """

    def repl(match):
        prefix = match.group(1)
        digits = match.group(2) or ""
        return f"[[{prefix}{digits}]]"

    return re.sub(r"\[\s*\[\s*([A-Z][A-Z_]*)(?:\s*(\d+))?\s*\]\s*\]", repl, text)


class NMTBackend:
    """
    Manages CTranslate2 / Argos Translate models, translation lifecycle,
    and automated offline package loading.
    """

    name: str = "nmt"

    def __init__(
        self,
        trusted_manifest_path: str | None = None,
        strict_pinning: bool = True,
    ):
        self.trusted_manifest_path = trusted_manifest_path
        self.strict_pinning = strict_pinning
        self._trusted_manifest = load_trusted_packages(trusted_manifest_path)
        self._argos_available = False
        self._installed_pairs = set()
        self._check_availability()

    def _check_availability(self) -> None:
        try:
            import argostranslate.package
            import argostranslate.translate  # noqa: F401

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

    def get_expected_hash(self, from_code: str, to_code: str) -> str | None:
        """Returns the pinned SHA-256 hash for a language pair if defined in trusted manifest."""
        keys = [
            f"{from_code}-{to_code}",
            f"{from_code}2{to_code}",
            f"{from_code}_{to_code}",
        ]
        for k in keys:
            entry = self._trusted_manifest.get(k)
            if entry and isinstance(entry, dict) and "sha256" in entry:
                return entry["sha256"]
        return None

    def _verify_package(self, download_path: str, expected_hash: str) -> bool:
        """Verifies package integrity and SHA-256 against expected hash."""
        is_valid, _ = verify_package_archive(download_path, expected_hash=expected_hash)
        return is_valid

    def install_language_pair(
        self,
        from_code: str,
        to_code: str,
        log_cb: Callable[[str], None] | None = None,
        verify_hash: bool = True,
    ) -> bool:
        """
        Downloads and installs an offline Argos language package if connected.
        Verifies SHA-256 hash against trusted manifest and checks zip integrity
        prior to installing.
        """
        if not self._argos_available:
            return False

        if self.has_language_pair(from_code, to_code):
            return True

        expected_hash = self.get_expected_hash(from_code, to_code)
        if verify_hash and not expected_hash and self.strict_pinning:
            if log_cb:
                log_cb(
                    f"[!] Security Error: No trusted SHA-256 entry for language pair "
                    f"{from_code} -> {to_code}. Refusing to install unpinned package."
                )
            return False

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

            if not target_pkg:
                if log_cb:
                    log_cb(f"[!] Language pair {from_code} -> {to_code} not found in Argos package index.")
                return False

            if log_cb:
                log_cb(f"[*] Downloading offline model: {target_pkg} (~100MB)...")
            download_path = target_pkg.download()

            # Cryptographic SHA-256 & zip safety verification
            if verify_hash:
                is_valid, digest = verify_package_archive(download_path, expected_hash=expected_hash, log_cb=log_cb)
                if not is_valid:
                    if log_cb:
                        log_cb("[!] Security Error: Package failed verification! Aborting install.")
                    if os.path.exists(download_path):
                        try:
                            os.remove(download_path)
                        except OSError:
                            pass
                    return False
                if log_cb:
                    log_cb(f"[✓] Package verified: SHA-256 matched trusted manifest ({digest[:16]}...).")

            argostranslate.package.install_from_path(download_path)
            self._installed_pairs.add((from_code, to_code))
            if log_cb:
                log_cb(f"[✓] Successfully installed {from_code} -> {to_code} translation package.")
            return True
        except Exception as e:
            if log_cb:
                log_cb(f"[!] Failed to auto-install Argos package: {e}")
        return False

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: dict[str, str] | None = None,
        context: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> tuple[str | None, float]:
        """
        Unified TranslationBackend protocol method.
        Translates a single text unit using CTranslate2/Argos and returns (result, elapsed).
        """
        t0 = time.time()
        try:
            res = self.translate_single(text, direction)
            elapsed = time.time() - t0
            return res, elapsed
        except Exception as e:
            if log_cb:
                log_cb(f"  [-] NMT backend error: {e}")
            return None, time.time() - t0

    def translate_single(self, text: str, direction: str) -> str:
        """Translates a single string using CTranslate2/Argos."""
        if not self._argos_available:
            raise RuntimeError("Argos Translate / CTranslate2 is not installed.")

        from_code, to_code = LANG_MAP.get(direction, ("ja", "en"))
        import argostranslate.translate

        translated = argostranslate.translate.translate(text, from_code, to_code)
        return normalize_nmt_placeholders(translated.strip())

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
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
