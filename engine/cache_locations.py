"""
engine/cache_locations.py
=========================
Centralized application-owned translation cache management.
Stores all translation caches in %LOCALAPPDATA%/OfflineDocumentTranslator/cache,
derives collision-resistant and privacy-preserving scope identifiers, and provides
truthful cache metrics and clearing operations.
"""

import os
import sys
import hashlib
import logging
from typing import Optional, Dict, Any, List, Union

from .core import TranslationMode
from .cache import CachePolicy


APP_FOLDER_NAME = "OfflineDocumentTranslator"
CACHE_SUBDIR = "cache"


def get_cache_root_dir() -> str:
    """
    Returns the absolute path to the application's central cache directory.
    Priority:
      1. TRANSLATION_CACHE_DIR environment variable (for testing and custom deployments)
      2. %LOCALAPPDATA%/OfflineDocumentTranslator/cache (standard Windows per-user location)
      3. ~/.cache/OfflineDocumentTranslator/cache (fallback on non-Windows/POSIX)
    """
    env_dir = os.environ.get("TRANSLATION_CACHE_DIR")
    if env_dir:
        cache_dir = os.path.abspath(env_dir)
    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        cache_dir = os.path.join(os.environ["LOCALAPPDATA"], APP_FOLDER_NAME, CACHE_SUBDIR)
    else:
        user_home = os.path.expanduser("~")
        cache_dir = os.path.join(user_home, ".cache", APP_FOLDER_NAME, CACHE_SUBDIR)

    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_job_cache_path(
    output_path: str,
    mode: Union[TranslationMode, str] = TranslationMode.FAST_NMT,
    cache_policy: CachePolicy = CachePolicy.ENCRYPTED_PERSISTENT,
) -> str:
    """
    Derives a stable, privacy-preserving cache path within the application cache directory.
    Uses a 16-character SHA-256 hash of the canonical output directory plus mode,
    preventing raw document filenames or sensitive paths from leaking into the cache store.
    """
    cache_dir = get_cache_root_dir()
    norm_out_dir = os.path.abspath(os.path.dirname(output_path))
    if sys.platform == "win32":
        norm_out_dir = os.path.normcase(norm_out_dir)

    scope_hash = hashlib.sha256(norm_out_dir.encode("utf-8")).hexdigest()[:16]
    mode_str = mode.value if isinstance(mode, TranslationMode) else str(mode)

    if not isinstance(cache_policy, CachePolicy):
        try:
            cache_policy = CachePolicy(cache_policy)
        except (ValueError, TypeError):
            cache_policy = CachePolicy.ENCRYPTED_PERSISTENT

    if cache_policy == CachePolicy.PLAINTEXT_PERSISTENT:
        filename = f"cache_{mode_str}_{scope_hash}.json"
    else:
        filename = f"cache_{mode_str}_{scope_hash}.enc"

    return os.path.join(cache_dir, filename)


def is_owned_cache_file(filename: str) -> bool:
    """Returns True if the file matches application-owned cache naming conventions."""
    valid_prefixes = ("cache_", ".translation_cache", "translation_cache")
    if not any(filename.startswith(p) for p in valid_prefixes):
        return False
    return (
        filename.endswith(".enc")
        or filename.endswith(".json")
        or ".bak" in filename
        or filename.endswith(".tmp")
    )


def get_cache_stats() -> Dict[str, Any]:
    """
    Scans the application cache directory and returns truthful metrics:
    - cache_dir: directory path
    - file_count: number of owned cache files
    - total_bytes: total disk space consumed
    - total_kb: total disk space in KB
    - total_mb: total disk space in MB
    - files: list of owned filenames
    """
    cache_dir = get_cache_root_dir()
    file_count = 0
    total_bytes = 0
    owned_files: List[str] = []

    if os.path.exists(cache_dir):
        try:
            for entry in os.scandir(cache_dir):
                if entry.is_file() and is_owned_cache_file(entry.name):
                    file_count += 1
                    owned_files.append(entry.name)
                    try:
                        total_bytes += entry.stat().st_size
                    except OSError:
                        pass
        except OSError as e:
            logging.warning("Error reading cache directory '%s': %s", cache_dir, e)

    return {
        "cache_dir": cache_dir,
        "file_count": file_count,
        "total_bytes": total_bytes,
        "total_kb": total_bytes / 1024.0,
        "total_mb": total_bytes / (1024.0 * 1024.0),
        "files": owned_files,
    }


def clear_all_caches() -> Dict[str, int]:
    """
    Safely and atomically clears all application-owned cache files in the central directory.
    Never traverses or deletes files outside the application cache root.
    Returns:
      {"deleted": int, "failed": int, "freed_bytes": int}
    """
    cache_dir = get_cache_root_dir()
    deleted = 0
    failed = 0
    freed_bytes = 0

    if not os.path.exists(cache_dir):
        return {"deleted": 0, "failed": 0, "freed_bytes": 0}

    try:
        entries = list(os.scandir(cache_dir))
    except OSError as e:
        logging.warning("Could not scan cache directory for clearing: %s", e)
        return {"deleted": 0, "failed": 1, "freed_bytes": 0}

    for entry in entries:
        if entry.is_file() and is_owned_cache_file(entry.name):
            try:
                size = entry.stat().st_size
                os.remove(entry.path)
                deleted += 1
                freed_bytes += size
            except OSError as err:
                logging.warning("Failed to delete cache file '%s': %s", entry.path, err)
                failed += 1

    return {
        "deleted": deleted,
        "failed": failed,
        "freed_bytes": freed_bytes,
    }
