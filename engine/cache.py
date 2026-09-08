"""
engine/cache.py
===============
Modular translation caching subsystem.
Provides abstract TranslationCache interface, JSONFileCache with atomic writes and TTL,
and NullCache for ephemeral/privacy-sensitive sessions.
"""

from abc import ABC, abstractmethod
import os
import json
import time
import tempfile
from typing import Optional, Dict, Any, Callable

CACHE_TTL_DAYS = 30


class TranslationCache(ABC):
    """Abstract base class defining the contract for translation caches."""

    @abstractmethod
    def get(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> Optional[str]:
        """Retrieves a cached translation by key, direction, configuration fingerprint, and mode."""
        pass

    @abstractmethod
    def put(self, key: str, direction: str, fingerprint: str, value: str, mode: str = "default") -> None:
        """Stores a translated string in cache."""
        pass

    @abstractmethod
    def delete(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> bool:
        """Deletes an entry from cache (e.g., if placeholders were invalidated). Returns True if deleted."""
        pass

    @abstractmethod
    def save(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Persists cache changes to storage if applicable."""
        pass

    @abstractmethod
    def clear(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Wipes all cached entries."""
        pass

    @abstractmethod
    def load(self, direction: Optional[str] = None) -> None:
        """Loads cached entries from persistent storage."""
        pass

    @abstractmethod
    def record_access(
        self,
        key: str,
        direction: str,
        fingerprint: str,
        mode: str = "default",
        timestamp: Optional[float] = None
    ) -> None:
        """Records the last access time of an entry for TTL pruning."""
        pass

    @abstractmethod
    def prune(self, now: Optional[float] = None) -> int:
        """Removes expired entries according to TTL policy. Returns number of pruned entries."""
        pass


class JSONFileCache(TranslationCache):
    """
    JSON file-backed persistent cache with atomic replacement and TTL pruning.
    Maintains exact schema compatibility:
    {
        "_meta": {"last_prune": float},
        "_timestamps": {"<mode>|<direction>|<fingerprint>|<key>": float},
        "<mode>": {
            "<direction>": {
                "<fingerprint>": {
                    "<key>": "<translated_text>"
                }
            }
        }
    }
    """

    def __init__(self, cache_file: str = "translation_cache.json", ttl_days: int = CACHE_TTL_DAYS):
        self.cache_file = cache_file
        self.ttl_days = ttl_days
        self._data: Dict[str, Any] = {}
        self._loaded: bool = False

    @property
    def data(self) -> Dict[str, Any]:
        """Direct access to internal dict for backward compatibility."""
        return self._data

    @data.setter
    def data(self, new_data: Dict[str, Any]) -> None:
        self._data = new_data
        self._loaded = True

    def load(self, direction: Optional[str] = None) -> None:
        """Loads cache file from disk, automatically triggering once-per-day TTL prune."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            self._data = {}
        self._loaded = True

        # Prune stale entries at most once per day
        meta = self._data.get("_meta", {})
        last_prune = meta.get("last_prune", 0)
        now = time.time()
        if now - last_prune > 86400:
            pruned = self.prune(now)
            self._data.setdefault("_meta", {})["last_prune"] = now
            if pruned > 0:
                self.save()

    def get_bucket(self, direction: str, fingerprint: str, mode: str = "default") -> Dict[str, str]:
        """Returns the dictionary bucket corresponding to mode, direction, and fingerprint."""
        mode_cache = self._data.setdefault(mode, {})
        direction_cache = mode_cache.setdefault(direction, {})
        return direction_cache.setdefault(fingerprint, {})

    def get(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> Optional[str]:
        bucket = self.get_bucket(direction, fingerprint, mode)
        return bucket.get(key)

    def put(self, key: str, direction: str, fingerprint: str, value: str, mode: str = "default") -> None:
        bucket = self.get_bucket(direction, fingerprint, mode)
        bucket[key] = value
        self.record_access(key, direction, fingerprint, mode)

    def delete(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> bool:
        bucket = self.get_bucket(direction, fingerprint, mode)
        if key in bucket:
            del bucket[key]
            compound_key = f"{mode}|{direction}|{fingerprint}|{key}"
            if "_timestamps" in self._data and compound_key in self._data["_timestamps"]:
                del self._data["_timestamps"][compound_key]
            return True
        return False

    def record_access(
        self,
        key: str,
        direction: str,
        fingerprint: str,
        mode: str = "default",
        timestamp: Optional[float] = None
    ) -> None:
        compound_key = f"{mode}|{direction}|{fingerprint}|{key}"
        timestamps = self._data.setdefault("_timestamps", {})
        timestamps[compound_key] = timestamp if timestamp is not None else time.time()

    def prune(self, now: Optional[float] = None) -> int:
        if now is None:
            now = time.time()
        cutoff = now - (self.ttl_days * 86400)
        timestamps = self._data.setdefault("_timestamps", {})

        # Populate timestamps for untracked entries
        for mode_key in list(self._data.keys()):
            if mode_key.startswith("_"):
                continue
            mode_dict = self._data.get(mode_key)
            if not isinstance(mode_dict, dict):
                continue
            for dir_key, dir_dict in mode_dict.items():
                if not isinstance(dir_dict, dict):
                    continue
                for fp_key, fp_dict in dir_dict.items():
                    if not isinstance(fp_dict, dict):
                        continue
                    for k in list(fp_dict.keys()):
                        ck = f"{mode_key}|{dir_key}|{fp_key}|{k}"
                        if ck not in timestamps:
                            timestamps[ck] = now

        stale_keys = [k for k, ts in timestamps.items() if ts < cutoff]
        pruned_count = 0
        for ck in stale_keys:
            timestamps.pop(ck, None)
            parts = ck.split("|", 3)
            if len(parts) == 4:
                m, d, fp, k = parts
                bucket = self._data.get(m, {}).get(d, {}).get(fp, {})
                if isinstance(bucket, dict) and k in bucket:
                    bucket.pop(k, None)
                    pruned_count += 1

        # Clean up empty branches
        for mode_key in list(self._data.keys()):
            if mode_key.startswith("_"):
                continue
            mode_dict = self._data.get(mode_key)
            if isinstance(mode_dict, dict):
                for dir_key in list(mode_dict.keys()):
                    dir_dict = mode_dict.get(dir_key)
                    if isinstance(dir_dict, dict):
                        for fp_key in list(dir_dict.keys()):
                            fp_dict = dir_dict.get(fp_key)
                            if isinstance(fp_dict, dict) and not fp_dict:
                                dir_dict.pop(fp_key, None)
                        if not dir_dict:
                            mode_dict.pop(dir_key, None)
                if not mode_dict:
                    self._data.pop(mode_key, None)

        return pruned_count

    def save(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
        temp_file = None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            fd, temp_file = tempfile.mkstemp(prefix=".translation_cache_", suffix=".tmp", dir=cache_dir)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(self._data, f, ensure_ascii=False, indent=2)
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

    def clear(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        self._data = {}
        self.save(log_cb=log_cb)


class NullCache(TranslationCache):
    """No-op cache for privacy-sensitive environments or zero-persistence test runs."""

    def get(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> Optional[str]:
        return None

    def put(self, key: str, direction: str, fingerprint: str, value: str, mode: str = "default") -> None:
        pass

    def delete(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> bool:
        return False

    def save(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        pass

    def clear(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        pass

    def load(self, direction: Optional[str] = None) -> None:
        pass

    def record_access(
        self,
        key: str,
        direction: str,
        fingerprint: str,
        mode: str = "default",
        timestamp: Optional[float] = None
    ) -> None:
        pass

    def prune(self, now: Optional[float] = None) -> int:
        return 0

    def get_bucket(self, direction: str, fingerprint: str, mode: str = "default") -> Dict[str, str]:
        return {}
