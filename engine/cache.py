"""
engine/cache.py
===============
Modular translation caching subsystem.
Provides abstract TranslationCache interface, JSONFileCache with atomic writes and TTL,
EncryptedFileCache for encrypted-at-rest persistence, and NullCache for ephemeral sessions.
"""

from abc import ABC, abstractmethod
import os
import sys
import json
import time
import uuid
import base64
import getpass
import platform
import logging
import tempfile
from enum import Enum
from typing import Optional, Dict, Any, Callable, Union

try:
    from cryptography.fernet import Fernet, InvalidToken
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    from cryptography.hazmat.primitives import hashes
    HAS_CRYPTOGRAPHY = True
except ImportError:
    HAS_CRYPTOGRAPHY = False
    Fernet = None
    InvalidToken = Exception
    PBKDF2HMAC = None
    hashes = None

CACHE_TTL_DAYS = 30
CACHE_SALT_DEFAULT = b"offline-doc-translator-cache-salt-v1"


class CachePolicy(str, Enum):
    """Explicit cache storage policy."""
    ENCRYPTED_PERSISTENT = "encrypted_persistent"
    MEMORY_ONLY = "memory_only"
    PLAINTEXT_PERSISTENT = "plaintext_persistent"


def derive_machine_key(salt: Optional[bytes] = None) -> bytes:
    """
    Derives a deterministic, machine- and user-bound 32-byte urlsafe base64 Fernet key.
    Combines Windows MachineGuid (if on Windows), hardware node UUID, platform hostname,
    and current OS username, then hashes via PBKDF2HMAC (SHA-256, 100,000 iterations).
    """
    if not HAS_CRYPTOGRAPHY or PBKDF2HMAC is None or hashes is None:
        raise RuntimeError("The 'cryptography' library is required to derive a Fernet key.")

    parts = []
    if sys.platform == "win32":
        try:
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\Cryptography") as rk:
                guid, _ = winreg.QueryValueEx(rk, "MachineGuid")
                parts.append(str(guid))
        except Exception:
            pass

    try:
        parts.append(str(uuid.getnode()))
    except Exception:
        pass

    parts.append(platform.node() or os.environ.get("COMPUTERNAME", ""))
    parts.append(getpass.getuser() or os.environ.get("USERNAME", ""))

    material = ":".join(parts).encode("utf-8")
    salt = salt or CACHE_SALT_DEFAULT

    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(material))


def derive_fernet_key(
    key_material: Optional[Union[str, bytes]] = None,
    salt: Optional[bytes] = None,
) -> bytes:
    """
    Normalizes or derives a valid 32-byte urlsafe base64-encoded Fernet key.
    - If key_material is None, calls derive_machine_key(salt).
    - If key_material is already a 44-byte urlsafe base64 string/bytes decodable to 32 bytes, returns it as bytes.
    - If key_material is raw 32 bytes, returns base64.urlsafe_b64encode(key_material).
    - Otherwise, treats key_material as a passphrase and derives a 32-byte key via PBKDF2HMAC.
    """
    if not HAS_CRYPTOGRAPHY:
        raise RuntimeError("The 'cryptography' library is required to derive a Fernet key.")

    if key_material is None:
        return derive_machine_key(salt=salt)

    if isinstance(key_material, str):
        key_material = key_material.encode("utf-8")

    # If it's already a 44-byte base64url Fernet key
    if len(key_material) == 44:
        try:
            decoded = base64.urlsafe_b64decode(key_material)
            if len(decoded) == 32:
                return key_material
        except Exception:
            pass

    # If it's 32 raw bytes
    if len(key_material) == 32:
        return base64.urlsafe_b64encode(key_material)

    salt = salt or CACHE_SALT_DEFAULT
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100_000,
    )
    return base64.urlsafe_b64encode(kdf.derive(key_material))


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

    def __init__(
        self,
        cache_file: str = "translation_cache.json",
        ttl_days: int = CACHE_TTL_DAYS,
        legacy_candidates: Optional[List[str]] = None,
    ):
        self.cache_file = cache_file
        self.ttl_days = ttl_days
        self.legacy_candidates = list(legacy_candidates) if legacy_candidates else []
        self._migrated_from_legacy: Optional[str] = None
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

    def load(self, direction: Optional[str] = None, legacy_candidates: Optional[List[str]] = None) -> None:
        """Loads cache file from disk, automatically triggering once-per-day TTL prune."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._data = json.load(f)
            except Exception:
                self._data = {}
        else:
            candidates: List[str] = []
            if legacy_candidates:
                candidates.extend(legacy_candidates)
            candidates.extend(self.legacy_candidates)
            migrated = False
            for legacy_path in candidates:
                if (
                    legacy_path
                    and os.path.exists(legacy_path)
                    and os.path.isfile(legacy_path)
                    and os.path.abspath(legacy_path) != os.path.abspath(self.cache_file)
                ):
                    try:
                        with open(legacy_path, "r", encoding="utf-8") as f:
                            self._data = json.load(f)
                        self._migrated_from_legacy = legacy_path
                        migrated = True
                        break
                    except Exception:
                        pass
            if not migrated:
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
            temp_file = None

            # If this save was preceded by a legacy cache migration, safely backup the legacy file
            if self._migrated_from_legacy and os.path.exists(self._migrated_from_legacy):
                if os.path.abspath(self._migrated_from_legacy) != os.path.abspath(self.cache_file):
                    try:
                        backup_path = f"{self._migrated_from_legacy}.bak.{int(time.time())}"
                        os.replace(self._migrated_from_legacy, backup_path)
                        if log_cb:
                            log_cb(f"Migrated legacy cache to target cache and backed up original to {os.path.basename(backup_path)}")
                        self._migrated_from_legacy = None
                    except Exception as bak_err:
                        logging.warning("Could not backup legacy cache file '%s': %s", self._migrated_from_legacy, bak_err)
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


class EncryptedFileCache(JSONFileCache):
    """
    Encrypted file-backed persistent translation cache.
    Uses cryptography.fernet with a machine-derived or user-specified key to encrypt
    the cache at rest. Transparently handles atomic writes, TTL pruning, and falls back
    to plaintext JSON if cryptography is unavailable or when reading legacy cache files.
    """

    def __init__(
        self,
        cache_file: str = "translation_cache.enc",
        key: Optional[Union[str, bytes]] = None,
        ttl_days: int = CACHE_TTL_DAYS,
        fallback_to_plain: bool = False,
        legacy_candidates: Optional[List[str]] = None,
    ):
        super().__init__(cache_file=cache_file, ttl_days=ttl_days, legacy_candidates=legacy_candidates)
        self.key = key
        self.fallback_to_plain = fallback_to_plain
        self.fernet: Optional[Any] = None
        self._crypto_available: bool = HAS_CRYPTOGRAPHY
        self._migrated_from_legacy: Optional[str] = None

        if HAS_CRYPTOGRAPHY:
            try:
                key_material = key if key is not None else os.environ.get("TRANSLATION_CACHE_KEY")
                derived_key = derive_fernet_key(key_material)
                self.fernet = Fernet(derived_key)
            except Exception as e:
                if not fallback_to_plain:
                    raise
                logging.warning("[!] Failed to initialize cache encryption (%s); falling back to plaintext.", e)
                self.fernet = None
        else:
            if not fallback_to_plain:
                raise RuntimeError("The 'cryptography' library is required for EncryptedFileCache but is not installed.")
            logging.warning("[!] 'cryptography' library is not available; EncryptedFileCache operating in plaintext fallback mode.")

    def load(self, direction: Optional[str] = None, legacy_candidates: Optional[List[str]] = None) -> None:
        """Loads and decrypts cache file from disk, automatically triggering once-per-day TTL prune."""
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "rb") as f:
                    raw_bytes = f.read()

                if not raw_bytes:
                    self._data = {}
                elif self.fernet is not None:
                    stripped = raw_bytes.lstrip()
                    # Backward compatibility: if the file is plaintext JSON, load it directly
                    if stripped.startswith(b"{") or stripped.startswith(b"["):
                        self._data = json.loads(raw_bytes.decode("utf-8"))
                        self._migrated_from_legacy = self.cache_file
                    else:
                        try:
                            decrypted = self.fernet.decrypt(raw_bytes)
                            self._data = json.loads(decrypted.decode("utf-8"))
                        except (InvalidToken, Exception) as decrypt_err:
                            logging.warning(
                                "[!] Warning: Failed to decrypt cache file '%s' (%s). Initializing fresh cache.",
                                self.cache_file,
                                decrypt_err,
                            )
                            self._data = {}
                else:
                    self._data = json.loads(raw_bytes.decode("utf-8"))
            except Exception as e:
                logging.warning("[!] Warning: Failed to load cache file '%s': %s", self.cache_file, e)
                self._data = {}
        else:
            # Check for legacy plaintext or document-adjacent cache files for migration
            all_candidates: List[str] = []
            if legacy_candidates:
                all_candidates.extend(legacy_candidates)
            all_candidates.extend(self.legacy_candidates)

            base, ext = os.path.splitext(self.cache_file)
            if ext != ".json":
                all_candidates.append(base + ".json")
            cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
            all_candidates.append(os.path.join(cache_dir, ".translation_cache.json"))
            all_candidates.append(os.path.join(cache_dir, "translation_cache.json"))
            all_candidates.append(os.path.join(cache_dir, ".translation_cache.enc"))
            all_candidates.append(os.path.join(cache_dir, "translation_cache.enc"))

            seen = set()
            unique_candidates: List[str] = []
            for c in all_candidates:
                if not c:
                    continue
                norm = os.path.abspath(c)
                if norm != os.path.abspath(self.cache_file) and norm not in seen:
                    seen.add(norm)
                    unique_candidates.append(norm)

            migrated = False
            for legacy_path in unique_candidates:
                if os.path.exists(legacy_path) and os.path.isfile(legacy_path):
                    try:
                        with open(legacy_path, "rb") as f:
                            raw_bytes = f.read()
                        if not raw_bytes:
                            continue
                        stripped = raw_bytes.lstrip()
                        if stripped.startswith(b"{") or stripped.startswith(b"["):
                            self._data = json.loads(raw_bytes.decode("utf-8"))
                        elif self.fernet is not None:
                            try:
                                decrypted = self.fernet.decrypt(raw_bytes)
                                self._data = json.loads(decrypted.decode("utf-8"))
                            except Exception:
                                continue
                        else:
                            continue

                        self._migrated_from_legacy = legacy_path
                        logging.info("Discovered legacy cache '%s'; queued for migration to encrypted format.", legacy_path)
                        migrated = True
                        break
                    except Exception as e:
                        logging.warning("Failed to read legacy cache candidate '%s': %s", legacy_path, e)

            if not migrated:
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

    def save(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Atomically persists encrypted cache payload to disk."""
        cache_dir = os.path.dirname(os.path.abspath(self.cache_file))
        temp_file = None
        try:
            os.makedirs(cache_dir, exist_ok=True)
            fd, temp_file = tempfile.mkstemp(prefix=".translation_cache_enc_", suffix=".tmp", dir=cache_dir)
            plaintext = json.dumps(self._data, ensure_ascii=False, indent=2).encode("utf-8")
            if self.fernet is not None:
                payload = self.fernet.encrypt(plaintext)
            else:
                payload = plaintext

            with os.fdopen(fd, "wb") as f:
                f.write(payload)
            os.replace(temp_file, self.cache_file)
            temp_file = None

            # If this save was preceded by a legacy cache migration, safely backup the legacy file
            if self._migrated_from_legacy and os.path.exists(self._migrated_from_legacy):
                if os.path.abspath(self._migrated_from_legacy) != os.path.abspath(self.cache_file):
                    try:
                        backup_path = f"{self._migrated_from_legacy}.bak.{int(time.time())}"
                        os.replace(self._migrated_from_legacy, backup_path)
                        if log_cb:
                            log_cb(f"Migrated legacy cache to encrypted cache and backed up original to {os.path.basename(backup_path)}")
                        self._migrated_from_legacy = None
                    except Exception as bak_err:
                        logging.warning("Could not backup legacy cache file '%s': %s", self._migrated_from_legacy, bak_err)
        except Exception as e:
            if log_cb:
                log_cb(f"[!] Warning: Failed to save translation cache: {e}")
            logging.warning("[!] Warning: Failed to save translation cache: %s", e)
        finally:
            if temp_file and os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
