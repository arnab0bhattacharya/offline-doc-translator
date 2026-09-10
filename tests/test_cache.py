"""
tests/test_cache.py
===================
Unit tests for the modular TranslationCache subsystem:
- TranslationCache abstract base class
- JSONFileCache with atomic storage and TTL pruning
- NullCache no-op implementation
- TranslationEngine integration and pluggability
"""

import os
import json
import time
import tempfile
import unittest
from unittest.mock import patch

from engine.cache import (
    TranslationCache,
    JSONFileCache,
    NullCache,
    EncryptedFileCache,
    CachePolicy,
    derive_machine_key,
    derive_fernet_key,
    CACHE_TTL_DAYS,
    HAS_CRYPTOGRAPHY,
)
from engine.core import TranslationEngine, TranslationMode, TranslationResult


class CustomMemoryCache(TranslationCache):
    """Custom in-memory cache implementation for testing pluggability."""

    def __init__(self):
        self.store = {}
        self.save_calls = 0

    def get(self, key: str, direction: str, fingerprint: str, mode: str = "default"):
        return self.store.get((mode, direction, fingerprint, key))

    def put(self, key: str, direction: str, fingerprint: str, value: str, mode: str = "default"):
        self.store[(mode, direction, fingerprint, key)] = value

    def delete(self, key: str, direction: str, fingerprint: str, mode: str = "default") -> bool:
        k = (mode, direction, fingerprint, key)
        if k in self.store:
            del self.store[k]
            return True
        return False

    def save(self, log_cb=None):
        self.save_calls += 1

    def clear(self, log_cb=None):
        self.store.clear()

    def load(self, direction=None):
        pass

    def record_access(self, key: str, direction: str, fingerprint: str, mode: str = "default", timestamp=None):
        pass

    def prune(self, now=None) -> int:
        return 0


class TestTranslationCacheInterface(unittest.TestCase):
    """Verifies that TranslationCache enforces abstract method contracts."""

    def test_abstract_class_cannot_be_instantiated(self):
        with self.assertRaises(TypeError):
            TranslationCache()


class TestJSONFileCache(unittest.TestCase):
    """Verifies JSONFileCache operations, atomic persistence, and TTL lifecycle."""

    def test_put_get_delete(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            cache = JSONFileCache(cache_file=cache_file)

            # Initially absent
            self.assertIsNone(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"))

            # Put and retrieve
            cache.put("h1", "ja2en", "fp1", "Translated Text", mode="fast_nmt")
            self.assertEqual(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"), "Translated Text")

            # Delete
            deleted = cache.delete("h1", "ja2en", "fp1", mode="fast_nmt")
            self.assertTrue(deleted)
            self.assertIsNone(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"))

            # Delete absent
            self.assertFalse(cache.delete("h1", "ja2en", "fp1", mode="fast_nmt"))

    def test_save_and_reload(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            cache1 = JSONFileCache(cache_file=cache_file)
            cache1.put("k1", "en2ja", "fpA", "翻訳1", mode="pure_llm")
            cache1.save()

            self.assertTrue(os.path.exists(cache_file))

            # Reload in new instance
            cache2 = JSONFileCache(cache_file=cache_file)
            cache2.load("en2ja")
            self.assertEqual(cache2.get("k1", "en2ja", "fpA", mode="pure_llm"), "翻訳1")

    def test_ttl_pruning_removes_expired_entries(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            cache = JSONFileCache(cache_file=cache_file, ttl_days=10)

            now = time.time()
            old_time = now - (15 * 86400)   # 15 days old (> 10 days)
            fresh_time = now - (2 * 86400)  # 2 days old (< 10 days)

            cache.put("old_key", "ja2en", "fp1", "Old Translation", mode="fast_nmt")
            cache.record_access("old_key", "ja2en", "fp1", mode="fast_nmt", timestamp=old_time)

            cache.put("new_key", "ja2en", "fp1", "New Translation", mode="fast_nmt")
            cache.record_access("new_key", "ja2en", "fp1", mode="fast_nmt", timestamp=fresh_time)

            pruned = cache.prune(now=now)
            self.assertEqual(pruned, 1)
            self.assertIsNone(cache.get("old_key", "ja2en", "fp1", mode="fast_nmt"))
            self.assertEqual(cache.get("new_key", "ja2en", "fp1", mode="fast_nmt"), "New Translation")

    def test_clear_empties_storage_and_file(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            cache = JSONFileCache(cache_file=cache_file)
            cache.put("k1", "ja2en", "fp1", "Text", mode="fast_nmt")
            cache.save()

            cache.clear()
            self.assertEqual(cache.data, {})

            # Reload
            cache_reloaded = JSONFileCache(cache_file=cache_file)
            cache_reloaded.load()
            self.assertIsNone(cache_reloaded.get("k1", "ja2en", "fp1", mode="fast_nmt"))


class TestNullCache(unittest.TestCase):
    """Verifies NullCache provides zero persistence with no file emissions."""

    def test_null_cache_operations(self):
        cache = NullCache()
        cache.put("k1", "ja2en", "fp1", "Val", mode="fast_nmt")
        self.assertIsNone(cache.get("k1", "ja2en", "fp1", mode="fast_nmt"))
        self.assertFalse(cache.delete("k1", "ja2en", "fp1", mode="fast_nmt"))
        self.assertEqual(cache.prune(), 0)
        cache.save()
        cache.clear()
        cache.load()


class TestEngineCacheIntegration(unittest.TestCase):
    """Verifies TranslationEngine correctly interacts with custom TranslationCache implementations."""

    def test_engine_with_null_cache_bypasses_caching(self):
        class MockNMT:
            name = "nmt"
            def __init__(self):
                self.calls = 0
            def is_ready(self, direction):
                return True
            def translate(self, text, direction, **kwargs):
                self.calls += 1
                return f"[NMT {self.calls}: {text}]", 0.01

        backend = MockNMT()
        engine = TranslationEngine(
            mode=TranslationMode.FAST_NMT,
            backend=backend,
            cache=NullCache(),
        )

        res1 = engine.translate_chunk("これはテストです。", "ja2en")
        self.assertEqual(res1.source_backend, "nmt")
        self.assertEqual(backend.calls, 1)

        # Identical chunk with NullCache should re-invoke backend (no cache hit)
        res2 = engine.translate_chunk("これはテストです。", "ja2en")
        self.assertEqual(res2.source_backend, "nmt")
        self.assertEqual(backend.calls, 2)

    def test_engine_with_custom_cache_implementation(self):
        class MockNMT:
            name = "nmt"
            def is_ready(self, direction):
                return True
            def translate(self, text, direction, **kwargs):
                return f"[CUSTOM_NMT: {text}]", 0.01

        custom_cache = CustomMemoryCache()
        engine = TranslationEngine(
            mode=TranslationMode.FAST_NMT,
            backend=MockNMT(),
            cache=custom_cache,
        )

        # 1. First translation -> stored in custom cache
        res1 = engine.translate_chunk("これはテストです。", "ja2en")
        self.assertEqual(res1.source_backend, "nmt")
        self.assertEqual(len(custom_cache.store), 1)

        # 2. Second translation -> served from custom cache
        res2 = engine.translate_chunk("これはテストです。", "ja2en")
        self.assertEqual(res2.source_backend, "cache")
        self.assertEqual(res2.text, "[CUSTOM_NMT: これはテストです。]")

        # 3. Save cache forwards to custom cache save()
        engine.save_cache_atomically()
        self.assertEqual(custom_cache.save_calls, 1)


class TestKeyDerivation(unittest.TestCase):
    """Verifies machine-bound and explicit Fernet key derivation routines."""

    def test_derive_machine_key(self):
        key1 = derive_machine_key()
        key2 = derive_machine_key()
        self.assertEqual(key1, key2)
        self.assertEqual(len(key1), 44)

        # Different salt produces different key
        key3 = derive_machine_key(salt=b"custom-salt-12345678")
        self.assertNotEqual(key1, key3)
        self.assertEqual(len(key3), 44)

    def test_derive_fernet_key_with_explicit_keys(self):
        from cryptography.fernet import Fernet
        # 1. Native 44-char urlsafe base64 Fernet key
        gen_key = Fernet.generate_key()
        derived = derive_fernet_key(gen_key)
        self.assertEqual(derived, gen_key)

        # 2. Raw 32 bytes
        raw_32 = b"12345678901234567890123456789012"
        derived_raw = derive_fernet_key(raw_32)
        self.assertEqual(len(derived_raw), 44)
        Fernet(derived_raw)

        # 3. Arbitrary passphrase string
        passphrase = "my-secret-vault-passphrase"
        derived_pass = derive_fernet_key(passphrase)
        self.assertEqual(len(derived_pass), 44)
        Fernet(derived_pass)

        # 4. None falls back to machine key
        derived_none = derive_fernet_key(None)
        self.assertEqual(derived_none, derive_machine_key())


class TestEncryptedFileCache(unittest.TestCase):
    """Verifies EncryptedFileCache CRUD, atomic encryption at rest, and resilience."""

    def test_put_get_delete(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache = EncryptedFileCache(cache_file=cache_file)

            self.assertIsNone(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"))
            cache.put("h1", "ja2en", "fp1", "Secret Translated Text", mode="fast_nmt")
            self.assertEqual(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"), "Secret Translated Text")

            deleted = cache.delete("h1", "ja2en", "fp1", mode="fast_nmt")
            self.assertTrue(deleted)
            self.assertIsNone(cache.get("h1", "ja2en", "fp1", mode="fast_nmt"))

    def test_ciphertext_on_disk_is_encrypted_not_plaintext(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache = EncryptedFileCache(cache_file=cache_file)
            cache.put("secret_k", "ja2en", "fp1", "TOP SECRET TRANSLATION", mode="fast_nmt")
            cache.save()

            self.assertTrue(os.path.exists(cache_file))
            with open(cache_file, "rb") as f:
                raw_bytes = f.read()

            # Verify it is Fernet ciphertext (starts with standard Fernet version header gAAAAA)
            self.assertTrue(raw_bytes.startswith(b"gAAAAA"))
            self.assertNotIn(b"TOP SECRET TRANSLATION", raw_bytes)
            self.assertNotIn(b"secret_k", raw_bytes)

            # Plain JSON parser must fail
            with self.assertRaises(json.JSONDecodeError):
                json.loads(raw_bytes.decode("utf-8"))

    def test_save_and_reload_with_explicit_key(self):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache1 = EncryptedFileCache(cache_file=cache_file, key=key)
            cache1.put("k1", "en2ja", "fpA", "秘密の翻訳", mode="pure_llm")
            cache1.save()

            # Reload with same key
            cache2 = EncryptedFileCache(cache_file=cache_file, key=key)
            cache2.load("en2ja")
            self.assertEqual(cache2.get("k1", "en2ja", "fpA", mode="pure_llm"), "秘密の翻訳")

    def test_save_and_reload_with_machine_derived_key(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache1 = EncryptedFileCache(cache_file=cache_file, key=None)
            cache1.put("k_auto", "ja2en", "fpB", "Auto Key Text", mode="fast_nmt")
            cache1.save()

            cache2 = EncryptedFileCache(cache_file=cache_file, key=None)
            cache2.load("ja2en")
            self.assertEqual(cache2.get("k_auto", "ja2en", "fpB", mode="fast_nmt"), "Auto Key Text")

    def test_reload_with_wrong_key_fails_gracefully(self):
        from cryptography.fernet import Fernet
        key1 = Fernet.generate_key()
        key2 = Fernet.generate_key()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache1 = EncryptedFileCache(cache_file=cache_file, key=key1)
            cache1.put("k1", "ja2en", "fp", "Data", mode="fast_nmt")
            cache1.save()

            # Attempt loading with key2: must not raise, should reset to empty cache
            cache2 = EncryptedFileCache(cache_file=cache_file, key=key2)
            cache2.load()
            self.assertEqual([k for k in cache2.data if not k.startswith("_")], [])
            self.assertIsNone(cache2.get("k1", "ja2en", "fp", mode="fast_nmt"))

    def test_corrupt_file_recovery(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            with open(cache_file, "wb") as f:
                f.write(b"NOT_A_VALID_FERNET_TOKEN_GARBAGE_BYTES")

            cache = EncryptedFileCache(cache_file=cache_file)
            cache.load()
            self.assertEqual([k for k in cache.data if not k.startswith("_")], [])
            self.assertIsNone(cache.get("any_key", "ja2en", "fp", mode="fast_nmt"))

    def test_plaintext_json_backward_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            # Write a plain JSON cache file
            plain_data = {
                "_meta": {"last_prune": 0},
                "fast_nmt": {
                    "ja2en": {
                        "fp_legacy": {
                            "legacy_key": "Plain Legacy Translation"
                        }
                    }
                }
            }
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(plain_data, f)

            # EncryptedFileCache should transparently load plain JSON
            cache = EncryptedFileCache(cache_file=cache_file)
            cache.load()
            self.assertEqual(cache.get("legacy_key", "ja2en", "fp_legacy", mode="fast_nmt"), "Plain Legacy Translation")

            # Subsequent save re-encrypts the file
            cache.save()
            with open(cache_file, "rb") as f:
                saved_bytes = f.read()
            self.assertTrue(saved_bytes.startswith(b"gAAAAA"))

    def test_clear_writes_encrypted_empty_cache(self):
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.enc")
            cache = EncryptedFileCache(cache_file=cache_file)
            cache.put("k1", "ja2en", "fp", "Data", mode="fast_nmt")
            cache.save()

            cache.clear()
            self.assertEqual(cache.data, {})

            # Disk file is valid encrypted token containing empty dict
            with open(cache_file, "rb") as f:
                saved_bytes = f.read()
            self.assertTrue(saved_bytes.startswith(b"gAAAAA"))

            # Reload to verify
            cache_reloaded = EncryptedFileCache(cache_file=cache_file)
            cache_reloaded.load()
            self.assertEqual([k for k in cache_reloaded.data if not k.startswith("_")], [])
            self.assertIsNone(cache_reloaded.get("k1", "ja2en", "fp", mode="fast_nmt"))

    def test_fallback_when_cryptography_missing(self):
        with patch("engine.cache.HAS_CRYPTOGRAPHY", False):
            # 1. With fallback_to_plain=False, raises RuntimeError
            with self.assertRaises(RuntimeError):
                EncryptedFileCache(fallback_to_plain=False)

            # 2. With fallback_to_plain=True, operates in plain mode
            with tempfile.TemporaryDirectory() as td:
                cache_file = os.path.join(td, "plain_fallback.json")
                cache = EncryptedFileCache(cache_file=cache_file, fallback_to_plain=True)
                self.assertIsNone(cache.fernet)
                cache.put("k", "ja2en", "fp", "Plain Fallback", mode="fast_nmt")
                cache.save()

                with open(cache_file, "r", encoding="utf-8") as f:
                    content = json.load(f)
                self.assertEqual(content["fast_nmt"]["ja2en"]["fp"]["k"], "Plain Fallback")


class TestEngineEncryptedCacheIntegration(unittest.TestCase):
    """Verifies TranslationEngine integration with encrypted_cache=True."""

    def test_engine_encrypted_cache_end_to_end(self):
        class MockNMT:
            name = "nmt"
            def __init__(self):
                self.calls = 0
            def is_ready(self, direction):
                return True
            def translate(self, text, direction, **kwargs):
                self.calls += 1
                return f"[NMT:{text}]", 0.01

        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "engine_cache.enc")
            backend = MockNMT()

            engine1 = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                backend=backend,
                encrypted_cache=True,
                cache_file=cache_file,
            )
            self.assertIsInstance(engine1.cache_mgr, EncryptedFileCache)

            # Translate chunk 1
            res1 = engine1.translate_chunk("テスト文章", "ja2en")
            self.assertEqual(res1.source_backend, "nmt")
            self.assertEqual(backend.calls, 1)

            # Translate chunk 1 again (hit memory cache)
            res2 = engine1.translate_chunk("テスト文章", "ja2en")
            self.assertEqual(res2.source_backend, "cache")
            self.assertEqual(backend.calls, 1)

            # Persist to disk
            engine1.save_cache_atomically()

            # Verify file on disk is encrypted
            with open(cache_file, "rb") as f:
                disk_bytes = f.read()
            self.assertTrue(disk_bytes.startswith(b"gAAAAA"))

            # New engine loading the encrypted cache
            backend2 = MockNMT()
            engine2 = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                backend=backend2,
                encrypted_cache=True,
                cache_file=cache_file,
            )
            engine2.cache_mgr.load()

            # Translating the same chunk should now be served from disk-restored cache!
            res3 = engine2.translate_chunk("テスト文章", "ja2en")
            self.assertEqual(res3.source_backend, "cache")
            self.assertEqual(backend2.calls, 0)


class TestCachePolicyAndPrivacyDefaults(unittest.TestCase):
    """Verifies CachePolicy enum, defaults, legacy migration, and privacy guarantees."""

    def test_cache_policy_enum_values(self):
        self.assertEqual(CachePolicy.ENCRYPTED_PERSISTENT.value, "encrypted_persistent")
        self.assertEqual(CachePolicy.MEMORY_ONLY.value, "memory_only")
        self.assertEqual(CachePolicy.PLAINTEXT_PERSISTENT.value, "plaintext_persistent")

    def test_encrypted_file_cache_legacy_json_migration(self):
        with tempfile.TemporaryDirectory() as td:
            legacy_json = os.path.join(td, ".translation_cache.json")
            enc_file = os.path.join(td, ".translation_cache.enc")

            # 1. Create a legacy plaintext JSON cache
            plain_data = {
                "_meta": {"last_prune": 0},
                "fast_nmt": {
                    "ja2en": {
                        "fp_legacy": {
                            "legacy_hash": "Migrated Secret Text"
                        }
                    }
                }
            }
            with open(legacy_json, "w", encoding="utf-8") as f:
                json.dump(plain_data, f)

            # 2. Instantiate EncryptedFileCache targeting .translation_cache.enc
            cache = EncryptedFileCache(cache_file=enc_file)
            cache.load("ja2en")

            # 3. Assert legacy data was imported into memory
            self.assertEqual(cache.get("legacy_hash", "ja2en", "fp_legacy", mode="fast_nmt"), "Migrated Secret Text")

            # 4. Save cache: should write encrypted payload and backup legacy file
            logs = []
            cache.save(log_cb=lambda msg: logs.append(msg))
            self.assertTrue(os.path.exists(enc_file))

            with open(enc_file, "rb") as f:
                enc_bytes = f.read()
            self.assertTrue(enc_bytes.startswith(b"gAAAAA"))
            self.assertNotIn(b"Migrated Secret Text", enc_bytes)

            # Original .translation_cache.json must be deleted, and NO .bak or plaintext file remains
            self.assertFalse(os.path.exists(legacy_json))
            remaining_files = os.listdir(td)
            self.assertEqual(remaining_files, [os.path.basename(enc_file)])

            # 5. Reload into a new cache instance: must load from .enc file directly
            cache2 = EncryptedFileCache(cache_file=enc_file)
            cache2.load("ja2en")
            self.assertEqual(cache2.get("legacy_hash", "ja2en", "fp_legacy", mode="fast_nmt"), "Migrated Secret Text")

    def test_encrypted_file_cache_purges_lingering_legacy_bak_files(self):
        """Migration safely purges existing lingering .bak files in the legacy directory."""
        with tempfile.TemporaryDirectory() as td:
            legacy_json = os.path.join(td, ".translation_cache.json")
            lingering_bak = os.path.join(td, ".translation_cache.json.bak.123456")
            enc_file = os.path.join(td, "cache_fast_nmt_scope.enc")

            with open(legacy_json, "w", encoding="utf-8") as f:
                json.dump({"fast_nmt": {"ja2en": {"fp": {"k": "v"}}}}, f)
            with open(lingering_bak, "w", encoding="utf-8") as f:
                f.write("old backup plaintext")

            cache = EncryptedFileCache(cache_file=enc_file, legacy_candidates=[legacy_json])
            cache.load("ja2en")
            cache.save()

            self.assertTrue(os.path.exists(enc_file))
            self.assertFalse(os.path.exists(legacy_json))
            self.assertFalse(os.path.exists(lingering_bak))
            self.assertEqual(os.listdir(td), [os.path.basename(enc_file)])

    def test_engine_defaults_to_encrypted_cache(self):
        engine = TranslationEngine()
        self.assertEqual(engine.cache_policy, CachePolicy.ENCRYPTED_PERSISTENT)
        self.assertIsInstance(engine.cache_mgr, EncryptedFileCache)

    def test_engine_memory_only_uses_null_cache(self):
        engine = TranslationEngine(cache_policy=CachePolicy.MEMORY_ONLY)
        self.assertEqual(engine.cache_policy, CachePolicy.MEMORY_ONLY)
        self.assertIsInstance(engine.cache_mgr, NullCache)

    def test_engine_plaintext_persistent_uses_json_cache(self):
        engine = TranslationEngine(cache_policy=CachePolicy.PLAINTEXT_PERSISTENT)
        self.assertEqual(engine.cache_policy, CachePolicy.PLAINTEXT_PERSISTENT)
        self.assertIsInstance(engine.cache_mgr, JSONFileCache)

    def test_engine_crypto_unavailable_falls_back_to_null_cache_not_plain(self):
        with patch("engine.cache.HAS_CRYPTOGRAPHY", False):
            engine = TranslationEngine(cache_policy=CachePolicy.ENCRYPTED_PERSISTENT)
            # Must NOT be JSONFileCache (which writes plaintext)
            self.assertNotIsInstance(engine.cache_mgr, JSONFileCache)
            self.assertIsInstance(engine.cache_mgr, NullCache)


if __name__ == "__main__":
    unittest.main()

