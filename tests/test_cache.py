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

from engine.cache import TranslationCache, JSONFileCache, NullCache, CACHE_TTL_DAYS
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


if __name__ == "__main__":
    unittest.main()
