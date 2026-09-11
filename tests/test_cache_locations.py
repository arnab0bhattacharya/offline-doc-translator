"""
tests/test_cache_locations.py
==============================
Unit tests for centralized application cache storage, path derivation,
accurate cache statistics, safe clearing, and legacy migration.
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

from engine.cache import CachePolicy, EncryptedFileCache
from engine.cache_locations import (
    cleanup_legacy_cache_remnants,
    clear_all_caches,
    get_cache_root_dir,
    get_cache_stats,
    get_job_cache_path,
    is_owned_cache_file,
)
from engine.core import TranslationMode


class TestCacheLocations(unittest.TestCase):
    def setUp(self):
        self.temp_root = tempfile.mkdtemp(prefix="test_cache_loc_")
        self.doc_dir = os.path.join(self.temp_root, "docs")
        self.cache_dir = os.path.join(self.temp_root, "app_cache")
        os.makedirs(self.doc_dir, exist_ok=True)
        os.makedirs(self.cache_dir, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.temp_root, ignore_errors=True)

    def test_get_cache_root_dir_env_override(self):
        with patch.dict(os.environ, {"TRANSLATION_CACHE_DIR": self.cache_dir}):
            root = get_cache_root_dir()
            self.assertEqual(os.path.abspath(root), os.path.abspath(self.cache_dir))
            self.assertTrue(os.path.isdir(root))

    def test_get_cache_root_dir_default(self):
        with patch.dict(os.environ, {}, clear=False):
            if "TRANSLATION_CACHE_DIR" in os.environ:
                del os.environ["TRANSLATION_CACHE_DIR"]
            root = get_cache_root_dir()
            self.assertTrue(os.path.isdir(root))
            self.assertIn("OfflineDocumentTranslator", root)

    def test_get_job_cache_path_properties(self):
        with patch.dict(os.environ, {"TRANSLATION_CACHE_DIR": self.cache_dir}):
            doc_out_1 = os.path.join(self.doc_dir, "report_en.docx")
            doc_out_2 = os.path.join(self.doc_dir, "presentation_en.pptx")

            # Same directory and same mode produces identical cache store path
            path1 = get_job_cache_path(doc_out_1, TranslationMode.FAST_NMT)
            path2 = get_job_cache_path(doc_out_2, TranslationMode.FAST_NMT)
            self.assertEqual(path1, path2)

            # Sibling directory produces different cache store path
            other_dir = os.path.join(self.temp_root, "other_docs")
            os.makedirs(other_dir, exist_ok=True)
            doc_other = os.path.join(other_dir, "report_en.docx")
            path_other = get_job_cache_path(doc_other, TranslationMode.FAST_NMT)
            self.assertNotEqual(path1, path_other)

            # Different mode produces different cache store path
            path_llm = get_job_cache_path(doc_out_1, TranslationMode.PURE_LLM)
            self.assertNotEqual(path1, path_llm)

            # Encrypted persistent vs plaintext persistent extensions
            path_enc = get_job_cache_path(doc_out_1, cache_policy=CachePolicy.ENCRYPTED_PERSISTENT)
            path_plain = get_job_cache_path(doc_out_1, cache_policy=CachePolicy.PLAINTEXT_PERSISTENT)
            self.assertTrue(path_enc.endswith(".enc"))
            self.assertTrue(path_plain.endswith(".json"))

            # Sensitive filenames are not leaked in cache filename
            filename = os.path.basename(path_enc)
            self.assertNotIn("report_en", filename)
            self.assertTrue(filename.startswith("cache_fast_nmt_"))

    def test_is_owned_cache_file(self):
        self.assertTrue(is_owned_cache_file("cache_fast_nmt_abcdef0123456789.enc"))
        self.assertTrue(is_owned_cache_file("cache_pure_llm_abcdef0123456789.json"))
        self.assertTrue(is_owned_cache_file(".translation_cache_enc_xyz.tmp"))
        self.assertTrue(is_owned_cache_file(".translation_cache.json"))
        self.assertTrue(is_owned_cache_file("cache_fast_nmt_abc.json.bak.123456"))

        # Non-cache files should be rejected
        self.assertFalse(is_owned_cache_file("report.docx"))
        self.assertFalse(is_owned_cache_file("system.log"))
        self.assertFalse(is_owned_cache_file("some_other_data.json"))
        self.assertFalse(is_owned_cache_file("key.enc"))

    def test_get_cache_stats_and_clear(self):
        with patch.dict(os.environ, {"TRANSLATION_CACHE_DIR": self.cache_dir}):
            # Initially empty
            stats0 = get_cache_stats()
            self.assertEqual(stats0["file_count"], 0)
            self.assertEqual(stats0["total_bytes"], 0)

            # Add two owned cache files and one non-cache file
            f1 = os.path.join(self.cache_dir, "cache_fast_nmt_0001.enc")
            f2 = os.path.join(self.cache_dir, "cache_pure_llm_0002.json")
            unrelated = os.path.join(self.cache_dir, "unrelated_user_file.txt")

            with open(f1, "wb") as f:
                f.write(b"12345678")  # 8 bytes
            with open(f2, "wb") as f:
                f.write(b"1234")  # 4 bytes
            with open(unrelated, "wb") as f:
                f.write(b"do not delete me")

            stats1 = get_cache_stats()
            self.assertEqual(stats1["file_count"], 2)
            self.assertEqual(stats1["total_bytes"], 12)
            self.assertIn("cache_fast_nmt_0001.enc", stats1["files"])
            self.assertIn("cache_pure_llm_0002.json", stats1["files"])

            # Clear cache
            res = clear_all_caches()
            self.assertEqual(res["deleted"], 2)
            self.assertEqual(res["failed"], 0)
            self.assertEqual(res["freed_bytes"], 12)

            # Non-owned file preserved
            self.assertTrue(os.path.exists(unrelated))
            self.assertFalse(os.path.exists(f1))
            self.assertFalse(os.path.exists(f2))

            # Stats after clearing
            stats2 = get_cache_stats()
            self.assertEqual(stats2["file_count"], 0)
            self.assertEqual(stats2["total_bytes"], 0)

    def test_legacy_candidate_migration_end_to_end(self):
        with patch.dict(os.environ, {"TRANSLATION_CACHE_DIR": self.cache_dir}):
            # Setup a legacy cache in the document directory
            legacy_file = os.path.join(self.doc_dir, ".translation_cache.json")
            sample_data = {"fast_nmt": {"ja2en": {"fp123": {"こんにちは": "Hello"}}}}
            with open(legacy_file, "w", encoding="utf-8") as f:
                json.dump(sample_data, f)

            doc_path = os.path.join(self.doc_dir, "output.docx")
            target_cache_file = get_job_cache_path(doc_path, TranslationMode.FAST_NMT)

            # Central cache does not exist yet
            self.assertFalse(os.path.exists(target_cache_file))

            # Create EncryptedFileCache with legacy_candidates
            cache = EncryptedFileCache(
                cache_file=target_cache_file,
                legacy_candidates=[legacy_file],
            )
            cache.load()

            # Verify data loaded from legacy cache
            val = cache.get("こんにちは", "ja2en", "fp123", mode="fast_nmt")
            self.assertEqual(val, "Hello")

            # Persist to disk
            cache.save()

            # Target encrypted cache was written
            self.assertTrue(os.path.exists(target_cache_file))

            # Old legacy file was deleted, and NO .bak or plaintext file remains in the document directory
            self.assertFalse(os.path.exists(legacy_file))
            remaining_doc_files = [
                fn
                for fn in os.listdir(self.doc_dir)
                if ".translation_cache" in fn or "translation_cache" in fn or ".bak" in fn
            ]
            self.assertEqual(remaining_doc_files, [])

            # Re-loading without legacy candidates uses the newly migrated central cache
            fresh_cache = EncryptedFileCache(cache_file=target_cache_file)
            fresh_cache.load()
            self.assertEqual(fresh_cache.get("こんにちは", "ja2en", "fp123", mode="fast_nmt"), "Hello")

    def test_cleanup_legacy_cache_remnants(self):
        """cleanup_legacy_cache_remnants purges all legacy cache variants and .bak remnants."""
        files_to_create = [
            "translation_cache.json",
            ".translation_cache.json",
            "translation_cache.enc",
            ".translation_cache.enc",
            "translation_cache.json.bak.12345",
            ".translation_cache.json.bak.67890",
            "unrelated_document.docx",
        ]
        for f in files_to_create:
            with open(os.path.join(self.doc_dir, f), "w", encoding="utf-8") as fh:
                fh.write("dummy")

        deleted_count = cleanup_legacy_cache_remnants(self.doc_dir)
        self.assertEqual(deleted_count, 6)
        remaining = os.listdir(self.doc_dir)
        self.assertEqual(remaining, ["unrelated_document.docx"])


if __name__ == "__main__":
    unittest.main()
