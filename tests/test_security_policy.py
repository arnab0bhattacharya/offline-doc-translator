"""
tests/test_security_policy.py
==============================
Unit tests for disk-aware archive admission, compression ratio limits,
ZipSecurityError semantics, and clean error recovery.
"""

import os
import shutil
import tempfile
import unittest
import zipfile
from collections import namedtuple
from unittest.mock import patch

from engine.errors import ErrorCode, TranslatorError
from engine.security_policy import DEFAULT_POLICY, DocumentSecurityPolicy
from formats.base import BaseFormatHandler, ZipSecurityError


class TestDiskAwareZipAdmission(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_sec_policy_")
        self.extract_dir = os.path.join(self.test_dir, "extracted")
        self.dummy_zip = os.path.join(self.test_dir, "sample.zip")
        with zipfile.ZipFile(self.dummy_zip, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("word/document.xml", b"<w:document>Sample Content</w:document>")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_default_security_policy_extended_fields(self):
        policy = DEFAULT_POLICY
        self.assertEqual(policy.min_free_disk_after_extract_bytes, 500 * 1024 * 1024)
        self.assertEqual(policy.max_compression_ratio, 100.0)
        self.assertEqual(policy.max_archive_metadata_bytes, 10 * 1024 * 1024)

    def test_zip_security_error_type_hierarchy(self):
        err = ZipSecurityError(detail="Test security violation", code=ErrorCode.E04)
        self.assertIsInstance(err, TranslatorError)
        self.assertIsInstance(err, zipfile.BadZipFile)
        self.assertEqual(err.code, ErrorCode.E04)
        self.assertIn("Test security violation", str(err))

    def test_extract_zip_rejects_insufficient_disk_space(self):
        # Mock disk_usage to report only 50 MB free space, whereas policy requires 500 MB reserve
        FakeDiskUsage = namedtuple("usage", ["total", "used", "free"])
        low_disk = FakeDiskUsage(total=1000 * 1024 * 1024, used=950 * 1024 * 1024, free=50 * 1024 * 1024)

        with patch("shutil.disk_usage", return_value=low_disk):
            with self.assertRaises(TranslatorError) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)

            self.assertEqual(ctx.exception.code, ErrorCode.E04)
            self.assertIn("Insufficient disk space", ctx.exception.detail)
            # Rejection must not leave partially extracted directory
            self.assertFalse(os.path.exists(self.extract_dir))

    def test_extract_zip_rejects_excessive_compression_ratio(self):
        # Create a bomb-like member: 100,000 bytes uncompressed in 50 bytes compressed (ratio ~2000:1)
        fake_info = zipfile.ZipInfo("huge.xml")
        fake_info.file_size = 500_000
        fake_info.compress_size = 100  # 5000:1 ratio > 100:1 limit

        with patch.object(zipfile.ZipFile, "infolist", return_value=[fake_info]):
            with self.assertRaises(TranslatorError) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)

            self.assertEqual(ctx.exception.code, ErrorCode.E04)
            self.assertIn("maximum compression ratio", ctx.exception.detail)
            self.assertFalse(os.path.exists(self.extract_dir))

    def test_extract_zip_handles_zero_compress_size_safely(self):
        # Stored entry with 0 compress size (e.g. empty file) must not divide by zero
        fake_info = zipfile.ZipInfo("empty.xml")
        fake_info.file_size = 0
        fake_info.compress_size = 0

        with patch.object(zipfile.ZipFile, "infolist", return_value=[fake_info]):
            with patch.object(zipfile.ZipFile, "extractall"):
                # Should not raise ZeroDivisionError
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)

    def test_extract_zip_corrupt_file_raises_structured_e04(self):
        corrupt_path = os.path.join(self.test_dir, "corrupt.docx")
        with open(corrupt_path, "wb") as f:
            f.write(b"This is definitely not a zip file.")

        with self.assertRaises(TranslatorError) as ctx:
            BaseFormatHandler.extract_zip(corrupt_path, self.extract_dir)

        self.assertEqual(ctx.exception.code, ErrorCode.E04)
        self.assertIn("Corrupted or invalid zip archive", ctx.exception.detail)
        self.assertFalse(os.path.exists(self.extract_dir))

    def test_extract_zip_custom_disk_reserve_policy(self):
        # Custom policy requiring minimal 1 KB reserve
        custom_policy = DocumentSecurityPolicy(
            min_free_disk_after_extract_bytes=1024,
            max_compression_ratio=200.0,
        )
        BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir, policy=custom_policy)
        self.assertTrue(os.path.exists(os.path.join(self.extract_dir, "word", "document.xml")))


if __name__ == "__main__":
    unittest.main()
