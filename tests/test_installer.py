"""
tests/test_installer.py
=======================
Unit tests for installer scripts and package verification security.
"""

import os
import sys
import tempfile
import zipfile
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from installer.install_argos_packages import (
    verify_package_archive,
    install_pair,
    check_privileges,
)


class TestInstallerSecurity(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_installer_")
        self._orig_log_file = sys.modules["installer.install_argos_packages"].LOG_FILE
        sys.modules["installer.install_argos_packages"].LOG_FILE = os.path.join(self.test_dir, "test.log")

    def tearDown(self):
        import shutil
        sys.modules["installer.install_argos_packages"].LOG_FILE = self._orig_log_file
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_verify_package_archive_valid(self):
        zip_path = os.path.join(self.test_dir, "valid.argosmodel")
        with zipfile.ZipFile(zip_path, "w") as zf:
            zf.writestr("model/model.bin", b"binary data")
            zf.writestr("metadata.json", b'{"from_code": "ja", "to_code": "en"}')

        is_safe, digest = verify_package_archive(zip_path)
        self.assertTrue(is_safe)
        self.assertEqual(len(digest), 64)  # 64-char hex SHA-256

        # With matching expected_hash
        is_safe_match, _ = verify_package_archive(zip_path, expected_hash=digest)
        self.assertTrue(is_safe_match)

        # With mismatching expected_hash
        is_safe_mismatch, _ = verify_package_archive(zip_path, expected_hash="0000000000000000000000000000000000000000000000000000000000000000")
        self.assertFalse(is_safe_mismatch)

    def test_verify_package_archive_corrupt(self):
        corrupt_path = os.path.join(self.test_dir, "corrupt.argosmodel")
        with open(corrupt_path, "wb") as f:
            f.write(b"not a valid zip file header")

        is_safe, digest = verify_package_archive(corrupt_path)
        self.assertFalse(is_safe)

    def test_verify_package_archive_nonexistent(self):
        nonexistent = os.path.join(self.test_dir, "missing.argosmodel")
        is_safe, digest = verify_package_archive(nonexistent)
        self.assertFalse(is_safe)
        self.assertEqual(digest, "")

    def test_verify_package_archive_path_traversal(self):
        traversal_path = os.path.join(self.test_dir, "traversal.argosmodel")
        with zipfile.ZipFile(traversal_path, "w") as zf:
            zf.writestr("../../../evil.exe", b"malicious executable")

        is_safe, digest = verify_package_archive(traversal_path)
        self.assertFalse(is_safe)

    def test_check_privileges_runs_safely(self):
        # Should execute without throwing any exception
        check_privileges()

    @patch.dict(sys.modules, {"argostranslate": None, "argostranslate.package": None})
    def test_install_pair_missing_argostranslate(self):
        # When argostranslate is not available, install_pair returns False safely
        result = install_pair("ja", "en")
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
