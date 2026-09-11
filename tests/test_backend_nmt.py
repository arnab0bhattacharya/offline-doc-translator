import json
import os
import sys
import tempfile
import unittest
import zipfile
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backend_nmt import (
    NMTBackend,
    compute_file_sha256,
    load_trusted_packages,
    normalize_nmt_placeholders,
    verify_package_archive,
)


class TestNMTBackend(unittest.TestCase):
    def test_placeholder_normalization(self):
        raw_1 = "Revenue increased by [ [ N0 ] ] % in Q3."
        norm_1 = normalize_nmt_placeholders(raw_1)
        self.assertEqual(norm_1, "Revenue increased by [[N0]] % in Q3.")

        raw_2 = "Total profit was $[ [ N0 ] ] and margin was [ [ N1 ] ] %."
        norm_2 = normalize_nmt_placeholders(raw_2)
        self.assertEqual(norm_2, "Total profit was $[[N0]] and margin was [[N1]] %.")

        raw_3 = "[ [ GLOSSARY_A ] ] terms"
        self.assertEqual(normalize_nmt_placeholders(raw_3), "[[GLOSSARY_A]] terms")

    def test_nmt_backend_init(self):
        backend = NMTBackend()
        # Should initialize gracefully whether Argos is installed or not
        self.assertIsInstance(backend.is_available(), bool)


class TestTrustedPackagesManifest(unittest.TestCase):
    """Verifies trusted packages manifest loading and fallback handling."""

    def test_load_trusted_packages_default(self):
        manifest = load_trusted_packages()
        self.assertIn("ja-en", manifest)
        self.assertIn("en-ja", manifest)

        ja_en = manifest["ja-en"]
        self.assertEqual(ja_en["from_code"], "ja")
        self.assertEqual(ja_en["to_code"], "en")
        self.assertEqual(ja_en["package_version"], "1.1")
        self.assertEqual(ja_en["sha256"], "623e3477959a815eb0a5ef53e09079ae8f1f9d3bbcd230473baf28c03fb83335")

        en_ja = manifest["en-ja"]
        self.assertEqual(en_ja["from_code"], "en")
        self.assertEqual(en_ja["to_code"], "ja")
        self.assertEqual(en_ja["package_version"], "1.1")
        self.assertEqual(en_ja["sha256"], "16300cc4eaa85320520cabcf433b63d01be40ef6966251de72043a083408f716")

    def test_load_trusted_packages_fallback(self):
        # Non-existent file path should return built-in fallback dictionary
        manifest = load_trusted_packages("/nonexistent/file/path.json")
        self.assertIn("ja-en", manifest)
        self.assertIn("en-ja", manifest)

    def test_load_trusted_packages_custom(self):
        with tempfile.TemporaryDirectory() as td:
            custom_path = os.path.join(td, "custom_trusted.json")
            custom_data = {
                "_comment": "test",
                "custom-pair": {"from_code": "es", "to_code": "en", "sha256": "abcdef123456"},
            }
            with open(custom_path, "w", encoding="utf-8") as f:
                json.dump(custom_data, f)

            manifest = load_trusted_packages(custom_path)
            self.assertIn("custom-pair", manifest)
            self.assertNotIn("_comment", manifest)


class TestPackageArchiveVerification(unittest.TestCase):
    """Verifies SHA-256 calculation, zip CRC validation, and ZipSlip prevention."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.td.cleanup()

    def _create_valid_archive(self, filename="model.argosmodel") -> str:
        path = os.path.join(self.td.name, filename)
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("model/model.bin", b"model weights binary data")
            zf.writestr("metadata.json", b'{"from_code": "ja", "to_code": "en"}')
        return path

    def test_compute_file_sha256(self):
        test_file = os.path.join(self.td.name, "hash_test.bin")
        with open(test_file, "wb") as f:
            f.write(b"offline-doc-translator")

        import hashlib

        expected = hashlib.sha256(b"offline-doc-translator").hexdigest()
        self.assertEqual(compute_file_sha256(test_file), expected)

    def test_verify_valid_archive_with_matching_hash(self):
        path = self._create_valid_archive()
        digest = compute_file_sha256(path)

        # 1. Without expected_hash
        is_valid, out_digest = verify_package_archive(path)
        self.assertTrue(is_valid)
        self.assertEqual(out_digest, digest)

        # 2. With matching expected_hash
        is_valid, _ = verify_package_archive(path, expected_hash=digest)
        self.assertTrue(is_valid)

        # 3. Case insensitivity check
        is_valid, _ = verify_package_archive(path, expected_hash=digest.upper())
        self.assertTrue(is_valid)

    def test_verify_archive_with_mismatching_hash(self):
        path = self._create_valid_archive()
        logs = []
        is_valid, _ = verify_package_archive(
            path, expected_hash="1111111111111111111111111111111111111111111111111111111111111111", log_cb=logs.append
        )
        self.assertFalse(is_valid)
        self.assertTrue(any("mismatch" in m.lower() for m in logs))

    def test_verify_corrupt_archive(self):
        path = os.path.join(self.td.name, "corrupt.argosmodel")
        with open(path, "wb") as f:
            f.write(b"this is corrupt non-zip data")

        is_valid, _ = verify_package_archive(path)
        self.assertFalse(is_valid)

    def test_verify_zip_slip_path_traversal(self):
        path = os.path.join(self.td.name, "malicious.argosmodel")
        with zipfile.ZipFile(path, "w") as zf:
            zf.writestr("../../evil_startup.bat", b"malicious code")

        logs = []
        is_valid, _ = verify_package_archive(path, log_cb=logs.append)
        self.assertFalse(is_valid)
        self.assertTrue(any("unsafe path" in m.lower() for m in logs))

    def test_verify_missing_file(self):
        is_valid, digest = verify_package_archive("/nonexistent/file.argosmodel")
        self.assertFalse(is_valid)
        self.assertEqual(digest, "")


class TestNMTBackendPackagePinning(unittest.TestCase):
    """Verifies NMTBackend integration with trusted package pinning."""

    def test_get_expected_hash(self):
        backend = NMTBackend()
        ja_en_hash = backend.get_expected_hash("ja", "en")
        self.assertEqual(ja_en_hash, "623e3477959a815eb0a5ef53e09079ae8f1f9d3bbcd230473baf28c03fb83335")

        en_ja_hash = backend.get_expected_hash("en", "ja")
        self.assertEqual(en_ja_hash, "16300cc4eaa85320520cabcf433b63d01be40ef6966251de72043a083408f716")

        # Unpinned pair returns None
        self.assertIsNone(backend.get_expected_hash("fr", "de"))

    def test_verify_package_helper_method(self):
        backend = NMTBackend()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "pkg.argosmodel")
            with zipfile.ZipFile(path, "w") as zf:
                zf.writestr("model.bin", b"test")

            correct_hash = compute_file_sha256(path)
            self.assertTrue(backend._verify_package(path, correct_hash))
            self.assertFalse(backend._verify_package(path, "wrong_hash"))

    def test_install_language_pair_refuses_unpinned_pair_under_strict_mode(self):
        backend = NMTBackend(strict_pinning=True)
        backend._argos_available = True

        logs = []
        result = backend.install_language_pair("fr", "de", log_cb=logs.append)
        self.assertFalse(result)
        self.assertTrue(any("unpinned" in m.lower() or "no trusted" in m.lower() for m in logs))

    def test_install_language_pair_already_installed_returns_true(self):
        backend = NMTBackend()
        backend._argos_available = True
        backend._installed_pairs.add(("ja", "en"))

        # Should return True without checking network or packages
        self.assertTrue(backend.install_language_pair("ja", "en"))

    def test_install_aborts_and_deletes_download_on_hash_mismatch(self):
        backend = NMTBackend()
        backend._argos_available = True
        backend._installed_pairs.clear()

        with tempfile.TemporaryDirectory() as td:
            download_file = os.path.join(td, "downloaded.argosmodel")
            with zipfile.ZipFile(download_file, "w") as zf:
                zf.writestr("model.bin", b"some unverified data")

            mock_pkg = MagicMock()
            mock_pkg.from_code = "ja"
            mock_pkg.to_code = "en"
            mock_pkg.download.return_value = download_file

            logs = []
            with (
                patch("argostranslate.package.update_package_index"),
                patch("argostranslate.package.get_available_packages", return_value=[mock_pkg]),
                patch("argostranslate.package.install_from_path") as mock_install,
            ):
                success = backend.install_language_pair("ja", "en", log_cb=logs.append)

            self.assertFalse(success)
            self.assertTrue(any("failed verification" in m.lower() for m in logs))
            # Downloaded unverified file must be deleted
            self.assertFalse(os.path.exists(download_file))
            mock_install.assert_not_called()

    def test_install_succeeds_when_hash_matches(self):
        backend = NMTBackend()
        backend._argos_available = True
        backend._installed_pairs.clear()

        with tempfile.TemporaryDirectory() as td:
            download_file = os.path.join(td, "downloaded.argosmodel")
            with zipfile.ZipFile(download_file, "w") as zf:
                zf.writestr("model.bin", b"verified model bytes")

            # Update backend manifest to match this exact test file
            file_hash = compute_file_sha256(download_file)
            backend._trusted_manifest["ja-en"]["sha256"] = file_hash

            mock_pkg = MagicMock()
            mock_pkg.from_code = "ja"
            mock_pkg.to_code = "en"
            mock_pkg.download.return_value = download_file

            logs = []
            with (
                patch("argostranslate.package.update_package_index"),
                patch("argostranslate.package.get_available_packages", return_value=[mock_pkg]),
                patch("argostranslate.package.install_from_path") as mock_install,
            ):
                success = backend.install_language_pair("ja", "en", log_cb=logs.append)

            self.assertTrue(success)
            self.assertIn(("ja", "en"), backend._installed_pairs)
            mock_install.assert_called_once_with(download_file)
            self.assertTrue(any("successfully installed" in m.lower() for m in logs))


if __name__ == "__main__":
    unittest.main()
