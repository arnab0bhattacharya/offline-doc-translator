"""
tests/test_preflight.py
=======================
Tests for preflight checks, system diagnostics, and standardized error codes.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.errors import ERROR_MESSAGES, ErrorCode, TranslatorError
from engine.preflight import (
    check_disk_space,
    check_ram,
    run_preflight,
)


class TestPreflight(unittest.TestCase):
    def test_error_definitions(self):
        for code in ErrorCode:
            self.assertIn(code, ERROR_MESSAGES)
            meta = ERROR_MESSAGES[code]
            self.assertTrue(len(meta["title"]) > 0)
            self.assertTrue(len(meta["message"]) > 0)
            self.assertTrue(len(meta["action"]) > 0)

    def test_translator_error_formatting(self):
        err = TranslatorError(ErrorCode.E01, detail="Connection refused on port 11434")
        self.assertEqual(err.code, ErrorCode.E01)
        dialog_str = err.format_user_dialog()
        self.assertIn("E01", dialog_str)
        self.assertIn("Translation Engine Offline", dialog_str)
        self.assertIn("Connection refused on port 11434", dialog_str)

    def test_ram_check(self):
        has_ram, ram_mb = check_ram(min_free_mb=10)
        self.assertTrue(has_ram)
        self.assertGreater(ram_mb, 10.0)

    def test_disk_space_check(self):
        has_disk, disk_mb = check_disk_space(".", min_free_mb=10)
        self.assertTrue(has_disk)
        self.assertGreater(disk_mb, 10.0)

    def test_preflight_file_not_found(self):
        with self.assertRaises(TranslatorError) as ctx:
            run_preflight(
                model_name="mock_model",
                input_path="non_existent_file_12345.pptx",
                output_path="out.pptx",
                ollama_url="http://localhost:11434",
                check_model=False,
                require_ollama=False,
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E04)


if __name__ == "__main__":
    unittest.main()
