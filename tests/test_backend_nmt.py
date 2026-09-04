"""
tests/test_backend_nmt.py
=========================
Unit tests for the NMT backend and placeholder normalization.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backend_nmt import normalize_nmt_placeholders, NMTBackend


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


if __name__ == "__main__":
    unittest.main()
