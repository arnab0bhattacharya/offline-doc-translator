"""
tests/test_core.py
==================
Unit tests for the translation engine, number masking, bidirectional gates,
and XML-anchored prompt construction.
"""

import unittest
import os
import sys
import tempfile

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core import (
    contains_japanese,
    contains_latin,
    should_translate,
    mask_numbers,
    mask_glossary_terms,
    verify_placeholders,
    unmask_numbers,
    unmask_protected_text,
    escape_xml,
    clean_llm_response,
    build_prompts,
    TranslationEngine,
    TranslationMode,
)
from engine.errors import ErrorCode, TranslatorError


class TestCoreEngine(unittest.TestCase):

    def test_language_detection(self):
        # Japanese script detection
        self.assertTrue(contains_japanese("これはテストです。"))
        self.assertTrue(contains_japanese("Q3売上高: 100億円"))
        self.assertFalse(contains_japanese("This is pure English."))
        self.assertFalse(contains_japanese("12345 + 6789 = 0"))

        # Latin prose detection (requires 2+ consecutive letters on word boundary)
        self.assertTrue(contains_latin("This is a report."))
        self.assertTrue(contains_latin("Revenue grew by 15%."))
        self.assertFalse(contains_latin("123"))
        self.assertFalse(contains_latin("F 1"))  # isolated single letters

    def test_symmetric_gate(self):
        # ja2en: translate only if Japanese is present
        self.assertTrue(should_translate("売上高の推移", "ja2en"))
        self.assertTrue(should_translate("Total Q3: 100万円", "ja2en"))
        self.assertFalse(should_translate("Total Q3: $100M", "ja2en"))

        # en2ja: translate only if Latin prose present AND no Japanese
        self.assertTrue(should_translate("Revenue decreased by 5%", "en2ja"))
        self.assertFalse(should_translate("Revenue 諸経費 5%", "en2ja"))  # Contains Japanese
        self.assertFalse(should_translate("12345", "en2ja"))  # No Latin prose

    def test_number_masking_and_unmasking(self):
        text = "Q3の売上高は 15.5% 増加し、合計は $1,000,000 (前年比 +5%) でした。"
        masked, num_map = mask_numbers(text)

        self.assertIn("[[N0]]", masked)
        self.assertIn("[[N1]]", masked)
        self.assertIn("[[N2]]", masked)

        # Placeholders must verify
        self.assertTrue(verify_placeholders(masked, num_map, masked))

        # Test unmasking
        unmasked = unmask_numbers(masked, num_map)
        self.assertEqual(unmasked, text)

    def test_duplicate_number_masking(self):
        # Identical numbers should share the same placeholder token
        text = "Region A: 50% vs Region B: 50%"
        masked, num_map = mask_numbers(text)
        self.assertEqual(len(num_map), 1)
        self.assertIn("[[N0]]", masked)
        self.assertNotIn("[[N1]]", masked)
        self.assertEqual(unmask_numbers(masked, num_map), text)

    def test_glossary_masking_enforces_target_in_all_modes(self):
        text = "Contract terms and Contract value"
        masked, glossary_map = mask_glossary_terms(text, {"Contract": "契約書"})
        self.assertEqual(masked.count("[[GLOSSARY_A]]"), 2)
        self.assertEqual(glossary_map, {"[[GLOSSARY_A]]": "契約書"})
        self.assertEqual(
            unmask_protected_text("[[GLOSSARY_A]] 条件", {}, glossary_map),
            "契約書 条件"
        )

    def test_pure_llm_restores_glossary_terms_deterministically(self):
        class FakeLLM:
            def translate_single(self, **kwargs):
                return "[[GLOSSARY_A]] 条件", 0.0

        engine = TranslationEngine(
            mode=TranslationMode.PURE_LLM,
            glossary={"Contract": "契約書"}
        )
        engine._llm_backend = FakeLLM()
        translated, was_translated, was_reverted = engine.translate_chunk("Contract terms", "en2ja")
        self.assertEqual(translated, "契約書 条件")
        self.assertTrue(was_translated)
        self.assertFalse(was_reverted)


    def test_placeholder_validation_requires_exact_counts(self):
        masked = "[[N0]] compared with [[N0]] and [[N1]]"
        num_map = {"[[N0]]": "50", "[[N1]]": "20"}
        self.assertTrue(verify_placeholders(masked, num_map, masked))
        self.assertFalse(verify_placeholders("[[N0]] and [[N1]]", num_map, masked))
        self.assertFalse(verify_placeholders("[[N0]] [[N0]] [[N1]] [[N2]]", num_map, masked))

    def test_cache_isolated_by_model_glossary_and_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_file = os.path.join(temp_dir, "cache.json")
            engine = TranslationEngine(
                model_name="model-a",
                mode=TranslationMode.PURE_LLM,
                glossary={"売上": "Revenue"},
                cache_file=cache_file,
            )
            engine.load_cache("ja2en")
            first = engine._get_direction_cache("ja2en", "Slide A")
            second = engine._get_direction_cache("ja2en", "Slide B")
            self.assertIsNot(first, second)
            first["key"] = "value"
            engine.save_cache_atomically()

            changed_model = TranslationEngine(model_name="model-b", cache_file=cache_file)
            changed_model.load_cache("ja2en")
            self.assertNotIn("key", changed_model._get_direction_cache("ja2en", "Slide A"))

    def test_fast_nmt_never_falls_back_to_llm(self):
        class UnreadyNMT:
            def is_ready(self, direction):
                return False

        engine = TranslationEngine(mode=TranslationMode.FAST_NMT)
        engine._nmt_backend = UnreadyNMT()
        with self.assertRaises(TranslatorError) as ctx:
            engine.translate_chunk("売上高", "ja2en")
        self.assertEqual(ctx.exception.code, ErrorCode.E08)

    def test_xml_escaping(self):
        raw = "AT&T <revenue> > $500M & 'profit' \"margin\""
        escaped = escape_xml(raw)
        self.assertNotIn("<revenue>", escaped)
        self.assertIn("&amp;", escaped)
        self.assertIn("&lt;revenue&gt;", escaped)
        self.assertIn("&gt;", escaped)

    def test_clean_llm_response(self):
        self.assertEqual(clean_llm_response("  Hello World  "), "Hello World")
        self.assertEqual(
            clean_llm_response("<target>\nTranslated text here\n</target>"),
            "Translated text here"
        )
        self.assertEqual(
            clean_llm_response("```json\nTranslated content\n```"),
            "Translated content"
        )

    def test_prompt_construction(self):
        masked = "Total revenue was [[N0]]% higher."
        num_map = {"[[N0]]": "15"}
        context = "Slide Header: Q3 Financial Results"

        p1_ja, p2_ja = build_prompts("en2ja", masked, num_map, context)
        self.assertIn("Reference context", p1_ja)
        self.assertIn("Slide Header: Q3 Financial Results", p1_ja)
        self.assertIn("Input JSON", p1_ja)
        self.assertIn(masked, p1_ja)
        self.assertIn("[[N0]]", p2_ja)


if __name__ == "__main__":
    unittest.main()
