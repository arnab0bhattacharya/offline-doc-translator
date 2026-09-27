"""
tests/test_nmt_clean_pipeline.py
================================
Unit tests for the clean NMT translation pipeline, post-translation glossary
alignment, numeric token extraction, and audit verification.
"""

import os
import sys
import tempfile
import unittest

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core import (
    TranslationEngine,
    TranslationMode,
    TranslationResult,
    extract_numeric_tokens,
    verify_nmt_numbers,
)


class MockNMTBackend:
    """Mock NMT backend that records inputs received and returns simulated translations."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self.mapping = mapping or {}
        self.received_texts: list[str] = []

    def is_ready(self, direction: str) -> bool:
        return True

    def translate_single(self, text: str, direction: str) -> str:
        self.received_texts.append(text)
        if text in self.mapping:
            return self.mapping[text]
        if direction == "ja2en":
            return f"Translated: {text}"
        return f"翻訳: {text}"


class MockLLMBackend:
    """Mock LLM backend that returns uncorrupted masked output."""

    def __init__(self):
        self.received_texts: list[str] = []

    def is_ready(self, direction: str) -> bool:
        return True

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: dict[str, str] | None = None,
        context: str | None = None,
        log_cb=None,
    ) -> tuple[str, float]:
        self.received_texts.append(text)
        return text, 0.05


class TestNMTNumericVerification(unittest.TestCase):
    """Test numeric token extraction and date/number auditing for NMT."""

    def test_extract_numeric_tokens(self):
        text = "In 2024, sales reached 1,250,000 units, up 15.5% compared to 3 years ago."
        tokens = extract_numeric_tokens(text)
        self.assertEqual(tokens, ["2024", "1250000", "15.5", "3"])

    def test_extract_numeric_tokens_empty(self):
        self.assertEqual(extract_numeric_tokens("No numbers here."), [])

    def test_verify_nmt_numbers_exact_match(self):
        src = "2024年に15.5%の成長"
        tgt = "15.5% growth in 2024"
        valid, missing = verify_nmt_numbers(src, tgt)
        self.assertTrue(valid)
        self.assertEqual(missing, [])

    def test_verify_nmt_numbers_missing_number(self):
        src = "第3四半期の売上は42億円でした"
        tgt = "The sales for the quarter were 3 billion yen."
        valid, missing = verify_nmt_numbers(src, tgt)
        self.assertFalse(valid)
        self.assertIn("42", missing)

    def test_verify_nmt_numbers_month_conversion(self):
        # Japanese '10月' translated natively to English 'October'
        src = "2024年10月5日に開始"
        tgt = "Started on October 5, 2024"
        valid, missing = verify_nmt_numbers(src, tgt)
        self.assertTrue(valid)
        self.assertEqual(missing, [])

    def test_verify_nmt_numbers_month_abbreviation(self):
        src = "2023年3月15日"
        tgt = "Mar 15, 2023"
        valid, missing = verify_nmt_numbers(src, tgt)
        self.assertTrue(valid)
        self.assertEqual(missing, [])


class TestNMTCleanPipeline(unittest.TestCase):
    """Test that NMT translates clean unmasked text and applies post-translation glossary."""

    def test_nmt_receives_unmasked_text(self):
        """Ensures [[N0]] or [[GLOSSARY_A]] are never passed to the NMT model."""
        mock_backend = MockNMTBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "sales"}
            engine = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.FAST_NMT)

            source = "2024年に売上高は15%増加しました。"
            result = engine.translate_chunk(source, direction="ja2en")

            # Check that backend received clean text without synthetic masks
            self.assertGreater(len(mock_backend.received_texts), 0)
            chunk_received = mock_backend.received_texts[0]
            self.assertNotIn("[[N", chunk_received)
            self.assertNotIn("[[GLOSSARY", chunk_received)
            self.assertIn("2024", chunk_received)
            self.assertIn("15%", chunk_received)
            self.assertIsInstance(result, TranslationResult)

    def test_nmt_glossary_post_substitution_ja2en(self):
        """Tests that NMT translates sentence, then glossary term replaces default model phrasing."""
        mock_backend = MockNMTBackend(
            mapping={
                "売上高は15%増加しました。": "Revenue increased by 15%.",
                "売上高": "Revenue",
            }
        )
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "sales"}
            engine = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.FAST_NMT)

            # User glossary mandates "sales" instead of "Revenue"
            result = engine.translate_chunk(
                "売上高は15%増加しました。",
                direction="ja2en",
            )

            # Post-translation replacement should have substituted "Revenue" with "sales"
            self.assertIn("sales increased by 15%.", result.text)
            self.assertNotIn("Revenue", result.text)

    def test_nmt_glossary_already_matching_target(self):
        """If model output already contains target term, no redundant replacement occurs."""
        mock_backend = MockNMTBackend(
            mapping={
                "売上高は15%増加しました。": "Sales increased by 15%.",
            }
        )
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "Sales"}
            engine = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.FAST_NMT)

            result = engine.translate_chunk(
                "売上高は15%増加しました。",
                direction="ja2en",
            )
            self.assertEqual(result.text, "Sales increased by 15%.")

    def test_nmt_glossary_post_substitution_en2ja(self):
        """Tests en2ja post-translation glossary substitution."""
        mock_backend = MockNMTBackend(
            mapping={
                "The corporate revenue rose by 10%.": "企業の収入は10％増加しました。",
                "revenue": "収入",
            }
        )
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"revenue": "売上高"}
            engine = TranslationEngine(
                mode=TranslationMode.FAST_NMT,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.FAST_NMT)

            result = engine.translate_chunk(
                "The corporate revenue rose by 10%.",
                direction="en2ja",
            )
            self.assertIn("企業の売上高は10％増加しました。", result.text)

    def test_llm_mode_retains_synthetic_masking(self):
        """Ensures PURE_LLM mode continues to use synthetic masking for prompt anchoring."""
        mock_llm = MockLLMBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "sales"}
            engine = TranslationEngine(
                mode=TranslationMode.PURE_LLM,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_llm, mode=TranslationMode.PURE_LLM)

            source = "2024年に売上高は15%増加しました。"
            result = engine.translate_chunk(source, direction="ja2en")

            # Verify text sent to LLM contains [[N0]] and [[GLOSSARY_A]]
            self.assertGreater(len(mock_llm.received_texts), 0)
            text_received = mock_llm.received_texts[0]
            self.assertIn("[[N0]]", text_received)
            self.assertIn("[[GLOSSARY_A]]", text_received)

            # Verify final output has been unmasked
            self.assertNotIn("[[N0]]", result.text)
            self.assertNotIn("[[GLOSSARY_A]]", result.text)
            self.assertIn("2024", result.text)
            self.assertIn("sales", result.text)

    def test_machine_translation_mode_receives_unmasked_text(self):
        """Ensures MACHINE_TRANSLATION mode translates clean text and executes smoothly."""
        mock_backend = MockNMTBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "sales"}
            engine = TranslationEngine(
                mode=TranslationMode.MACHINE_TRANSLATION,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.MACHINE_TRANSLATION)

            source = "2024年に売上高は15%増加しました。"
            result = engine.translate_chunk(source, direction="ja2en")
            self.assertTrue(result.was_translated)
            self.assertFalse(result.was_reverted)
            self.assertEqual(result.source_backend, "nmt")

    def test_ai_translation_mode_retains_synthetic_masking(self):
        """Ensures AI_TRANSLATION mode uses synthetic masking and unmasks."""
        mock_llm = MockLLMBackend()
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            glossary = {"売上高": "sales"}
            engine = TranslationEngine(
                mode=TranslationMode.AI_TRANSLATION,
                cache_file=cache_file,
                glossary=glossary,
            )
            engine.set_backend(mock_llm, mode=TranslationMode.AI_TRANSLATION)

            source = "2024年に売上高は15%増加しました。"
            result = engine.translate_chunk(source, direction="ja2en")
            self.assertTrue(result.was_translated)
            self.assertIn("sales", result.text)


if __name__ == "__main__":
    unittest.main()
