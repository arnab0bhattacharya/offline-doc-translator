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
    hash_text,
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

    def test_verify_nmt_numbers_full_width_zenkaku(self):
        """Verifies that Japanese full-width numerals (NFKC) match standard ASCII numbers."""
        src = "２０２４年１０月５日に開始、売上は１５．５％増"
        tgt = "Started on October 5, 2024, sales up 15.5%"
        audit = verify_nmt_numbers(src, tgt)
        self.assertTrue(audit.passed)
        self.assertEqual(audit.missing, [])
        self.assertEqual(audit.added, [])

    def test_verify_nmt_numbers_ordinals_matching(self):
        """Verifies that ordinal indicators (1st, 2nd, etc.) match base ordinal quantities."""
        src = "第1四半期の決算報告"
        tgt = "Financial report for the 1st quarter"
        audit = verify_nmt_numbers(src, tgt)
        self.assertTrue(audit.passed)
        self.assertEqual(audit.missing, [])
        self.assertEqual(audit.added, [])

    def test_verify_nmt_numbers_ordinal_mismatch(self):
        """Verifies that ordinal number alterations (1st -> 2nd) fail the audit gate."""
        src = "第1四半期の決算報告"
        tgt = "Financial report for the 2nd quarter"
        audit = verify_nmt_numbers(src, tgt)
        self.assertFalse(audit.passed)
        self.assertIn("1", audit.missing)
        self.assertIn("2", audit.added)

    def test_verify_nmt_numbers_repeated_count_mismatch(self):
        """Verifies multiset comparison catches dropped duplicate numbers."""
        src = "A社は100億円、B社も100億円の出資を行いました。"
        tgt = "Company A invested 100 billion yen."
        audit = verify_nmt_numbers(src, tgt)
        self.assertFalse(audit.passed)
        self.assertIn("100", audit.missing)
        self.assertEqual(audit.added, [])

    def test_extract_numeric_tokens_units_and_scales(self):
        """Verifies extraction and canonicalization of scale multipliers and physical/digital units."""
        self.assertEqual(extract_numeric_tokens("Revenue 42M"), ["42M"])
        self.assertEqual(extract_numeric_tokens("Revenue was 42 million"), ["42"])
        self.assertEqual(extract_numeric_tokens("Weight is 500kg"), ["500kg"])
        self.assertEqual(extract_numeric_tokens("Weight is 500 kg"), ["500kg"])
        self.assertEqual(extract_numeric_tokens("Memory is 32GB"), ["32GB"])
        self.assertEqual(extract_numeric_tokens("Speed is 10Gbps"), ["10Gbps"])

    def test_extract_numeric_tokens_ignores_underscore_identifiers(self):
        """Verifies that identifier-like strings with underscores are not extracted as numbers."""
        self.assertEqual(extract_numeric_tokens("Device 500kg_v2"), [])
        self.assertEqual(extract_numeric_tokens("Identifier part_42M_rev1"), [])
        self.assertEqual(extract_numeric_tokens("Variable counter_10k_max"), [])

    def test_verify_nmt_numbers_unit_and_scale_matching(self):
        """Verifies that unit scale changes (42M -> 42k or 42) fail, while safe equivalences pass."""
        # 42M matches 42M and 42 million
        self.assertTrue(verify_nmt_numbers("Revenue 42M", "Revenue 42M").passed)
        self.assertTrue(verify_nmt_numbers("Revenue 42M", "Revenue was 42 million").passed)

        # 42M fails against 42k or plain 42
        audit_k = verify_nmt_numbers("Revenue 42M", "Revenue was 42k")
        self.assertFalse(audit_k.passed)
        self.assertIn("42M", audit_k.missing)
        self.assertIn("42k", audit_k.added)

        audit_plain = verify_nmt_numbers("Revenue 42M", "Revenue was 42")
        self.assertFalse(audit_plain.passed)
        self.assertIn("42M", audit_plain.missing)
        self.assertIn("42", audit_plain.added)

        # 500kg matches 500 kg, but fails against 500g or plain 500
        self.assertTrue(verify_nmt_numbers("Weight is 500kg", "Weight is 500 kg").passed)
        self.assertFalse(verify_nmt_numbers("Weight is 500kg", "Weight is 500g").passed)
        self.assertFalse(verify_nmt_numbers("Weight is 500kg", "Weight is 500").passed)

    def test_verify_nmt_numbers_disambiguates_modal_may(self):
        """Verifies that English modal verb 'may' cannot satisfy a lost quantity 5 without date context."""
        # 5社 (5 companies) -> lost quantity 5 translated to modal verb "may" -> MUST FAIL
        audit_modal = verify_nmt_numbers("5社が参加した。", "Several companies may participate.")
        self.assertFalse(audit_modal.passed)
        self.assertIn("5", audit_modal.missing)

        # 5月 (May in date context) -> DOES match 'May'
        audit_date = verify_nmt_numbers("2024年5月に開始。", "Started in May 2024.")
        self.assertTrue(audit_date.passed)
        self.assertEqual(audit_date.missing, [])

    def test_verify_nmt_numbers_month_multiset_counts(self):
        """Verifies multiset tracking for months: one October cannot satisfy two occurrences of 10."""
        # Two 10月 vs single October -> fails
        audit_fail = verify_nmt_numbers("10月と10月の会合", "Meeting in October")
        self.assertFalse(audit_fail.passed)
        self.assertIn("10", audit_fail.missing)

        # Two 10月 vs two Octobers -> passes
        audit_pass = verify_nmt_numbers("10月と10月の会合", "Meetings in October and October")
        self.assertTrue(audit_pass.passed)

    def test_verify_nmt_numbers_spelled_out_ordinals(self):
        """Verifies that spelled-out English ordinals (first through tenth) match corresponding ordinals."""
        # 第1四半期 matches 'first quarter'
        audit_1 = verify_nmt_numbers("第1四半期の決算報告", "Financial report for the first quarter")
        self.assertTrue(audit_1.passed)

        # 第1四半期 fails against 'second quarter'
        audit_mismatch = verify_nmt_numbers("第1四半期の決算報告", "Financial report for the second quarter")
        self.assertFalse(audit_mismatch.passed)
        self.assertIn("1", audit_mismatch.missing)

        # Multiset ordinal counts: two 第1 ordinals require two 'first' words
        audit_two_fail = verify_nmt_numbers("第1四半期と第1事業部", "first quarter and division")
        self.assertFalse(audit_two_fail.passed)
        self.assertIn("1", audit_two_fail.missing)

        audit_two_pass = verify_nmt_numbers("第1四半期と第1事業部", "first quarter and first division")
        self.assertTrue(audit_two_pass.passed)

    def test_structured_audit_result_properties_and_backward_compatibility(self):
        """Verifies NumericAuditResult dataclass properties, summary formatting, and tuple unpacking."""
        audit = verify_nmt_numbers("100", "200")
        self.assertFalse(audit.passed)
        self.assertIn("missing ['100']", audit.summary())
        self.assertIn("added ['200']", audit.summary())
        # Test tuple unpacking backward compatibility: (valid, missing) = result
        valid, missing = audit
        self.assertFalse(valid)
        self.assertEqual(missing, ["100"])
        # Test index access
        self.assertFalse(audit[0])
        self.assertEqual(audit[1], ["100"])

    def test_verify_nmt_numbers_bidirectional_scale_words(self):
        """Verifies scale-word equivalences operate symmetrically in both directions."""
        # Attached to word
        self.assertTrue(verify_nmt_numbers("Revenue 42M", "Revenue was 42 million").passed)
        # Word to attached (reverse)
        self.assertTrue(verify_nmt_numbers("Revenue was 42 million", "Revenue 42M").passed)
        # Word to word
        self.assertTrue(verify_nmt_numbers("Revenue was 42 million", "Revenue was 42 million").passed)
        # Scale mismatch
        audit_scale_diff = verify_nmt_numbers("Revenue was 42 million", "Revenue was 42 thousand")
        self.assertFalse(audit_scale_diff.passed)
        self.assertIn("42M", audit_scale_diff.missing)
        self.assertIn("42", audit_scale_diff.added)

        audit_scale_k = verify_nmt_numbers("Revenue was 42 million", "Revenue 42k")
        self.assertFalse(audit_scale_k.passed)
        self.assertIn("42M", audit_scale_k.missing)
        self.assertIn("42k", audit_scale_k.added)

        # Scale dropped
        audit_plain = verify_nmt_numbers("Revenue was 42 million", "Revenue was 42")
        self.assertFalse(audit_plain.passed)
        self.assertIn("42M", audit_plain.missing)
        self.assertIn("42", audit_plain.added)

    def test_verify_nmt_numbers_bidirectional_month_conversion(self):
        """Verifies month equivalences operate symmetrically in ja2en and en2ja."""
        # ja2en
        self.assertTrue(verify_nmt_numbers("Started in 5月 2024", "Started in May 2024").passed)
        # en2ja (reverse)
        self.assertTrue(verify_nmt_numbers("Started in May 2024", "2024年5月に開始").passed)
        # Month alteration in reverse
        audit_mismatch = verify_nmt_numbers("Started in May 2024", "2024年10月に開始")
        self.assertFalse(audit_mismatch.passed)
        self.assertIn("10", audit_mismatch.added)

        # Other months en2ja
        self.assertTrue(verify_nmt_numbers("Started in October 2024", "2024年10月に開始").passed)

    def test_verify_nmt_numbers_bidirectional_spelled_out_ordinals(self):
        """Verifies ordinal word equivalences operate symmetrically across CJK boundaries."""
        # ja2en
        self.assertTrue(verify_nmt_numbers("第1四半期と2024年", "2024年とfirst quarter").passed)
        # en2ja (reverse)
        self.assertTrue(verify_nmt_numbers("2024年とfirst quarter", "第1四半期と2024年").passed)
        # Ordinal alteration in reverse
        audit_mismatch = verify_nmt_numbers("2024年とfirst quarter", "第2四半期と2024年")
        self.assertFalse(audit_mismatch.passed)
        self.assertIn("2", audit_mismatch.added)

        # Correct second quarter
        self.assertTrue(verify_nmt_numbers("2024年とsecond quarter", "第2四半期と2024年").passed)

    def test_verify_nmt_numbers_may_modal_verb_strict_date_anchoring(self):
        """Verifies strict date anchoring prevents English modal verb 'may' from satisfying missing month."""
        # Missing date number 5 translated to modal verb "may" -> REJECTED
        audit_modal = verify_nmt_numbers("5月の売上は増加した", "Sales may increase.")
        self.assertFalse(audit_modal.passed)
        self.assertIn("5", audit_modal.missing)

        audit_modal_cap = verify_nmt_numbers("5月の売上は増加した", "Sales May increase.")
        self.assertFalse(audit_modal_cap.passed)
        self.assertIn("5", audit_modal_cap.missing)

        # Proper preposition date context -> ACCEPTED
        self.assertTrue(verify_nmt_numbers("5月の売上は増加した", "Sales in May increased.").passed)
        # Proper month + year date context -> ACCEPTED
        self.assertTrue(verify_nmt_numbers("2024年5月に開始", "Started in May 2024.").passed)
        self.assertTrue(verify_nmt_numbers("2024年5月に開始", "Started in late May 2024.").passed)


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

    def test_nmt_number_audit_failure_reverts_and_refuses_cache(self):
        """Ensures that when an NMT model drops a number, the output is reverted and NOT cached."""
        mock_backend = MockNMTBackend(
            mapping={
                "売上高は42億円でした。": "Sales were reported for the period.",  # Number '42' dropped!
            }
        )
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            review_log = os.path.join(td, "review.txt")
            engine = TranslationEngine(
                mode=TranslationMode.MACHINE_TRANSLATION,
                cache_file=cache_file,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.MACHINE_TRANSLATION)

            source = "売上高は42億円でした。"
            result = engine.translate_chunk(
                source,
                direction="ja2en",
                location_id="sheet1_A1",
                review_log_path=review_log,
            )

            # 1. Output must be reverted to original source text
            self.assertFalse(result.was_translated)
            self.assertTrue(result.was_reverted)
            self.assertEqual(result.text, source)

            # 2. Key must be registered in failed_this_run
            key = hash_text(source)
            self.assertIn(key, engine.failed_this_run)

            # 3. Cache must NOT contain the corrupted output
            cached = engine._cache_mgr.get(key=key, direction="ja2en", fingerprint="", mode="machine_translation")
            self.assertIsNone(cached)

            # 4. Human review log must be recorded
            self.assertTrue(os.path.exists(review_log))
            with open(review_log, encoding="utf-8") as f:
                content = f.read()
                self.assertIn("sheet1_A1", content)
                self.assertIn(key[:16], content)

    def test_nmt_cache_read_rejects_and_bypasses_bad_cached_number(self):
        """Ensures that stale/legacy cache entries with missing numbers are rejected on read and evicted."""
        mock_backend = MockNMTBackend(
            mapping={
                "売上高は42億円でした。": "Sales were 42 billion yen.",  # Valid translation
            }
        )
        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            engine = TranslationEngine(
                mode=TranslationMode.MACHINE_TRANSLATION,
                cache_file=cache_file,
            )
            engine.set_backend(mock_backend, mode=TranslationMode.MACHINE_TRANSLATION)

            source = "売上高は42億円でした。"
            key = hash_text(source)

            # Poison cache with an old/bad entry that lost the number '42'
            fp = engine._cache_fingerprint(None)
            engine._cache_mgr.put(
                key=key,
                direction="ja2en",
                fingerprint=fp,
                value="Old bad cached translation without number.",
                mode="machine_translation",
            )

            # Request translation: engine should detect numeric failure on cache read,
            # evict the corrupt cache hit, and generate a fresh translation via backend.
            result = engine.translate_chunk(source, direction="ja2en")
            self.assertTrue(result.was_translated)
            self.assertFalse(result.was_reverted)
            self.assertEqual(result.text, "Sales were 42 billion yen.")
            self.assertEqual(result.source_backend, "nmt")  # Freshly generated, not served from bad cache!

    def test_nmt_cache_read_deletes_bad_entry_from_cache_mgr_and_disk(self):
        """Verifies that bad cache entries are deleted from persistent cache manager and removed from disk on save."""

        class FailingBackend:
            name = "nmt"

            def is_ready(self, direction):
                return True

            def translate(self, text, direction, **kwargs):
                raise RuntimeError("Backend failed deliberately")

        with tempfile.TemporaryDirectory() as td:
            cache_file = os.path.join(td, "cache.json")
            engine = TranslationEngine(
                mode=TranslationMode.MACHINE_TRANSLATION,
                cache_file=cache_file,
            )
            engine.set_backend(FailingBackend(), mode=TranslationMode.MACHINE_TRANSLATION)

            source = "売上高は42億円でした。"
            key = hash_text(source)
            fp = engine._cache_fingerprint(None)

            # Store bad entry in cache
            engine._cache_mgr.put(
                key=key,
                direction="ja2en",
                fingerprint=fp,
                value="Corrupt cached text with no numbers.",
                mode="machine_translation",
            )
            engine.save_cache_atomically()

            # Verify bad entry is in cache manager
            self.assertIsNotNone(
                engine._cache_mgr.get(key=key, direction="ja2en", fingerprint=fp, mode="machine_translation")
            )

            # Request translation: should fail audit on cache read, call delete() on cache_mgr,
            # then attempt backend (which fails), reverting to source text.
            res = engine.translate_chunk(source, direction="ja2en")
            self.assertFalse(res.was_translated)
            self.assertTrue(res.was_reverted)

            # Verify that bad entry has been deleted from cache manager!
            self.assertIsNone(
                engine._cache_mgr.get(key=key, direction="ja2en", fingerprint=fp, mode="machine_translation")
            )

            # Save cache to disk and verify disk persistence
            engine.save_cache_atomically()
            new_engine = TranslationEngine(
                mode=TranslationMode.MACHINE_TRANSLATION,
                cache_file=cache_file,
            )
            new_fp = new_engine._cache_fingerprint(None)
            # Fresh engine loading from disk must NOT contain the deleted corrupt entry
            self.assertIsNone(
                new_engine._cache_mgr.get(key=key, direction="ja2en", fingerprint=new_fp, mode="machine_translation")
            )


if __name__ == "__main__":
    unittest.main()
