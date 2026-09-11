"""
tests/test_backend_protocol.py
==============================
Unit tests for the TranslationBackend protocol interface,
runtime checkability, polymorphic backend dispatch in TranslationEngine,
and backward compatibility.
"""

import unittest
from collections.abc import Callable
from unittest.mock import patch

from engine.backend_base import TranslationBackend
from engine.backend_llm import LLMBackend
from engine.backend_nmt import NMTBackend
from engine.core import TranslationEngine, TranslationMode, TranslationResult
from engine.errors import ErrorCode, TranslatorError


class MockCustomBackend:
    """A valid implementation of TranslationBackend."""

    name: str = "custom_mock"

    def __init__(self, ready: bool = True):
        self._ready = ready
        self.call_history = []

    def is_available(self) -> bool:
        return True

    def is_ready(self, direction: str) -> bool:
        return self._ready

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: dict[str, str] | None = None,
        context: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> tuple[str | None, float]:
        self.call_history.append((text, direction, placeholder_map, context))
        if log_cb:
            log_cb(f"MockCustomBackend translated: {text}")
        return f"[CUSTOM: {text}]", 0.05


class IncompleteBackend:
    """Missing translate method - must not satisfy TranslationBackend."""

    name: str = "incomplete"

    def is_available(self) -> bool:
        return True

    def is_ready(self, direction: str) -> bool:
        return True


class TestTranslationBackendProtocol(unittest.TestCase):
    """Verifies that TranslationBackend protocol behaves as a strict runtime contract."""

    def test_nmt_backend_implements_protocol(self):
        backend = NMTBackend()
        self.assertTrue(isinstance(backend, TranslationBackend))
        self.assertEqual(backend.name, "nmt")
        self.assertTrue(hasattr(backend, "is_available"))
        self.assertTrue(hasattr(backend, "is_ready"))
        self.assertTrue(hasattr(backend, "translate"))

    def test_llm_backend_implements_protocol(self):
        backend = LLMBackend()
        self.assertTrue(isinstance(backend, TranslationBackend))
        self.assertEqual(backend.name, "llm")
        self.assertTrue(hasattr(backend, "is_available"))
        self.assertTrue(hasattr(backend, "is_ready"))
        self.assertTrue(hasattr(backend, "translate"))

    def test_custom_backend_runtime_checkable(self):
        mock_backend = MockCustomBackend()
        self.assertTrue(isinstance(mock_backend, TranslationBackend))

        incomplete = IncompleteBackend()
        self.assertFalse(isinstance(incomplete, TranslationBackend))

    def test_nmt_translate_method_success(self):
        backend = NMTBackend()
        with patch.object(backend, "translate_single", return_value="Translated text with [[N0]]"):
            res, elapsed = backend.translate("Source text with [[N0]]", "ja2en")
            self.assertEqual(res, "Translated text with [[N0]]")
            self.assertGreaterEqual(elapsed, 0.0)

    def test_nmt_translate_method_error_handling(self):
        backend = NMTBackend()
        logs = []
        with patch.object(backend, "translate_single", side_effect=RuntimeError("Engine failure")):
            res, elapsed = backend.translate("Failed text", "ja2en", log_cb=logs.append)
            self.assertIsNone(res)
            self.assertGreaterEqual(elapsed, 0.0)
            self.assertTrue(any("NMT backend error" in log for log in logs))

    def test_llm_translate_method_routes_to_translate_single(self):
        backend = LLMBackend()
        with patch.object(backend, "translate_single", return_value=("Translated LLM output", 0.42)) as mock_single:
            res, elapsed = backend.translate(
                text="Source text",
                direction="en2ja",
                placeholder_map={"[[N0]]": "123"},
                context="Context string",
            )
            self.assertEqual(res, "Translated LLM output")
            self.assertEqual(elapsed, 0.42)
            mock_single.assert_called_once_with(
                masked_text="Source text",
                number_map={"[[N0]]": "123"},
                direction="en2ja",
                context="Context string",
                log_cb=None,
            )

    def test_llm_is_ready_matches_is_available(self):
        backend = LLMBackend()
        with patch.object(backend, "is_available", return_value=True):
            self.assertTrue(backend.is_ready("ja2en"))
        with patch.object(backend, "is_available", return_value=False):
            self.assertFalse(backend.is_ready("ja2en"))


class TestEnginePolymorphicDispatch(unittest.TestCase):
    """Verifies that TranslationEngine uniformly dispatches to TranslationBackend instances."""

    def test_get_backend_by_mode(self):
        engine_nmt = TranslationEngine(mode=TranslationMode.FAST_NMT)
        self.assertIsInstance(engine_nmt.get_backend(), TranslationBackend)
        self.assertEqual(engine_nmt.get_backend().name, "nmt")

        engine_llm = TranslationEngine(mode=TranslationMode.PURE_LLM)
        self.assertIsInstance(engine_llm.get_backend(), TranslationBackend)
        self.assertEqual(engine_llm.get_backend().name, "llm")

    def test_engine_accepts_custom_backend_via_init(self):
        custom = MockCustomBackend()
        engine = TranslationEngine(backend=custom)
        self.assertIs(engine.get_backend(), custom)

    def test_engine_set_backend(self):
        engine = TranslationEngine(mode=TranslationMode.FAST_NMT)
        custom = MockCustomBackend()
        engine.set_backend(custom)
        self.assertIs(engine.get_backend(), custom)

    def test_translate_chunk_dispatches_to_custom_backend(self):
        custom = MockCustomBackend()
        engine = TranslationEngine(backend=custom)

        res = engine.translate_chunk("これはテストです。", "ja2en")
        self.assertIsInstance(res, TranslationResult)
        self.assertTrue(res.was_translated)
        self.assertFalse(res.was_reverted)
        self.assertEqual(res.source_backend, "custom_mock")
        self.assertEqual(res.text, "[CUSTOM: これはテストです。]")
        self.assertEqual(len(custom.call_history), 1)

    def test_translate_chunk_unready_nmt_raises_error(self):
        class UnreadyBackend(MockCustomBackend):
            name = "nmt"

            def is_ready(self, direction: str) -> bool:
                return False

        engine = TranslationEngine(mode=TranslationMode.FAST_NMT, backend=UnreadyBackend(ready=False))
        with self.assertRaises(TranslatorError) as ctx:
            engine.translate_chunk("これはテストです。", "ja2en")
        self.assertEqual(ctx.exception.code, ErrorCode.E08)

    def test_translate_chunk_backend_exception_triggers_graceful_revert(self):
        class ExplodingBackend(MockCustomBackend):
            name = "exploding"

            def translate(self, *args, **kwargs):
                raise RuntimeError("Boom!")

        engine = TranslationEngine(backend=ExplodingBackend())
        logs = []
        res = engine.translate_chunk("これはテストです。", "ja2en", log_cb=logs.append)
        self.assertIsInstance(res, TranslationResult)
        self.assertFalse(res.was_translated)
        self.assertTrue(res.was_reverted)
        self.assertEqual(res.text, "これはテストです。")
        self.assertEqual(res.source_backend, "exploding")
        self.assertTrue(any("translation error" in log for log in logs))


if __name__ == "__main__":
    unittest.main()
