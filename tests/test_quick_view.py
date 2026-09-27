"""
tests/test_quick_view.py
========================
Tests for QuickView dual-engine routing (MT vs. AI Translation),
Hardware Guard concurrency lock, and Custom Glossary preservation.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.queue_manager import JobStatus, TranslationJob
from gui.controllers.translation_controller import TranslationController
from gui.views.quick_view import (
    QuickView,
)


class TestQuickViewDualEngineAndGuard(unittest.TestCase):
    """Test QuickView dual engine dispatch and Hardware Guard lock."""

    def test_controller_is_busy_property(self):
        controller = TranslationController()
        # Empty queue -> not busy
        self.assertFalse(controller.is_busy)

        # Add a running job
        mock_job = MagicMock(spec=TranslationJob)
        mock_job.status = JobStatus.RUNNING
        controller.queue._jobs.append(mock_job)
        self.assertTrue(controller.is_busy)

        # Complete job -> not busy
        mock_job.status = JobStatus.COMPLETED
        self.assertFalse(controller.is_busy)

        # Queued job -> busy
        mock_job.status = JobStatus.QUEUED
        self.assertTrue(controller.is_busy)

    def test_set_locked_state_updates_ui(self):
        view = MagicMock(spec=QuickView)
        view.quick_translate_btn = MagicMock()
        view.quick_status = MagicMock()

        # Lock
        QuickView.set_locked_state(view, True, "financial_report.pptx")
        self.assertTrue(view._is_locked)
        view.quick_translate_btn.configure.assert_called_with(state="disabled", text="⏸  Document Active")
        status_text = view.quick_status.configure.call_args.kwargs["text"]
        self.assertIn("financial_report.pptx", status_text)
        self.assertIn("paused to protect RAM & CPU", status_text)

        # Unlock
        QuickView.set_locked_state(view, False)
        self.assertFalse(view._is_locked)
        view.quick_translate_btn.configure.assert_called_with(state="normal", text="▶   Translate Text")

    @patch("gui.views.quick_view.get_ollama_manager")
    def test_worker_mt_mode_evicts_ollama_and_uses_madlad(self, mock_get_mgr):
        view = MagicMock(spec=QuickView)
        view.after = MagicMock(side_effect=lambda delay, fn, *args: fn(*args))
        view._set_quick_result = MagicMock()
        view.quick_translate_btn = MagicMock()
        view.glossary_text = MagicMock()
        view.glossary_text.get.return_value = "API -> インターフェース\n"

        mock_mgr = MagicMock()
        mock_get_mgr.return_value = mock_mgr

        mock_madlad = MagicMock()
        mock_madlad.translate.return_value = ("これは[[GLOSSARY_A]]と[[N0]]です。", 0.35)
        view.madlad_backend = mock_madlad

        # Run worker with is_ai=False (MT Mode)
        QuickView._quick_translate_worker(view, "This is API and 100.", "en2ja", is_ai_or_model=False)

        # 1. Ollama models must be evicted from RAM
        mock_mgr.unload_all_models.assert_called_once()

        # 2. MADLAD backend must be invoked
        mock_madlad.translate.assert_called_once()
        kwargs = mock_madlad.translate.call_args.kwargs
        self.assertIn("[[GLOSSARY_A]]", kwargs["text"])
        self.assertIn("[[N0]]", kwargs["text"])

        # 3. Result unmasked
        view._set_quick_result.assert_called_once()
        final_text, status = view._set_quick_result.call_args[0]
        self.assertEqual(final_text, "これはインターフェースと100です。")
        self.assertIn("MADLAD-400 3B", status)
        self.assertIn("1 glossary term", status)

    @patch("gui.views.quick_view.check_ollama_status", return_value=True)
    def test_worker_ai_mode_evicts_madlad_and_uses_ollama(self, mock_check_ollama):
        view = MagicMock(spec=QuickView)
        view.after = MagicMock(side_effect=lambda delay, fn, *args: fn(*args))
        view._set_quick_result = MagicMock()
        view.quick_translate_btn = MagicMock()
        view.glossary_text = MagicMock()
        view.glossary_text.get.return_value = ""

        mock_madlad = MagicMock()
        view.madlad_backend = mock_madlad

        mock_llm = MagicMock()
        mock_llm.translate_single.return_value = ("こんにちは世界", 0.8)

        with patch("gui.views.quick_view.LLMBackend", return_value=mock_llm):
            QuickView._quick_translate_worker(
                view, "Hello world", "en2ja", is_ai_or_model=True, model="gemma4:e2b-it-qat"
            )

        # 1. MADLAD must be evicted from RAM
        mock_madlad.unload.assert_called_once()

        # 2. LLM backend must be invoked
        mock_llm.translate_single.assert_called_once()

        # 3. Result verified
        view._set_quick_result.assert_called_once()
        final_text, status = view._set_quick_result.call_args[0]
        self.assertEqual(final_text, "こんにちは世界")
        self.assertIn("gemma4:e2b-it-qat", status)

    def test_worker_aborts_when_controller_is_busy(self):
        view = MagicMock(spec=QuickView)
        view.after = MagicMock(side_effect=lambda delay, fn, *args: fn(*args))
        view._set_quick_result = MagicMock()
        view.quick_translate_btn = MagicMock()

        mock_controller = MagicMock()
        mock_controller.is_busy = True  # Document batch actively running
        view.controller = mock_controller

        QuickView._quick_translate_worker(view, "Some text", "en2ja", is_ai_or_model=False)

        view._set_quick_result.assert_called_once()
        args = view._set_quick_result.call_args[0]
        self.assertEqual(args[0], "")
        self.assertIn("Document translation started", args[1])


if __name__ == "__main__":
    unittest.main()
