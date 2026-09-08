"""
tests/test_run_job.py
=====================
Unit tests for the shared execute_translation job runner.
"""

import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock

from engine.core import TranslationMode
from engine.errors import TranslatorError, ErrorCode
from engine.run_job import execute_translation


class TestRunJob(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_run_job_")

    def tearDown(self):
        import shutil
        shutil.rmtree(self.test_dir, ignore_errors=True)

    @patch("engine.run_job.get_handler")
    @patch("engine.run_job.TranslationEngine")
    @patch("engine.run_job.run_nmt_preflight")
    @patch("engine.run_job.run_preflight")
    def test_execute_translation_fast_nmt(
        self, mock_preflight, mock_nmt_preflight, mock_engine_cls, mock_get_handler
    ):
        mock_handler = MagicMock()
        mock_handler.translate.return_value = {"total": 5, "translated": 5, "reverted": 0, "skipped": 0}
        mock_get_handler.return_value = mock_handler

        in_path = os.path.join(self.test_dir, "doc.docx")
        out_path = os.path.join(self.test_dir, "doc_en.docx")

        stats = execute_translation(
            input_path=in_path,
            output_path=out_path,
            direction="ja2en",
            mode=TranslationMode.FAST_NMT,
            model_name="gemma4:e2b-it-qat",
            glossary={"A": "B"},
        )

        self.assertEqual(stats["total"], 5)
        mock_preflight.assert_called_once_with(
            model_name="gemma4:e2b-it-qat",
            input_path=in_path,
            output_path=out_path,
            check_model=False,
            require_ollama=False,
        )
        mock_nmt_preflight.assert_called_once_with("ja2en")
        mock_get_handler.assert_called_once_with(".docx", mock_engine_cls.return_value)
        mock_handler.translate.assert_called_once()

    @patch("engine.run_job.get_handler")
    @patch("engine.run_job.TranslationEngine")
    @patch("engine.run_job.run_nmt_preflight")
    @patch("engine.run_job.run_preflight")
    def test_execute_translation_pure_llm(
        self, mock_preflight, mock_nmt_preflight, mock_engine_cls, mock_get_handler
    ):
        mock_handler = MagicMock()
        mock_handler.translate.return_value = {"total": 3, "translated": 3, "reverted": 0, "skipped": 0}
        mock_get_handler.return_value = mock_handler

        in_path = os.path.join(self.test_dir, "doc.pptx")
        out_path = os.path.join(self.test_dir, "doc_en.pptx")

        stats = execute_translation(
            input_path=in_path,
            output_path=out_path,
            direction="en2ja",
            mode="pure_llm",  # test string conversion
            model_name="gemma4:e2b-it-qat",
            glossary={},
        )

        self.assertEqual(stats["total"], 3)
        mock_preflight.assert_called_once_with(
            model_name="gemma4:e2b-it-qat",
            input_path=in_path,
            output_path=out_path,
            check_model=True,
            require_ollama=True,
        )
        mock_nmt_preflight.assert_not_called()
        mock_get_handler.assert_called_once_with(".pptx", mock_engine_cls.return_value)

    @patch("engine.run_job.run_preflight")
    def test_execute_translation_preflight_failure(self, mock_preflight):
        mock_preflight.side_effect = TranslatorError(ErrorCode.E04, "File not found")

        in_path = os.path.join(self.test_dir, "nonexistent.docx")
        out_path = os.path.join(self.test_dir, "out.docx")

        with self.assertRaises(TranslatorError) as ctx:
            execute_translation(
                input_path=in_path,
                output_path=out_path,
                direction="ja2en",
                mode=TranslationMode.FAST_NMT,
                model_name="test",
                glossary={},
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E04)


    @patch("engine.run_job.get_handler")
    @patch("engine.run_job.TranslationEngine")
    @patch("engine.run_job.run_preflight")
    def test_execute_translation_cleans_old_review_log(
        self, mock_preflight, mock_engine_cls, mock_get_handler
    ):
        mock_handler = MagicMock()
        mock_handler.translate.return_value = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        mock_get_handler.return_value = mock_handler

        in_path = os.path.join(self.test_dir, "doc.xlsx")
        out_path = os.path.join(self.test_dir, "doc_en.xlsx")
        review_log = f"{out_path}.needs_review.log"

        with open(review_log, "w", encoding="utf-8") as f:
            f.write("old review data\n")

        self.assertTrue(os.path.exists(review_log))

        execute_translation(
            input_path=in_path,
            output_path=out_path,
            direction="ja2en",
            mode=TranslationMode.FAST_NMT,
            model_name="test",
            glossary={},
        )

        # Before translate runs, the old review log should have been removed
        self.assertFalse(os.path.exists(review_log))

    def test_execute_translation_cancelled_before_run(self):
        import threading
        evt = threading.Event()
        evt.set()

        in_path = os.path.join(self.test_dir, "doc.docx")
        out_path = os.path.join(self.test_dir, "doc_en.docx")

        with self.assertRaises(TranslatorError) as ctx:
            execute_translation(
                input_path=in_path,
                output_path=out_path,
                direction="ja2en",
                mode=TranslationMode.FAST_NMT,
                model_name="test",
                glossary={},
                cancel_event=evt,
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E09)


if __name__ == "__main__":
    unittest.main()
