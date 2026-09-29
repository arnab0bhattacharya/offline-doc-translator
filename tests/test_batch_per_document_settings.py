"""
tests/test_batch_per_document_settings.py
=========================================
Tests for per-document translation direction and engine settings,
StagedFileList management, JobRow visual badges, and TranslationController
batch dispatching with per-file specifications.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import customtkinter as ctk

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core import TranslationMode
from engine.queue_manager import JobStatus, TranslationJob
from gui.controllers.translation_controller import TranslationController
from gui.widgets.job_row import JobRow
from gui.widgets.staged_file_list import StagedFileList, StagedItem


class TestStagedFileListAndSpecs(unittest.TestCase):
    """Tests for StagedFileList data models, per-file specs, and presets."""

    @classmethod
    def setUpClass(cls):
        cls.root = ctk.CTk()
        cls.root.withdraw()

    @classmethod
    def tearDownClass(cls):
        cls.root.destroy()

    def setUp(self):
        self.temp_files = []
        for name in ["doc1.docx", "doc2.pptx", "doc3.xlsx"]:
            path = os.path.abspath(f"temp_test_{name}")
            with open(path, "w", encoding="utf-8") as f:
                f.write("dummy")
            self.temp_files.append(path)

    def tearDown(self):
        for path in self.temp_files:
            if os.path.exists(path):
                os.remove(path)

    def test_staged_item_defaults(self):
        item = StagedItem(path="dummy.docx")
        self.assertEqual(item.path, "dummy.docx")
        self.assertEqual(item.direction, "ja2en")
        self.assertEqual(item.mode, "machine_translation")

    def test_add_files_inherits_default_preset(self):
        staged = StagedFileList(self.root)
        staged.set_default_preset(
            direction="en2ja",
            mode="ai_translation",
            model_name="gemma4:12b",
        )
        added = staged.add_files(self.temp_files[:2])
        self.assertEqual(added, 2)

        specs = staged.get_staged_specs()
        self.assertEqual(len(specs), 2)
        self.assertEqual(specs[0]["direction"], "en2ja")
        self.assertEqual(specs[0]["mode"], "ai_translation")
        self.assertEqual(specs[0]["model_name"], "gemma4:12b")

        # Test backward-compatible get_files and selected_files
        self.assertEqual(staged.get_files(), self.temp_files[:2])
        self.assertEqual(staged.selected_files, self.temp_files[:2])

    def test_apply_preset_to_all(self):
        staged = StagedFileList(self.root)
        staged.add_files(self.temp_files)
        # Initially MT ja2en
        for spec in staged.get_staged_specs():
            self.assertEqual(spec["direction"], "ja2en")
            self.assertEqual(spec["mode"], "machine_translation")

        # Apply preset to all
        staged.apply_preset_to_all(
            direction="en2ja",
            mode="ai_translation",
            model_name="gemma4:27b",
        )
        for spec in staged.get_staged_specs():
            self.assertEqual(spec["direction"], "en2ja")
            self.assertEqual(spec["mode"], "ai_translation")
            self.assertEqual(spec["model_name"], "gemma4:27b")

    def test_per_document_inline_modification(self):
        staged = StagedFileList(self.root)
        staged.add_files(self.temp_files[:2])

        # Manually change item 0 to EN->JA and AI
        staged.staged_items[0].direction = "en2ja"
        staged.staged_items[0].mode = "ai_translation"

        specs = staged.get_staged_specs()
        self.assertEqual(specs[0]["direction"], "en2ja")
        self.assertEqual(specs[0]["mode"], "ai_translation")
        # Item 1 unchanged
        self.assertEqual(specs[1]["direction"], "ja2en")
        self.assertEqual(specs[1]["mode"], "machine_translation")

    def test_remove_file_and_clear(self):
        staged = StagedFileList(self.root)
        staged.add_files(self.temp_files)
        self.assertEqual(len(staged.get_files()), 3)

        staged.remove_file(self.temp_files[1])
        self.assertEqual(len(staged.get_files()), 2)
        self.assertNotIn(self.temp_files[1], staged.get_files())

        staged.clear()
        self.assertEqual(len(staged.get_files()), 0)
        self.assertEqual(len(staged.get_staged_specs()), 0)


class TestJobRowBadges(unittest.TestCase):
    """Tests that JobRow renders direction and engine badges correctly."""

    @classmethod
    def setUpClass(cls):
        try:
            cls.root = ctk.CTk()
            cls.root.withdraw()
        except Exception:
            cls.root = None

    @classmethod
    def tearDownClass(cls):
        if cls.root:
            try:
                cls.root.destroy()
            except Exception:
                pass

    def setUp(self):
        if self.root is None:
            self.skipTest("Tkinter display/library unavailable in test environment")

    def test_job_row_badges_mt(self):
        job = TranslationJob(
            id="job-mt-1",
            input_path="report.docx",
            output_path="report.translated.docx",
            direction="ja2en",
            mode=TranslationMode.MACHINE_TRANSLATION,
            model_name="madlad400-3b-mt",
            glossary={},
            status=JobStatus.QUEUED,
            progress=0.0,
            progress_message="Queued",
        )
        row = JobRow(self.root, job=job)
        self.assertIsNotNone(row.direction_badge)
        self.assertIsNotNone(row.engine_badge)
        self.assertIn("JA → EN", row.direction_badge.cget("text"))
        self.assertIn("⚡ MT", row.engine_badge.cget("text"))
        # Test dict alias
        self.assertEqual(row["direction_badge"], row.direction_badge)
        self.assertEqual(row["engine_badge"], row.engine_badge)

    def test_job_row_badges_ai(self):
        job = TranslationJob(
            id="job-ai-1",
            input_path="presentation.pptx",
            output_path="presentation.translated.pptx",
            direction="en2ja",
            mode=TranslationMode.AI_TRANSLATION,
            model_name="gemma4:e2b-it-qat",
            glossary={},
            status=JobStatus.RUNNING,
            progress=30.0,
            progress_message="Translating",
        )
        row = JobRow(self.root, job=job)
        self.assertIn("EN → JA", row.direction_badge.cget("text"))
        self.assertIn("🤖 gemma4", row.engine_badge.cget("text"))


class TestControllerBatchSpecs(unittest.TestCase):
    """Tests that TranslationController.start_batch handles mixed per-file specs."""

    @patch("os.path.exists", return_value=True)
    def test_start_batch_with_specs(self, mock_exists):
        controller = TranslationController()
        controller.add_job = MagicMock(side_effect=lambda **kwargs: (f"job-{len(kwargs)}", kwargs["output_path"]))

        specs = [
            {
                "path": "test1.docx",
                "direction": "ja2en",
                "mode": "machine_translation",
                "model_name": "madlad",
            },
            {
                "path": "test2.pptx",
                "direction": "en2ja",
                "mode": "ai_translation",
                "model_name": "gemma4:12b",
            },
        ]

        dispatched = controller.start_batch(
            input_files=specs,
            direction="ja2en",
            mode=TranslationMode.MACHINE_TRANSLATION,
            model_name="default_model",
        )

        self.assertEqual(len(dispatched), 2)
        self.assertEqual(controller.add_job.call_count, 2)

        call_0_kwargs = controller.add_job.call_args_list[0].kwargs
        self.assertEqual(call_0_kwargs["input_path"], "test1.docx")
        self.assertEqual(call_0_kwargs["direction"], "ja2en")
        self.assertEqual(call_0_kwargs["mode"], "machine_translation")

        call_1_kwargs = controller.add_job.call_args_list[1].kwargs
        self.assertEqual(call_1_kwargs["input_path"], "test2.pptx")
        self.assertEqual(call_1_kwargs["direction"], "en2ja")
        self.assertEqual(call_1_kwargs["mode"], "ai_translation")
        self.assertEqual(call_1_kwargs["model_name"], "gemma4:12b")

    @patch("os.path.exists", return_value=True)
    def test_start_batch_with_legacy_strings(self, mock_exists):
        controller = TranslationController()
        controller.add_job = MagicMock(side_effect=lambda **kwargs: (f"job-{len(kwargs)}", kwargs["output_path"]))

        files = ["test1.docx", "test2.pptx"]
        dispatched = controller.start_batch(
            input_files=files,
            direction="en2ja",
            mode=TranslationMode.MACHINE_TRANSLATION,
            model_name="madlad",
        )

        self.assertEqual(len(dispatched), 2)
        for call_args in controller.add_job.call_args_list:
            self.assertEqual(call_args.kwargs["direction"], "en2ja")
            self.assertEqual(call_args.kwargs["mode"], TranslationMode.MACHINE_TRANSLATION)


class TestDocumentsViewStagingIntegration(unittest.TestCase):
    """Tests that DocumentsView correctly synchronizes presets and updates the live summary."""

    def test_live_summary_formatting_mixed_specs(self):
        from gui.views.documents_view import DocumentsView

        view = MagicMock(spec=DocumentsView)
        view.start_btn = MagicMock()
        view.staged_summary_label = MagicMock()
        view.staged_list = MagicMock()

        # Simulate 2 MT JA->EN and 1 AI EN->JA
        view.staged_list.get_staged_specs.return_value = [
            {"path": "a.docx", "direction": "ja2en", "mode": "machine_translation"},
            {"path": "b.pptx", "direction": "ja2en", "mode": "machine_translation"},
            {"path": "c.xlsx", "direction": "en2ja", "mode": "ai_translation"},
        ]

        DocumentsView._on_staged_files_changed(view, ["a.docx", "b.pptx", "c.xlsx"])

        view.start_btn.configure.assert_called_with(text="▶   Start Translation (3)")
        view.staged_summary_label.configure.assert_called_once()
        summary_text = view.staged_summary_label.configure.call_args.kwargs["text"]
        self.assertIn("Ready to queue 3 documents:", summary_text)
        self.assertIn("2 via ⚡ MT (JA→EN)", summary_text)
        self.assertIn("1 via 🤖 AI (EN→JA)", summary_text)

    def test_live_summary_cleared_on_empty(self):
        from gui.views.documents_view import DocumentsView

        view = MagicMock(spec=DocumentsView)
        view.start_btn = MagicMock()
        view.staged_summary_label = MagicMock()

        DocumentsView._on_staged_files_changed(view, [])
        view.start_btn.configure.assert_called_with(text="▶   Start Translation")
        view.staged_summary_label.configure.assert_called_with(text="")

    def test_sync_presets_to_staged_defaults(self):
        from gui.theme import PINNED_OLLAMA_MODEL
        from gui.views.documents_view import DocumentsView

        view = MagicMock(spec=DocumentsView)
        view.direction_var = MagicMock(get=lambda: "en2ja")
        view.mode_var = MagicMock(get=lambda: "ai_translation")
        view.model_var = MagicMock(get=lambda: PINNED_OLLAMA_MODEL)
        view.staged_list = MagicMock()

        DocumentsView._sync_card2_presets_to_staged_defaults(view)
        view.staged_list.set_default_preset.assert_called_with("en2ja", "ai_translation", PINNED_OLLAMA_MODEL)

    def test_apply_preset_to_all_staged(self):
        from gui.theme import PINNED_OLLAMA_MODEL
        from gui.views.documents_view import DocumentsView

        view = MagicMock(spec=DocumentsView)
        view.direction_var = MagicMock(get=lambda: "ja2en")
        view.mode_var = MagicMock(get=lambda: "machine_translation")
        view.model_var = MagicMock(get=lambda: PINNED_OLLAMA_MODEL)
        view.staged_list = MagicMock()
        view.log = MagicMock()

        DocumentsView._apply_preset_to_all_staged(view)
        view.staged_list.apply_preset_to_all.assert_called_with("ja2en", "machine_translation", PINNED_OLLAMA_MODEL)


if __name__ == "__main__":
    unittest.main()
