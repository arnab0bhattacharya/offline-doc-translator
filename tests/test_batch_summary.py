import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from engine.queue_manager import JobStatus, TranslationJob
from gui.controllers.translation_controller import TranslationController
from gui.views.documents_view import DocumentsView


class TestBatchSummary(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_batch_summary_")
        self.controller = TranslationController()

    def tearDown(self):
        self.controller.shutdown()
        import shutil

        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_controller_batch_tracking_and_completion(self):
        job1 = TranslationJob(
            id="job1",
            input_path=os.path.join(self.test_dir, "doc1.docx"),
            output_path=os.path.join(self.test_dir, "doc1_ja2en.docx"),
            direction="ja2en",
            mode="fast_nmt",
            model_name="test",
            glossary={},
            status=JobStatus.QUEUED,
            progress=0.0,
            progress_message="Queued",
        )
        job2 = TranslationJob(
            id="job2",
            input_path=os.path.join(self.test_dir, "doc2.docx"),
            output_path=os.path.join(self.test_dir, "doc2_ja2en.docx"),
            direction="ja2en",
            mode="fast_nmt",
            model_name="test",
            glossary={},
            status=JobStatus.RUNNING,
            progress=50.0,
            progress_message="Translating",
        )

        self.controller.queue._jobs = [job1, job2]
        self.controller._current_batch_ids = {"job1", "job2"}

        # Not complete while QUEUED / RUNNING
        self.assertFalse(self.controller.is_batch_complete())

        # Finish job 1, job 2 still running
        job1.status = JobStatus.COMPLETED
        self.assertFalse(self.controller.is_batch_complete())

        # Finish job 2
        job2.status = JobStatus.FAILED
        self.assertTrue(self.controller.is_batch_complete())

    def test_controller_batch_summary_aggregation(self):
        out1 = os.path.join(self.test_dir, "sub", "doc1_ja2en.docx")
        out2 = os.path.join(self.test_dir, "sub", "doc2_ja2en.docx")
        os.makedirs(os.path.dirname(out1), exist_ok=True)
        with open(out1, "w") as f:
            f.write("test")
        with open(out2, "w") as f:
            f.write("test")

        rev1 = f"{out1}.needs_review.log"
        with open(rev1, "w") as f:
            f.write("Review item")

        job1 = TranslationJob(
            id="j1",
            input_path="doc1.docx",
            output_path=out1,
            direction="ja2en",
            mode="fast_nmt",
            model_name="test",
            glossary={},
            status=JobStatus.COMPLETED,
            progress=100.0,
            progress_message="Done",
            started_at=100.0,
            completed_at=160.0,
            review_log_path=rev1,
            result={"translated": 100, "reverted": 2, "skipped": 5, "total": 107},
        )
        job2 = TranslationJob(
            id="j2",
            input_path="doc2.docx",
            output_path=out2,
            direction="ja2en",
            mode="fast_nmt",
            model_name="test",
            glossary={},
            status=JobStatus.COMPLETED,
            progress=100.0,
            progress_message="Done",
            started_at=110.0,
            completed_at=180.0,
            result={"translated": 50, "reverted": 0, "skipped": 1, "total": 51},
        )

        self.controller.queue._jobs = [job1, job2]
        self.controller._current_batch_ids = {"j1", "j2"}

        summary = self.controller.get_batch_summary()
        self.assertEqual(summary["total_files"], 2)
        self.assertEqual(summary["completed_count"], 2)
        self.assertEqual(summary["failed_count"], 0)
        self.assertEqual(summary["translated_chunks"], 150)
        self.assertEqual(summary["reverted_chunks"], 2)
        self.assertEqual(summary["skipped_chunks"], 6)
        self.assertEqual(summary["elapsed_seconds"], 80.0)
        self.assertEqual(summary["formatted_time"], "1m 20s")
        self.assertEqual(len(summary["review_log_paths"]), 1)
        self.assertEqual(os.path.normpath(summary["common_dir"]), os.path.normpath(os.path.dirname(out1)))
        self.assertIn("Files: 2 completed, 0 failed", summary["formatted_text"])
        self.assertIn("Chunks: 150 translated, 2 reverted, 6 skipped", summary["formatted_text"])
        self.assertIn("Total time: 1m 20s", summary["formatted_text"])
        self.assertIn("Review logs: 1 file has items to review", summary["formatted_text"])

    def test_documents_view_batch_summary_card_lifecycle(self):
        view = MagicMock(spec=DocumentsView)
        view.controller = self.controller
        view._current_batch_ids = {"j1", "j2"}
        view._batch_summary_shown = False
        view._last_batch_summary = None

        view.batch_summary_card = MagicMock()
        view.summary_files_lbl = MagicMock()
        view.summary_chunks_lbl = MagicMock()
        view.summary_time_lbl = MagicMock()
        view.summary_review_lbl = MagicMock()
        view.summary_open_folder_btn = MagicMock()
        view.summary_copy_btn = MagicMock()
        view.bottom_box = MagicMock()

        summary_data = {
            "completed_count": 3,
            "failed_count": 0,
            "cancelled_count": 0,
            "translated_chunks": 42,
            "reverted_chunks": 1,
            "skipped_chunks": 2,
            "formatted_time": "45s",
            "review_log_paths": ["rev.log"],
            "output_paths": ["out.docx"],
            "common_dir": "C:/out",
            "formatted_text": "📊 Batch Summary\nFiles: 3 completed",
        }
        with patch.object(self.controller, "get_batch_summary", return_value=summary_data):
            DocumentsView._show_batch_summary(view)

            view.summary_files_lbl.configure.assert_called_with(text="Files: 3 completed, 0 failed")
            view.summary_chunks_lbl.configure.assert_called_with(text="Chunks: 42 translated, 1 reverted, 2 skipped")
            view.summary_time_lbl.configure.assert_called_with(text="Total time: 45s")
            from gui.theme import THEME

            view.summary_review_lbl.configure.assert_called_with(
                text="Review logs: 1 file has items to review", text_color=THEME["warning"]
            )
            view.batch_summary_card.pack.assert_called_once()

            # Test dismiss
            DocumentsView._dismiss_batch_summary(view)
            view.batch_summary_card.pack_forget.assert_called_once()

            # Test copy
            view.clipboard_clear = MagicMock()
            view.clipboard_append = MagicMock()
            view.update = MagicMock()
            view.after = MagicMock()
            DocumentsView._copy_batch_summary(view)
            view.clipboard_clear.assert_called_once()
            view.clipboard_append.assert_called_with(summary_data["formatted_text"])
            view.summary_copy_btn.configure.assert_called_with(text="✅  Copied!")
