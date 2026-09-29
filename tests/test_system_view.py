"""
tests/test_system_view.py
=========================
Unit tests for SystemView diagnostics, pinned Gemma 4 download management,
cache policy selection & persistence, and review log access.
"""

import unittest
from unittest.mock import MagicMock, patch

from engine.cache import CachePolicy
from gui.controllers.translation_controller import TranslationController
from gui.theme import PINNED_OLLAMA_MODEL
from gui.views.system_view import (
    CACHE_POLICY_MAP,
    REVERSE_CACHE_POLICY_MAP,
    SystemView,
)


class TestSystemViewUnit(unittest.TestCase):
    """Verifies SystemView logic and UI state transitions."""

    def setUp(self):
        self.controller = TranslationController()

    @patch("gui.views.system_view.check_ollama_status", return_value=True)
    @patch("gui.views.system_view.list_installed_models", return_value=["llama3:8b"])
    @patch("gui.views.system_view.get_ollama_manager")
    def test_refresh_ollama_status_pinned_model_missing(self, mock_mgr, mock_list, mock_status):
        mock_mgr.return_value.get_loaded_models.return_value = []
        mock_mgr.return_value.spawned_by_app = False

        view = MagicMock(spec=SystemView)
        view.sys_ollama_download_btn = MagicMock()
        view.sys_ollama_start_btn = MagicMock()
        view.sys_ollama_refresh_btn = MagicMock()
        view.sys_ollama_free_btn = MagicMock()
        view.sys_ollama_stop_btn = MagicMock()
        view.sys_ollama_msg = MagicMock()
        view.sys_ollama_desc = MagicMock()
        view.on_ollama_status = None

        SystemView.refresh_ollama_status(view)

        # Download button should prompt to download pinned model
        view.sys_ollama_download_btn.configure.assert_called_with(text="⬇   Download Gemma 4 Model (~1.6 GB)")
        view.sys_ollama_download_btn.pack.assert_called_with(side="left", padx=(0, 8))

    @patch("gui.views.system_view.check_ollama_status", return_value=True)
    @patch("gui.views.system_view.list_installed_models", return_value=[PINNED_OLLAMA_MODEL])
    @patch("gui.views.system_view.get_ollama_manager")
    def test_refresh_ollama_status_pinned_model_installed(self, mock_mgr, mock_list, mock_status):
        mock_mgr.return_value.get_loaded_models.return_value = [PINNED_OLLAMA_MODEL]
        mock_mgr.return_value.spawned_by_app = True

        view = MagicMock(spec=SystemView)
        view.sys_ollama_download_btn = MagicMock()
        view.sys_ollama_start_btn = MagicMock()
        view.sys_ollama_refresh_btn = MagicMock()
        view.sys_ollama_free_btn = MagicMock()
        view.sys_ollama_stop_btn = MagicMock()
        view.sys_ollama_msg = MagicMock()
        view.sys_ollama_desc = MagicMock()
        view.on_ollama_status = None

        SystemView.refresh_ollama_status(view)

        # Download button should offer re-download / update
        view.sys_ollama_download_btn.configure.assert_called_with(text="↻  Re-download / Update Model")
        view.sys_ollama_download_btn.pack.assert_called_with(side="left", padx=(0, 8))
        view.sys_ollama_stop_btn.pack.assert_called_with(side="left", padx=(0, 8))

    def test_cache_policy_mapping(self):
        self.assertEqual(CACHE_POLICY_MAP["Encrypted (Default)"], CachePolicy.ENCRYPTED_PERSISTENT)
        self.assertEqual(CACHE_POLICY_MAP["In-Memory (Privacy)"], CachePolicy.MEMORY_ONLY)
        self.assertEqual(CACHE_POLICY_MAP["Plaintext (Compatibility)"], CachePolicy.PLAINTEXT_PERSISTENT)
        self.assertEqual(REVERSE_CACHE_POLICY_MAP[CachePolicy.ENCRYPTED_PERSISTENT], "Encrypted (Default)")

    @patch("gui.views.system_view.save_app_settings")
    def test_on_cache_policy_changed(self, mock_save):
        view = MagicMock(spec=SystemView)
        view.controller = self.controller
        view.sys_cache_policy_desc = MagicMock()
        view.sys_cache_msg = MagicMock()
        view._get_policy_explanation = SystemView._get_policy_explanation

        SystemView._on_cache_policy_changed(view, "In-Memory (Privacy)")

        self.assertEqual(self.controller.default_cache_policy, CachePolicy.MEMORY_ONLY)
        mock_save.assert_called_with({"cache_policy": "memory_only"})

    @patch("os.path.exists", return_value=True)
    @patch("os.path.getsize", return_value=128)
    @patch("sys.platform", "win32")
    @patch("os.startfile")
    def test_open_review_log_gui_from_controller(self, mock_startfile, mock_getsize, mock_exists):
        self.controller.completed_review_logs = ["C:\\test\\file.docx.needs_review.log"]
        self.controller.last_review_log = "C:\\test\\file.docx.needs_review.log"

        view = MagicMock(spec=SystemView)
        view.controller = self.controller

        SystemView._open_review_log_gui(view)
        mock_startfile.assert_called_with("C:\\test\\file.docx.needs_review.log")


if __name__ == "__main__":
    unittest.main()
