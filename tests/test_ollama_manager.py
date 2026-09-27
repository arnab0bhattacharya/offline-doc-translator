"""
tests/test_ollama_manager.py
============================
Unit tests for engine/ollama_manager.py.
Validates binary discovery, background process launching, timeout handling,
model eviction (keep_alive: 0), process termination, and exit cleanup.
"""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.ollama_manager import (
    OllamaManager,
    find_ollama_binary,
    get_ollama_manager,
)


class TestOllamaManager(unittest.TestCase):
    def setUp(self):
        self.manager = OllamaManager(ollama_url="http://localhost:11434")

    # ── Binary Discovery ──

    @patch("shutil.which", return_value="C:\\tools\\ollama.exe")
    def test_find_ollama_binary_in_path(self, mock_which):
        path = find_ollama_binary()
        self.assertIsNotNone(path)
        self.assertIn("ollama.exe", path)

    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile")
    def test_find_ollama_binary_windows_fallback(self, mock_isfile, mock_which):
        def isfile_side_effect(path):
            return "Programs\\Ollama\\ollama.exe" in path

        mock_isfile.side_effect = isfile_side_effect
        with patch("sys.platform", "win32"):
            path = find_ollama_binary()
            self.assertIsNotNone(path)
            self.assertIn("ollama.exe", path)

    @patch("shutil.which", return_value=None)
    @patch("os.path.isfile", return_value=False)
    def test_find_ollama_binary_not_found(self, mock_isfile, mock_which):
        path = find_ollama_binary()
        self.assertIsNone(path)

    # ── Service Startup & Concurrency ──

    @patch("engine.ollama_manager.check_ollama_status", return_value=True)
    def test_start_service_already_running(self, mock_status):
        success, msg = self.manager.start_service()
        self.assertTrue(success)
        self.assertIn("already running", msg)
        self.assertFalse(self.manager.spawned_by_app)

    @patch("engine.ollama_manager.find_ollama_binary", return_value=None)
    @patch("engine.ollama_manager.check_ollama_status", return_value=False)
    def test_start_service_binary_missing(self, mock_status, mock_find):
        success, msg = self.manager.start_service()
        self.assertFalse(success)
        self.assertIn("executable not found", msg)

    @patch("engine.ollama_manager.find_ollama_binary", return_value="C:\\Ollama\\ollama.exe")
    @patch("subprocess.Popen")
    @patch("engine.ollama_manager.check_ollama_status")
    def test_start_service_successful_spawn(self, mock_status, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        # False first, then True
        mock_status.side_effect = [False, False, True]

        success, msg = self.manager.start_service(timeout=3.0)
        self.assertTrue(success)
        self.assertTrue(self.manager.spawned_by_app)
        self.assertIn("started successfully", msg)
        mock_popen.assert_called_once()

    @patch("engine.ollama_manager.find_ollama_binary", return_value="C:\\Ollama\\ollama.exe")
    @patch("subprocess.Popen")
    @patch("engine.ollama_manager.check_ollama_status", return_value=False)
    def test_start_service_process_crashes_immediately(self, mock_status, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 1
        mock_proc.returncode = 1
        mock_popen.return_value = mock_proc

        success, msg = self.manager.start_service(timeout=2.0)
        self.assertFalse(success)
        self.assertIn("exited immediately with code 1", msg)
        self.assertFalse(self.manager.spawned_by_app)

    @patch("engine.ollama_manager.find_ollama_binary", return_value="C:\\Ollama\\ollama.exe")
    @patch("subprocess.Popen")
    @patch("engine.ollama_manager.check_ollama_status", return_value=False)
    def test_start_service_timeout(self, mock_status, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        success, msg = self.manager.start_service(timeout=0.8)
        self.assertFalse(success)
        self.assertIn("failed to respond within", msg)

    @patch("engine.ollama_manager.find_ollama_binary", return_value="C:\\Ollama\\ollama.exe")
    @patch("subprocess.Popen")
    @patch("engine.ollama_manager.check_ollama_status")
    def test_start_service_concurrent_calls(self, mock_status, mock_popen, mock_find):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        mock_popen.return_value = mock_proc

        mock_status.side_effect = [False, False, True, True]

        # Call 1
        s1, _ = self.manager.start_service(timeout=2.0)
        self.assertTrue(s1)

        # Call 2 when already running
        s2, msg2 = self.manager.start_service(timeout=2.0)
        self.assertTrue(s2)
        self.assertIn("already running", msg2)
        # Verify Popen was only called once
        self.assertEqual(mock_popen.call_count, 1)

    # ── Model Memory Eviction (keep_alive: 0) ──

    @patch("requests.get")
    def test_get_loaded_models(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "models": [
                {"name": "gemma4:e2b-it-qat", "size": 1800000000},
                {"name": "llama3:8b", "size": 4500000000},
            ]
        }
        mock_get.return_value = mock_resp

        models = self.manager.get_loaded_models()
        self.assertEqual(models, ["gemma4:e2b-it-qat", "llama3:8b"])

    @patch("requests.post")
    def test_unload_model_generate_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_post.return_value = mock_resp

        res = self.manager.unload_model("gemma4:e2b-it-qat")
        self.assertTrue(res)
        mock_post.assert_called_with(
            "http://localhost:11434/api/generate",
            json={"model": "gemma4:e2b-it-qat", "prompt": "", "keep_alive": 0},
            timeout=5.0,
        )

    @patch("requests.post")
    def test_unload_model_fallback_chat(self, mock_post):
        resp_gen_fail = MagicMock(status_code=404)
        resp_chat_ok = MagicMock(status_code=200)
        mock_post.side_effect = [resp_gen_fail, resp_chat_ok]

        res = self.manager.unload_model("gemma4:e2b-it-qat")
        self.assertTrue(res)
        self.assertEqual(mock_post.call_count, 2)

    @patch.object(OllamaManager, "get_loaded_models", return_value=["model-a", "model-b"])
    @patch.object(OllamaManager, "unload_model", return_value=True)
    def test_unload_all_models(self, mock_unload, mock_get_loaded):
        unloaded = self.manager.unload_all_models()
        self.assertEqual(unloaded, ["model-a", "model-b"])
        self.assertEqual(mock_unload.call_count, 2)

    # ── Shutdown & Process Termination ──

    @patch.object(OllamaManager, "unload_all_models")
    def test_stop_service_when_not_spawned_by_app(self, mock_unload):
        self.manager._spawned_by_app = False
        success, msg = self.manager.stop_service()
        self.assertFalse(success)
        self.assertIn("not launched by this application", msg)
        mock_unload.assert_called_once()

    @patch.object(OllamaManager, "unload_all_models")
    def test_stop_service_when_spawned_by_app(self, mock_unload):
        mock_proc = MagicMock()
        mock_proc.poll.return_value = None
        self.manager._process = mock_proc
        self.manager._spawned_by_app = True

        success, msg = self.manager.stop_service()
        self.assertTrue(success)
        self.assertIn("stopped", msg)
        mock_proc.terminate.assert_called_once()
        self.assertIsNone(self.manager._process)
        self.assertFalse(self.manager.spawned_by_app)
        mock_unload.assert_called_once()

    @patch.object(OllamaManager, "stop_service")
    def test_cleanup_on_exit(self, mock_stop):
        self.manager.cleanup_on_exit()
        mock_stop.assert_called_once()

    def test_singleton_get_ollama_manager(self):
        m1 = get_ollama_manager()
        m2 = get_ollama_manager()
        self.assertIs(m1, m2)


if __name__ == "__main__":
    unittest.main()
