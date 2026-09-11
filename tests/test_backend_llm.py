import os
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backend_llm import LLMBackend
from engine.core import mask_numbers, unmask_numbers


class TestLLMBackend(unittest.TestCase):
    def setUp(self):
        self.backend = LLMBackend(model_name="mock_model")

    def test_backend_initialization(self):
        """Verify LLMBackend initializes with correct attributes."""
        self.assertEqual(self.backend.model_name, "mock_model")
        self.assertEqual(self.backend.ollama_url, "http://localhost:11434")
        self.assertEqual(self.backend.context_window, 2048)

    def test_custom_url_and_context(self):
        """Verify custom ollama_url and context_window are stored."""
        backend = LLMBackend(model_name="test_model", ollama_url="http://192.168.1.10:11434", context_window=4096)
        self.assertEqual(backend.ollama_url, "http://192.168.1.10:11434")
        self.assertEqual(backend.context_window, 4096)

    @patch("requests.post")
    def test_translate_single_success_first_attempt(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"response": "これは[[N0]]のテストです。"}
        mock_post.return_value = mock_resp

        masked = "This is a test of [[N0]]."
        num_map = {"[[N0]]": "42"}

        res, elapsed = self.backend.translate_single(masked, num_map, "en2ja")
        self.assertEqual(res, "これは[[N0]]のテストです。")
        self.assertEqual(mock_post.call_count, 1)

    @patch("requests.post")
    def test_translate_single_isomorphic_retry_on_missing_placeholder(self, mock_post):
        # Attempt 1 drops placeholder, Attempt 2 includes it
        resp_bad = MagicMock(status_code=200)
        resp_bad.json.return_value = {"response": "これはテストです。"}  # missing [[N0]]

        resp_good = MagicMock(status_code=200)
        resp_good.json.return_value = {"response": "これは[[N0]]のテストです。"}

        mock_post.side_effect = [resp_bad, resp_good]

        masked = "This is a test of [[N0]]."
        num_map = {"[[N0]]": "42"}

        res, elapsed = self.backend.translate_single(masked, num_map, "en2ja")
        self.assertEqual(res, "これは[[N0]]のテストです。")
        self.assertEqual(mock_post.call_count, 2)

    @patch("requests.post")
    def test_translate_single_failure_both_attempts(self, mock_post):
        resp_bad = MagicMock(status_code=200)
        resp_bad.json.return_value = {"response": "Invalid output"}

        mock_post.side_effect = [resp_bad, resp_bad]

        masked = "This is a test of [[N0]]."
        num_map = {"[[N0]]": "42"}

        res, elapsed = self.backend.translate_single(masked, num_map, "en2ja")
        self.assertIsNone(res)
        self.assertEqual(mock_post.call_count, 2)

    @patch("engine.backend_llm.LLMBackend.translate_single")
    def test_quick_translate_pipeline_integration(self, mock_translate):
        mock_translate.return_value = ("これは[[N0]]のテストです。", 0.42)

        text = "This is a test of 99."
        masked, num_map = mask_numbers(text)
        res, elapsed = self.backend.translate_single(masked, num_map, "en2ja")
        final = unmask_numbers(res, num_map)

        self.assertEqual(final, "これは99のテストです。")


if __name__ == "__main__":
    unittest.main()
