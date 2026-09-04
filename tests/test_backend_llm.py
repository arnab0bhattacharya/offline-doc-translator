"""
tests/test_backend_llm.py
=========================
Unit tests for the LLM backend.
Note: translate_single requires a live Ollama server, so we only test
instantiation and configuration here.
"""

import unittest
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backend_llm import LLMBackend


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
        backend = LLMBackend(
            model_name="test_model",
            ollama_url="http://192.168.1.10:11434",
            context_window=4096
        )
        self.assertEqual(backend.ollama_url, "http://192.168.1.10:11434")
        self.assertEqual(backend.context_window, 4096)


if __name__ == "__main__":
    unittest.main()
