"""
tests/test_nllb_backend.py
==========================
Unit tests for NLLB-200 1.3B backend, lazy loading, memory unload eviction,
manager diagnostics, and preflight validation.
"""

import os
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.backend_nllb import NLLB_FLORES_MAP, NLLBBackend
from engine.core import TranslationEngine, TranslationMode
from engine.errors import ErrorCode, TranslatorError
from engine.nllb_manager import (
    REQUIRED_MODEL_FILES,
    check_nllb_installed,
    download_nllb_model,
    get_nllb_model_info,
    import_local_model_folder,
)
from engine.preflight import check_nllb_ready, run_nllb_preflight


class TestNLLBManager(unittest.TestCase):
    """Test model file detection, directory info, and local folder importing."""

    def test_check_nllb_installed_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(check_nllb_installed(td))

    def test_check_nllb_installed_valid_files(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            self.assertTrue(check_nllb_installed(td))

    def test_check_nllb_installed_with_json_vocab(self):
        """Verifies that shared_vocabulary.json is also accepted as a valid vocabulary format."""
        with tempfile.TemporaryDirectory() as td:
            for fname in ("config.json", "sentencepiece.bpe.model", "model.bin"):
                min_bytes = REQUIRED_MODEL_FILES[fname]
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            # Write shared_vocabulary.json instead of shared_vocabulary.txt
            vpath = os.path.join(td, "shared_vocabulary.json")
            with open(vpath, "wb") as f:
                f.seek(60 * 1024)
                f.write(b"0")
            self.assertTrue(check_nllb_installed(td))

    def test_check_nllb_installed_truncated_file(self):
        with tempfile.TemporaryDirectory() as td:
            for fname in REQUIRED_MODEL_FILES:
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.write(b"tiny")  # Below minimum byte thresholds
            self.assertFalse(check_nllb_installed(td))

    def test_get_nllb_model_info(self):
        with tempfile.TemporaryDirectory() as td:
            info = get_nllb_model_info(td)
            self.assertFalse(info["installed"])
            self.assertGreater(len(info["missing_files"]), 0)

    def test_import_local_model_folder(self):
        with tempfile.TemporaryDirectory() as src_dir, tempfile.TemporaryDirectory() as dst_dir:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")

            ok, msg = import_local_model_folder(src_dir, target_dir=dst_dir)
            self.assertTrue(ok)
            self.assertTrue(check_nllb_installed(dst_dir))

    def test_download_nllb_model_already_installed(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            ok, msg = download_nllb_model(target_dir=td)
            self.assertTrue(ok)
            self.assertIn("already installed", msg)

    @patch("engine.nllb_manager.requests.get")
    def test_download_nllb_model_stream_success(self, mock_get):
        with tempfile.TemporaryDirectory() as td:

            def fake_get(url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                if "model.bin" in url:
                    chunk = b"x" * (1024 * 1024)
                    mock_resp.iter_content.return_value = [chunk] * 501
                elif "shared_vocabulary" in url:
                    chunk = b"v" * (60 * 1024)
                    mock_resp.iter_content.return_value = [chunk]
                elif "sentencepiece" in url:
                    chunk = b"s" * (1024 * 1024)
                    mock_resp.iter_content.return_value = [chunk] * 2
                else:  # config.json (min 50 bytes)
                    mock_resp.iter_content.return_value = [
                        b'{"format_version": 1, "model_type": "nllb-200-distilled-1.3B"}'
                    ]
                return mock_resp

            mock_get.side_effect = fake_get
            progress_calls = []

            def prog_cb(pct, status):
                progress_calls.append((pct, status))

            ok, msg = download_nllb_model(target_dir=td, progress_cb=prog_cb)
            self.assertTrue(ok)
            self.assertIn("installed successfully", msg)
            self.assertTrue(check_nllb_installed(td))
            self.assertGreater(len(progress_calls), 0)

    @patch("engine.nllb_manager.requests.get")
    def test_download_nllb_model_cancelled(self, mock_get):
        with tempfile.TemporaryDirectory() as td:
            cancel_evt = threading.Event()
            cancel_evt.set()

            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.iter_content.return_value = [b"chunk1"]
            mock_get.return_value = mock_resp

            ok, msg = download_nllb_model(target_dir=td, cancel_event=cancel_evt)
            self.assertFalse(ok)
            self.assertIn("cancelled", msg)

    @patch("engine.nllb_manager.requests.get")
    def test_download_nllb_model_http_error(self, mock_get):
        with tempfile.TemporaryDirectory() as td:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("404 Client Error: Not Found")
            mock_get.return_value = mock_resp

            ok, msg = download_nllb_model(target_dir=td)
            self.assertFalse(ok)
            self.assertIn("404 Client Error", msg)


class TestNLLBBackendLazyLoading(unittest.TestCase):
    """Test strict lazy loading and deterministic memory eviction."""

    def test_zero_memory_on_initialization(self):
        backend = NLLBBackend(model_dir="/dummy/path")
        self.assertIsNone(backend._translator)
        self.assertIsNone(backend._sp_processor)
        self.assertFalse(backend.is_model_loaded())

    def test_unload_evicts_memory(self):
        backend = NLLBBackend(model_dir="/dummy/path")
        # Simulate loaded state
        backend._translator = MagicMock()
        backend._sp_processor = MagicMock()
        backend._is_loaded = True
        self.assertTrue(backend.is_model_loaded())

        # Unload
        backend.unload()
        self.assertIsNone(backend._translator)
        self.assertIsNone(backend._sp_processor)
        self.assertFalse(backend.is_model_loaded())

    def test_flores_mapping_coverage(self):
        self.assertIn("ja", NLLB_FLORES_MAP)
        self.assertIn("en", NLLB_FLORES_MAP)
        self.assertEqual(NLLB_FLORES_MAP["ja"], "jpn_Jpan")
        self.assertEqual(NLLB_FLORES_MAP["en"], "eng_Latn")


class TestNLLBTranslationExecution(unittest.TestCase):
    """Test mocked CTranslate2 and SentencePiece translation pipeline."""

    def test_translate_batch_mocked(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")

            backend = NLLBBackend(model_dir=td)

            # Mock SentencePiece
            mock_sp = MagicMock()
            mock_sp.encode_as_pieces.return_value = ["_Hello", "_world"]
            mock_sp.decode_pieces.return_value = "こんにちは世界"

            # Mock CTranslate2
            mock_hyp = MagicMock()
            mock_hyp.hypotheses = [["jpn_Jpan", "_こんにちは", "_世界"]]
            mock_translator = MagicMock()
            mock_translator.translate_batch.return_value = [mock_hyp]

            backend._sp_processor = mock_sp
            backend._translator = mock_translator
            backend._is_loaded = True

            res = backend.translate_single("Hello world", direction="en2ja")
            self.assertEqual(res, "こんにちは世界")

            # Verify input structure passed to translate_batch
            mock_translator.translate_batch.assert_called_once()
            call_args, call_kwargs = mock_translator.translate_batch.call_args
            tokenized_batch = call_args[0]
            # Should have src_lang prefix and </s> suffix
            self.assertEqual(tokenized_batch[0][0], "eng_Latn")
            self.assertEqual(tokenized_batch[0][-1], "</s>")
            # Should specify target prefix
            self.assertEqual(call_kwargs["target_prefix"], [["jpn_Jpan"]])


class TestNLLBPreflightAndEngineIntegration(unittest.TestCase):
    """Test preflight validation and TranslationEngine integration with QUALITY_NMT."""

    def test_preflight_fails_when_uninstalled(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("engine.backend_nllb.get_nllb_model_dir", return_value=td):
                self.assertFalse(check_nllb_ready("ja2en"))
                with self.assertRaises(TranslatorError) as ctx:
                    run_nllb_preflight("ja2en")
                self.assertEqual(ctx.exception.code, ErrorCode.E08)

    def test_engine_routes_quality_nmt_to_nllb(self):
        engine = TranslationEngine(mode=TranslationMode.QUALITY_NMT)
        backend = engine.get_backend()
        self.assertIsInstance(backend, NLLBBackend)

    def test_engine_unload_backends(self):
        engine = TranslationEngine(mode=TranslationMode.QUALITY_NMT)
        # Mock nllb backend unload
        engine.nllb_backend._translator = MagicMock()
        engine.nllb_backend._is_loaded = True
        self.assertTrue(engine.nllb_backend.is_model_loaded())

        engine.unload_backends()
        self.assertFalse(engine.nllb_backend.is_model_loaded())


if __name__ == "__main__":
    unittest.main()
