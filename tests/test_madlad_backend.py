"""
tests/test_madlad_backend.py
============================
Unit tests for MADLAD-400 3B backend, lazy loading, memory unload eviction,
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

from engine.backend_madlad import MADLAD_LANG_MAP, MADLADBackend
from engine.core import TranslationEngine, TranslationMode
from engine.errors import ErrorCode, TranslatorError
from engine.madlad_manager import (
    REQUIRED_MODEL_FILES,
    check_madlad_installed,
    download_madlad_model,
    get_madlad_model_info,
    import_local_madlad_folder,
)
from engine.preflight import check_madlad_ready, run_madlad_preflight


class TestMADLADManager(unittest.TestCase):
    """Test model file detection, directory info, and local folder importing."""

    def test_check_madlad_installed_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(check_madlad_installed(td))

    def test_check_madlad_installed_valid_files(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_with_txt_vocab(self):
        """Verifies that shared_vocabulary.txt is also accepted as a valid vocabulary format."""
        with tempfile.TemporaryDirectory() as td:
            for fname in ("config.json", "spiece.model", "model.bin"):
                min_bytes = REQUIRED_MODEL_FILES[fname]
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            # Write shared_vocabulary.txt instead of shared_vocabulary.json
            vpath = os.path.join(td, "shared_vocabulary.txt")
            with open(vpath, "wb") as f:
                f.seek(60 * 1024)
                f.write(b"0")
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_with_alternate_sp_name(self):
        """Verifies sentencepiece.model is accepted if spiece.model is not found."""
        with tempfile.TemporaryDirectory() as td:
            for fname in ("config.json", "shared_vocabulary.json", "model.bin"):
                min_bytes = REQUIRED_MODEL_FILES[fname]
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            # Write sentencepiece.model instead of spiece.model
            sp_path = os.path.join(td, "sentencepiece.model")
            with open(sp_path, "wb") as f:
                f.seek(1 * 1024 * 1024 + 10)
                f.write(b"0")
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_truncated_file(self):
        with tempfile.TemporaryDirectory() as td:
            for fname in REQUIRED_MODEL_FILES:
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.write(b"tiny")  # Below minimum byte thresholds
            self.assertFalse(check_madlad_installed(td))

    def test_get_madlad_model_info(self):
        with tempfile.TemporaryDirectory() as td:
            info = get_madlad_model_info(td)
            self.assertFalse(info["installed"])
            self.assertGreater(len(info["missing_files"]), 0)

    def test_import_local_madlad_folder(self):
        test_required = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dst_dir,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required),
        ):
            for fname, min_bytes in test_required.items():
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")

            ok, msg = import_local_madlad_folder(src_dir, target_dir=dst_dir)
            self.assertTrue(ok)
            self.assertTrue(check_madlad_installed(dst_dir))

    def test_download_madlad_model_already_installed(self):
        test_required = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required),
        ):
            for fname, min_bytes in test_required.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")
            ok, msg = download_madlad_model(target_dir=td)
            self.assertTrue(ok)
            self.assertIn("already installed", msg)

    @patch("engine.madlad_manager.create_secure_session")
    def test_download_madlad_model_stream_success(self, mock_create_session):
        test_required = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", {}, clear=True),
        ):

            def fake_get(url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                if "model.bin" in url:
                    mock_resp.iter_content.return_value = [b"m" * 200]
                elif "shared_vocabulary" in url:
                    mock_resp.iter_content.return_value = [b"v" * 200]
                elif "spiece" in url:
                    mock_resp.iter_content.return_value = [b"s" * 200]
                else:  # config.json (min 50 bytes)
                    mock_resp.iter_content.return_value = [b'{"format_version": 1, "model_type": "madlad400-3b"}']
                return mock_resp

            mock_session.get.side_effect = fake_get
            progress_calls = []

            def prog_cb(pct, status):
                progress_calls.append((pct, status))

            ok, msg = download_madlad_model(target_dir=td, progress_cb=prog_cb)
            self.assertTrue(ok)
            self.assertIn("installed successfully", msg)
            self.assertTrue(check_madlad_installed(td))
            self.assertGreater(len(progress_calls), 0)

    @patch("engine.madlad_manager.create_secure_session")
    def test_download_madlad_model_cancelled(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        with tempfile.TemporaryDirectory() as td:
            cancel_evt = threading.Event()
            cancel_evt.set()

            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.iter_content.return_value = [b"chunk1"]
            mock_session.get.return_value = mock_resp

            ok, msg = download_madlad_model(target_dir=td, cancel_event=cancel_evt)
            self.assertFalse(ok)
            self.assertIn("cancelled", msg)

    @patch("engine.madlad_manager.create_secure_session")
    def test_download_madlad_model_http_error(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        with tempfile.TemporaryDirectory() as td:
            mock_resp = MagicMock()
            mock_resp.raise_for_status.side_effect = Exception("404 Client Error: Not Found")
            mock_session.get.return_value = mock_resp

            ok, msg = download_madlad_model(target_dir=td)
            self.assertFalse(ok)
            self.assertIn("404 Client Error", msg)

    @patch("engine.madlad_manager.create_secure_session")
    def test_download_madlad_model_sha256_mismatch(self, mock_create_session):
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session
        test_required = {"config.json": 10}
        test_hashes = {"config.json": "0000000000000000000000000000000000000000000000000000000000000000"}

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", test_hashes),
        ):
            mock_resp = MagicMock()
            mock_resp.raise_for_status.return_value = None
            mock_resp.iter_content.return_value = [b"test data with different hash"]
            mock_session.get.return_value = mock_resp

            ok, msg = download_madlad_model(target_dir=td)
            self.assertFalse(ok)
            self.assertIn("SHA256 mismatch", msg)
            # Ensure .tmp file was deleted
            tmp_file = os.path.join(td, "config.json.tmp")
            self.assertFalse(os.path.exists(tmp_file))


class TestMADLADBackendLazyLoading(unittest.TestCase):
    """Test strict lazy loading and deterministic memory eviction."""

    def test_zero_memory_on_initialization(self):
        backend = MADLADBackend(model_dir="/dummy/path")
        self.assertIsNone(backend._translator)
        self.assertIsNone(backend._sp_processor)
        self.assertFalse(backend.is_model_loaded())

    def test_unload_evicts_memory(self):
        backend = MADLADBackend(model_dir="/dummy/path")
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

    def test_madlad_mapping_coverage(self):
        self.assertIn("ja", MADLAD_LANG_MAP)
        self.assertIn("en", MADLAD_LANG_MAP)
        self.assertEqual(MADLAD_LANG_MAP["ja"], "ja")
        self.assertEqual(MADLAD_LANG_MAP["en"], "en")


class TestMADLADTranslationExecution(unittest.TestCase):
    """Test mocked CTranslate2 and SentencePiece translation pipeline."""

    def test_translate_batch_mocked(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, min_bytes in REQUIRED_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.seek(min_bytes + 10)
                    f.write(b"0")

            backend = MADLADBackend(model_dir=td)

            # Mock SentencePiece
            mock_sp = MagicMock()
            mock_sp.encode_as_pieces.return_value = ["_Hello", "_world"]
            mock_sp.decode_pieces.return_value = "こんにちは世界"

            # Mock CTranslate2
            mock_hyp = MagicMock()
            mock_hyp.hypotheses = [["_こんにちは", "_世界"]]
            mock_translator = MagicMock()
            mock_translator.translate_batch.return_value = [mock_hyp]

            backend._sp_processor = mock_sp
            backend._translator = mock_translator
            backend._is_loaded = True

            res = backend.translate_single("Hello world", direction="en2ja")
            self.assertEqual(res, "こんにちは世界")

            # Verify anti-repetition guards were supplied
            mock_translator.translate_batch.assert_called_once()
            call_args, call_kwargs = mock_translator.translate_batch.call_args
            self.assertEqual(call_kwargs.get("repetition_penalty"), 1.2)
            self.assertEqual(call_kwargs.get("no_repeat_ngram_size"), 3)

    def test_translate_single_multiline_decomposition(self):
        with tempfile.TemporaryDirectory() as td:
            backend = MADLADBackend(model_dir=td)

            # Mock SentencePiece
            mock_sp = MagicMock()
            mock_sp.encode_as_pieces.side_effect = lambda s: [s]
            mock_sp.decode_pieces.side_effect = lambda pieces: f"Trans_{pieces[0]}"

            # Mock CTranslate2
            def fake_translate_batch(batch, **kwargs):
                results = []
                for item in batch:
                    hyp = MagicMock()
                    content = item[0].replace("<2ja> ", "")
                    hyp.hypotheses = [[f"JP_{content}"]]
                    results.append(hyp)
                return results

            mock_translator = MagicMock()
            mock_translator.translate_batch.side_effect = fake_translate_batch

            backend._sp_processor = mock_sp
            backend._translator = mock_translator
            backend._is_loaded = True

            multiline_input = "Line 1\n\nLine 2\nLine 3"
            res = backend.translate_single(multiline_input, direction="en2ja")

            expected = "Trans_JP_Line 1\n\nTrans_JP_Line 2\nTrans_JP_Line 3"
            self.assertEqual(res, expected)
            # Only 3 non-empty lines were sent to translate_batch
            self.assertEqual(len(mock_translator.translate_batch.call_args[0][0]), 3)


class TestMADLADPreflightAndEngineIntegration(unittest.TestCase):
    """Test preflight validation and TranslationEngine integration with MADLAD_3B."""

    def test_preflight_fails_when_uninstalled(self):
        with tempfile.TemporaryDirectory() as td:
            with patch("engine.backend_madlad.get_madlad_model_dir", return_value=td):
                self.assertFalse(check_madlad_ready("ja2en"))
                with self.assertRaises(TranslatorError) as ctx:
                    run_madlad_preflight("ja2en")
                self.assertEqual(ctx.exception.code, ErrorCode.E08)

    def test_engine_routes_madlad_3b_backend(self):
        engine = TranslationEngine(mode=TranslationMode.MADLAD_3B)
        backend = engine.get_backend()
        self.assertIsInstance(backend, MADLADBackend)

    def test_engine_unload_backends(self):
        engine = TranslationEngine(mode=TranslationMode.MADLAD_3B)
        # Mock madlad backend unload
        engine.madlad_backend._translator = MagicMock()
        engine.madlad_backend._is_loaded = True
        self.assertTrue(engine.madlad_backend.is_model_loaded())

        engine.unload_backends()
        self.assertFalse(engine.madlad_backend.is_model_loaded())


class TestDynamicHeadroomAndMutualExclusivity(unittest.TestCase):
    """Test dynamic available memory headroom thread allocation and mutual engine eviction."""

    def test_dynamic_headroom_threads(self):
        from engine.system_specs import calculate_optimal_cpu_threads

        # Tight memory headroom (<= 2.2 GB available): must restrict to 2 threads
        self.assertEqual(
            calculate_optimal_cpu_threads(total_ram_gb=8.0, avail_ram_gb=1.2, logical_threads=12),
            2,
        )
        self.assertEqual(
            calculate_optimal_cpu_threads(total_ram_gb=16.0, avail_ram_gb=2.0, logical_threads=12),
            2,
        )

        # Moderate memory headroom (2.2 - 3.5 GB): cap at 3 threads
        self.assertEqual(
            calculate_optimal_cpu_threads(total_ram_gb=8.0, avail_ram_gb=3.0, logical_threads=12),
            3,
        )

        # Plentiful memory headroom (>= 4.5 GB) on 8 GB machine: cap at 4 threads
        self.assertEqual(
            calculate_optimal_cpu_threads(total_ram_gb=8.0, avail_ram_gb=5.0, logical_threads=12),
            4,
        )

        # High-RAM machine with plentiful headroom: scales up to tier limit
        self.assertEqual(
            calculate_optimal_cpu_threads(total_ram_gb=16.0, avail_ram_gb=10.0, logical_threads=12),
            6,
        )

    @patch("engine.ollama_manager.OllamaManager.unload_all_models")
    def test_ensure_backend_exclusive_mt_evicts_ollama(self, mock_unload_ollama):
        engine = TranslationEngine(mode=TranslationMode.MACHINE_TRANSLATION)
        engine.ensure_backend_exclusive(TranslationMode.MACHINE_TRANSLATION)
        mock_unload_ollama.assert_called_once()

    def test_ensure_backend_exclusive_ai_evicts_madlad(self):
        engine = TranslationEngine(mode=TranslationMode.AI_TRANSLATION)
        # Simulate MADLAD resident in memory
        engine.madlad_backend._translator = MagicMock()
        engine.madlad_backend._is_loaded = True
        self.assertTrue(engine.madlad_backend.is_model_loaded())

        # Calling ensure_backend_exclusive for AI Translation must evict MADLAD
        engine.ensure_backend_exclusive(TranslationMode.AI_TRANSLATION)
        self.assertFalse(engine.madlad_backend.is_model_loaded())


if __name__ == "__main__":
    unittest.main()
