"""
tests/test_madlad_backend.py
============================
Unit tests for MADLAD-400 3B backend, lazy loading, memory unload eviction,
manager diagnostics, and preflight validation.
"""

import os
import ssl
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
    EXACT_MODEL_FILES,
    REQUIRED_MODEL_FILES,
    _SystemSSLAdapter,
    check_madlad_installed,
    download_madlad_model,
    get_madlad_model_info,
    import_local_madlad_folder,
    verify_madlad_integrity,
)
from engine.preflight import check_madlad_ready, run_madlad_preflight


class TestMADLADManager(unittest.TestCase):
    """Test model file detection, directory info, and local folder importing."""

    def test_check_madlad_installed_empty_dir(self):
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(check_madlad_installed(td))

    def test_check_madlad_installed_valid_files(self):
        with tempfile.TemporaryDirectory() as td:
            for fname, exact_bytes in EXACT_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.truncate(exact_bytes)
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_with_txt_vocab(self):
        """Verifies that shared_vocabulary.txt is also accepted as a valid vocabulary format."""
        with tempfile.TemporaryDirectory() as td:
            for fname in ("config.json", "spiece.model", "model.bin"):
                exact_bytes = EXACT_MODEL_FILES[fname]
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.truncate(exact_bytes)
            # Write shared_vocabulary.txt instead of shared_vocabulary.json
            vpath = os.path.join(td, "shared_vocabulary.txt")
            with open(vpath, "wb") as f:
                f.truncate(60 * 1024)
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_with_alternate_sp_name(self):
        """Verifies sentencepiece.model is accepted if spiece.model is not found."""
        with tempfile.TemporaryDirectory() as td:
            for fname in ("config.json", "shared_vocabulary.json", "model.bin"):
                exact_bytes = EXACT_MODEL_FILES[fname]
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.truncate(exact_bytes)
            # Write sentencepiece.model instead of spiece.model
            sp_path = os.path.join(td, "sentencepiece.model")
            with open(sp_path, "wb") as f:
                f.truncate(EXACT_MODEL_FILES["spiece.model"])
            self.assertTrue(check_madlad_installed(td))

    def test_check_madlad_installed_size_mismatch(self):
        """Verifies that a file exceeding minimum floor but not matching exact size is rejected by default."""
        with tempfile.TemporaryDirectory() as td:
            for fname, exact_bytes in EXACT_MODEL_FILES.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.truncate(exact_bytes)
            # Corrupt model.bin to 2.95 GB + 10 bytes (different from EXACT_MODEL_FILES)
            bin_path = os.path.join(td, "model.bin")
            with open(bin_path, "wb") as f:
                f.truncate(2_950_000_010)
            self.assertFalse(check_madlad_installed(td))
            # But passes if exact_sizes=False
            self.assertTrue(check_madlad_installed(td, exact_sizes=False))

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
            self.assertEqual(info["status"], "not_installed")
            self.assertFalse(info["exact_size_ok"])
            self.assertGreater(len(info["missing_files"]), 0)

            # Write valid exact files
            for fname, exact_bytes in EXACT_MODEL_FILES.items():
                with open(os.path.join(td, fname), "wb") as f:
                    f.truncate(exact_bytes)
            info_ready = get_madlad_model_info(td)
            self.assertTrue(info_ready["installed"])
            self.assertEqual(info_ready["status"], "ready")
            self.assertTrue(info_ready["exact_size_ok"])
            self.assertFalse(info_ready["cryptographically_verified"])

    def test_import_local_madlad_folder(self):
        test_exact = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dst_dir,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", {}, clear=True),
        ):
            for fname, exact_size in test_exact.items():
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(b"0" * exact_size)

            ok, msg = import_local_madlad_folder(src_dir, target_dir=dst_dir)
            self.assertTrue(ok)
            self.assertTrue(check_madlad_installed(dst_dir))

    def test_import_local_madlad_folder_corrupted_rejected(self):
        """Verifies that local folder import rejects source directory with corrupted/mismatched files."""
        test_exact = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        with (
            tempfile.TemporaryDirectory() as src_dir,
            tempfile.TemporaryDirectory() as dst_dir,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", {}, clear=True),
        ):
            for fname, exact_size in test_exact.items():
                fpath = os.path.join(src_dir, fname)
                with open(fpath, "wb") as f:
                    f.write(b"0" * exact_size)
            # Corrupt model.bin in source folder
            with open(os.path.join(src_dir, "model.bin"), "wb") as f:
                f.write(b"corrupt")

            ok, msg = import_local_madlad_folder(src_dir, target_dir=dst_dir)
            self.assertFalse(ok)
            self.assertIn("failed integrity verification", msg)
            self.assertFalse(os.path.exists(os.path.join(dst_dir, "model.bin")))

    def test_download_madlad_model_already_installed(self):
        test_required = {
            "config.json": 50,
            "shared_vocabulary.json": 100,
            "spiece.model": 100,
            "model.bin": 100,
        }
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_required, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", {}, clear=True),
        ):
            for fname, min_bytes in test_required.items():
                fpath = os.path.join(td, fname)
                with open(fpath, "wb") as f:
                    f.write(b"0" * min_bytes)
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
        cfg_content = b'{"format_version": 1, "model_type": "madlad400-3b"}'
        test_exact = {
            "config.json": len(cfg_content),
            "shared_vocabulary.json": 200,
            "spiece.model": 200,
            "model.bin": 200,
        }
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
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
                else:  # config.json
                    mock_resp.iter_content.return_value = [cfg_content]
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
        test_required = {
            "config.json": 10,
            "shared_vocabulary.json": 10,
            "spiece.model": 10,
            "model.bin": 10,
        }
        test_hashes = {"config.json": "0000000000000000000000000000000000000000000000000000000000000000"}

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_required, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", {}, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", test_hashes, clear=True),
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

    @patch("engine.madlad_manager.create_secure_session")
    def test_download_madlad_model_existing_truncated_file_redownloaded(self, mock_create_session):
        """Verifies that an existing truncated file on disk is deleted and re-downloaded rather than skipped."""
        mock_session = MagicMock()
        mock_create_session.return_value = mock_session

        test_exact = {
            "config.json": 20,
            "shared_vocabulary.json": 20,
            "spiece.model": 20,
            "model.bin": 200,
        }

        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", {}, clear=True),
        ):
            # Create a truncated model.bin with only 50 bytes (less than 200 expected)
            bad_model = os.path.join(td, "model.bin")
            with open(bad_model, "wb") as f:
                f.write(b"t" * 50)

            def fake_get(url, **kwargs):
                mock_resp = MagicMock()
                mock_resp.raise_for_status.return_value = None
                if "model.bin" in url:
                    mock_resp.iter_content.return_value = [b"m" * 200]
                else:
                    mock_resp.iter_content.return_value = [b"x" * 20]
                return mock_resp

            mock_session.get.side_effect = fake_get

            ok, msg = download_madlad_model(target_dir=td)
            self.assertTrue(ok)
            self.assertEqual(os.path.getsize(bad_model), 200)

    def test_verify_madlad_integrity_valid_and_invalid(self):
        """Verifies cryptographic validation helper across intact and corrupt states."""
        with tempfile.TemporaryDirectory() as td:
            # Empty dir fails
            ok, errors = verify_madlad_integrity(td)
            self.assertFalse(ok)
            self.assertGreater(len(errors), 0)

            # Create dummy files matching EXACT_MODEL_FILES
            test_exact = {
                "config.json": 10,
                "shared_vocabulary.json": 10,
                "spiece.model": 10,
                "model.bin": 20,
            }
            test_hash = {
                "config.json": "03ac674216f3e15c761ee1a5e255f067953623c8b388b4459e13f978d7c846f4",  # sha256 of b'1234567890'
                "model.bin": "0000000000000000000000000000000000000000000000000000000000000000",
            }
            with (
                patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_exact, clear=True),
                patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
                patch.dict("engine.madlad_manager.MODEL_FILE_SHA256", test_hash, clear=True),
            ):
                with open(os.path.join(td, "config.json"), "wb") as f:
                    f.write(b"1234567890")
                with open(os.path.join(td, "shared_vocabulary.json"), "wb") as f:
                    f.write(b"v" * 10)
                with open(os.path.join(td, "spiece.model"), "wb") as f:
                    f.write(b"s" * 10)
                with open(os.path.join(td, "model.bin"), "wb") as f:
                    f.write(b"a" * 20)

                ok, errors = verify_madlad_integrity(td)
                self.assertFalse(ok)  # model.bin hash fails
                self.assertTrue(any("model.bin" in e and "checksum mismatch" in e for e in errors))

    def test_system_ssl_adapter_proxy_manager(self):
        """Verifies that _SystemSSLAdapter correctly attaches system SSL context to both pool and proxy managers."""
        adapter = _SystemSSLAdapter()
        adapter.init_poolmanager(connections=2, maxsize=2)
        ctx = adapter.poolmanager.connection_pool_kw.get("ssl_context")
        self.assertIsNotNone(ctx)

        proxy_mgr = adapter.proxy_manager_for("http://proxy.internal:8080")
        self.assertIsNotNone(proxy_mgr.connection_pool_kw.get("ssl_context"))

    def test_system_ssl_adapter_loads_default_certs(self):
        """Verifies that _get_ssl_context loads OS certificates without disabling TLS verification."""
        adapter = _SystemSSLAdapter()
        ctx = adapter._get_ssl_context()
        self.assertIsNotNone(ctx)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        if sys.platform == "win32":
            # On Windows, verify that CryptoAPI root certs were loaded into OpenSSL context
            ca_certs = ctx.get_ca_certs()
            self.assertGreater(len(ca_certs), 0)

    def test_create_secure_session_mounts_adapter(self):
        """Verifies create_secure_session mounts _SystemSSLAdapter on both http and https."""
        from engine.madlad_manager import create_secure_session

        session = create_secure_session()
        self.assertIsInstance(session.adapters.get("https://"), _SystemSSLAdapter)
        self.assertIsInstance(session.adapters.get("http://"), _SystemSSLAdapter)
        session.close()

    def test_backend_readiness_enforces_exact_sizes(self):
        """Verifies backend is_available and _ensure_loaded enforce exact model file sizes."""
        test_exact = {
            "config.json": 10,
            "shared_vocabulary.json": 10,
            "spiece.model": 10,
            "model.bin": 20,
        }
        with (
            tempfile.TemporaryDirectory() as td,
            patch.dict("engine.madlad_manager.REQUIRED_MODEL_FILES", test_exact, clear=True),
            patch.dict("engine.madlad_manager.EXACT_MODEL_FILES", test_exact, clear=True),
        ):
            for fname, exact_size in test_exact.items():
                with open(os.path.join(td, fname), "wb") as f:
                    f.write(b"0" * exact_size)

            backend = MADLADBackend(model_dir=td)
            self.assertTrue(backend.is_available())

            # Corrupt model.bin size
            with open(os.path.join(td, "model.bin"), "wb") as f:
                f.write(b"0" * (test_exact["model.bin"] + 5))

            self.assertFalse(backend.is_available())
            with self.assertRaises(RuntimeError):
                backend._ensure_loaded()


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
