"""
engine/madlad_manager.py
========================
Model lifecycle and storage manager for Google Research's MADLAD-400 3B (INT8 / CTranslate2).
Handles model discovery, offline verification, chunked streaming download with progress,
and local directory imports for air-gapped systems.
"""

import hashlib
import os
import shutil
import ssl
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

APP_FOLDER_NAME = "OfflineDocumentTranslator"
MODELS_SUBDIR = "models"
MADLAD_3B_DIR_NAME = "madlad400-3b-ct2-int8"

# Pinned commit hash for reproducible and secure downloads
MADLAD_REVISION = "aa32bbdeba7880eff2096ec044cb155a340a9400"

# Files required for CTranslate2 + SentencePiece execution (minimum size floors)
REQUIRED_MODEL_FILES = {
    "config.json": 50,  # minimum 50 bytes (actual 224 bytes)
    "shared_vocabulary.json": 50 * 1024,  # minimum 50 KB (actual ~5.48 MB)
    "spiece.model": 1 * 1024 * 1024,  # minimum 1 MB (actual ~4.43 MB)
    "model.bin": 1500 * 1024 * 1024,  # minimum 1.5 GB (actual ~2.95 GB)
}

# Exact byte sizes for downloads and strict verification
EXACT_MODEL_FILES = {
    "config.json": 224,  # exact 224 bytes
    "shared_vocabulary.json": 5_477_099,  # exact ~5.48 MB
    "spiece.model": 4_427_844,  # exact ~4.43 MB
    "model.bin": 2_950_208_329,  # exact ~2.95 GB
}

MODEL_FILE_SHA256 = {
    "config.json": "90fb54962455a4e0a0bc7235c0f063d7e46d9c1a1ae003af8059809abd6aeece",
    "shared_vocabulary.json": "c327551ce3ca6efc7b437e11a267f79979893332dda8a1d146e2c950815193f8",
    "spiece.model": "ef11ac9a22c7503492f56d48dce53be20e339b63605983e9f27d2cd0e0f3922c",
    "model.bin": "77b9fd9ab97c1259d07089b5f854393dad81bc5fb5647d3f9a5d101c94f40daa",
}

# Approximate total bytes for progress calculation (~2.96 GB)
TOTAL_MODEL_BYTES_APPROX = 2_960_113_496

# Hugging Face CDN endpoints pinned to immutable commit revision (Nextcloud-AI/madlad400-3b-mt-ct2-int8)
HF_BASE_URL = f"https://huggingface.co/Nextcloud-AI/madlad400-3b-mt-ct2-int8/resolve/{MADLAD_REVISION}"
MADLAD_DOWNLOAD_MANIFEST = {
    "config.json": f"{HF_BASE_URL}/config.json",
    "shared_vocabulary.json": f"{HF_BASE_URL}/shared_vocabulary.json",
    "spiece.model": f"{HF_BASE_URL}/spiece.model",
    "model.bin": f"{HF_BASE_URL}/model.bin",
}


class _SystemSSLAdapter(HTTPAdapter):
    """
    HTTPAdapter that incorporates native OS (Windows / Linux / macOS) root CA certificates.
    This resolves SSLCertVerificationError on Windows systems without disabling verification.
    """

    def init_poolmanager(self, *args: Any, **kwargs: Any) -> None:
        ctx = create_urllib3_context()
        try:
            ctx.load_default_certs()
        except Exception:
            pass
        kwargs["ssl_context"] = ctx
        super().init_poolmanager(*args, **kwargs)


def create_secure_session() -> requests.Session:
    """Creates a requests session configured with system trust store certificates."""
    session = requests.Session()
    adapter = _SystemSSLAdapter()
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session



def get_models_root_dir() -> str:
    """
    Returns the absolute path to the application's models directory.
    Priority:
      1. MADLAD_MODELS_DIR or NLLB_MODELS_DIR environment variable
      2. %LOCALAPPDATA%/OfflineDocumentTranslator/models (Windows standard)
      3. ~/.offline-translator/models (POSIX / fallback)
    """
    env_dir = os.environ.get("MADLAD_MODELS_DIR") or os.environ.get("NLLB_MODELS_DIR")
    if env_dir:
        models_dir = os.path.abspath(env_dir)
    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        models_dir = os.path.join(os.environ["LOCALAPPDATA"], APP_FOLDER_NAME, MODELS_SUBDIR)
    else:
        user_home = os.path.expanduser("~")
        models_dir = os.path.join(user_home, ".offline-translator", MODELS_SUBDIR)

    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_madlad_model_dir(custom_path: str | None = None) -> str:
    """Returns the dedicated directory path for MADLAD-400 3B."""
    if custom_path:
        return os.path.abspath(custom_path)
    return os.path.join(get_models_root_dir(), MADLAD_3B_DIR_NAME)


def check_madlad_installed(model_dir: str | None = None) -> bool:
    """
    Verifies if MADLAD-400 3B model files are present and valid on disk.
    Checks existence and non-trivial file sizes for all required CTranslate2 files.
    Accepts either shared_vocabulary.json or shared_vocabulary.txt, and spiece.model or sentencepiece.model.
    """
    target_dir = model_dir or get_madlad_model_dir()
    if not os.path.isdir(target_dir):
        return False

    core_files = {
        "config.json": REQUIRED_MODEL_FILES.get("config.json", 50),
        "model.bin": REQUIRED_MODEL_FILES.get("model.bin", 1500 * 1024 * 1024),
    }
    for filename, min_bytes in core_files.items():
        file_path = os.path.join(target_dir, filename)
        if not os.path.isfile(file_path):
            return False
        try:
            if os.path.getsize(file_path) < min_bytes:
                return False
        except OSError:
            return False

    # Check for SentencePiece model (spiece.model or sentencepiece.model)
    sp_ok = False
    min_sp = REQUIRED_MODEL_FILES.get("spiece.model", REQUIRED_MODEL_FILES.get("sentencepiece.model", 1 * 1024 * 1024))
    for sp_name in ("spiece.model", "sentencepiece.model", "sentencepiece.bpe.model"):
        sp_path = os.path.join(target_dir, sp_name)
        if os.path.isfile(sp_path):
            try:
                if os.path.getsize(sp_path) >= min_sp:
                    sp_ok = True
                    break
            except OSError:
                pass
    if not sp_ok:
        return False

    # Check for vocabulary file (either .json or .txt)
    vocab_ok = False
    min_vocab = REQUIRED_MODEL_FILES.get(
        "shared_vocabulary.json", REQUIRED_MODEL_FILES.get("shared_vocabulary.txt", 50 * 1024)
    )
    for vocab_name in ("shared_vocabulary.json", "shared_vocabulary.txt"):
        vpath = os.path.join(target_dir, vocab_name)
        if os.path.isfile(vpath):
            try:
                if os.path.getsize(vpath) >= min_vocab:
                    vocab_ok = True
                    break
            except OSError:
                pass

    return vocab_ok


def get_madlad_model_info(model_dir: str | None = None) -> dict[str, Any]:
    """Returns detailed diagnostic info about local MADLAD model status."""
    target_dir = model_dir or get_madlad_model_dir()
    installed = check_madlad_installed(target_dir)
    total_bytes = 0
    missing = []

    core_files = {
        "config.json": REQUIRED_MODEL_FILES.get("config.json", 50),
        "model.bin": REQUIRED_MODEL_FILES.get("model.bin", 1500 * 1024 * 1024),
    }
    min_sp = REQUIRED_MODEL_FILES.get("spiece.model", REQUIRED_MODEL_FILES.get("sentencepiece.model", 1 * 1024 * 1024))
    min_vocab = REQUIRED_MODEL_FILES.get(
        "shared_vocabulary.json", REQUIRED_MODEL_FILES.get("shared_vocabulary.txt", 50 * 1024)
    )

    if os.path.isdir(target_dir):
        for filename, min_bytes in core_files.items():
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path):
                try:
                    size = os.path.getsize(file_path)
                    total_bytes += size
                    if size < min_bytes:
                        missing.append(f"{filename} (corrupted/truncated)")
                except OSError:
                    missing.append(f"{filename} (unreadable)")
            else:
                missing.append(filename)

        sp_found = False
        for sp_name in ("spiece.model", "sentencepiece.model", "sentencepiece.bpe.model"):
            sp_path = os.path.join(target_dir, sp_name)
            if os.path.isfile(sp_path):
                sp_found = True
                try:
                    size = os.path.getsize(sp_path)
                    total_bytes += size
                    if size < min_sp:
                        missing.append(f"{sp_name} (corrupted/truncated)")
                except OSError:
                    missing.append(f"{sp_name} (unreadable)")
                break
        if not sp_found:
            missing.append("spiece.model")

        vocab_found = False
        for vocab_name in ("shared_vocabulary.json", "shared_vocabulary.txt"):
            vpath = os.path.join(target_dir, vocab_name)
            if os.path.isfile(vpath):
                vocab_found = True
                try:
                    size = os.path.getsize(vpath)
                    total_bytes += size
                    if size < min_vocab:
                        missing.append(f"{vocab_name} (corrupted/truncated)")
                except OSError:
                    missing.append(f"{vocab_name} (unreadable)")
                break

        if not vocab_found:
            missing.append("shared_vocabulary.json")
    else:
        missing = [*core_files.keys(), "spiece.model", "shared_vocabulary.json"]

    return {
        "installed": installed,
        "path": target_dir,
        "size_mb": round(total_bytes / (1024 * 1024), 1),
        "missing_files": missing,
    }


def download_madlad_model(
    target_dir: str | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> tuple[bool, str]:
    """
    Downloads MADLAD-400 3B INT8 model files from Hugging Face with chunked streaming.
    Streams to .tmp files, verifies integrity (size and SHA256 checksum), and renames atomically upon completion.

    Args:
        target_dir: Destination folder. Defaults to get_madlad_model_dir().
        progress_cb: Callback receiving (percentage: float 0.0-100.0, status_str: str).
        cancel_event: Optional threading.Event to abort download.
        log_cb: Optional logging callback.

    Returns:
        tuple[bool, str]: (success, status_or_error_message)
    """
    dest_dir = target_dir or get_madlad_model_dir()
    os.makedirs(dest_dir, exist_ok=True)

    if check_madlad_installed(dest_dir):
        if log_cb:
            log_cb("[✓] MADLAD-400 3B model is already installed.")
        if progress_cb:
            progress_cb(100.0, "Model installed.")
        return True, "MADLAD-400 3B model is already installed."

    if log_cb:
        log_cb(f"[*] Starting MADLAD-400 3B download (~3.0 GB) to: {dest_dir}")

    accumulated_bytes = 0
    headers = {"User-Agent": "OfflineDocTranslator/1.0"}
    session = create_secure_session()

    try:
        for filename, url in MADLAD_DOWNLOAD_MANIFEST.items():
            final_path = os.path.join(dest_dir, filename)
            tmp_path = final_path + ".tmp"
            expected_size = REQUIRED_MODEL_FILES.get(filename, 1)
            expected_hash = MODEL_FILE_SHA256.get(filename)

            # Skip if already downloaded and valid
            if os.path.exists(final_path) and os.path.getsize(final_path) >= expected_size:
                accumulated_bytes += os.path.getsize(final_path)
                continue

            if log_cb:
                log_cb(f"[*] Downloading {filename} from Hugging Face...")

            try:
                response = session.get(url, headers=headers, stream=True, timeout=(10, 60))
                response.raise_for_status()

                file_bytes_done = 0
                hasher = hashlib.sha256()
                t_start = time.time()

                with open(tmp_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                        if cancel_event and cancel_event.is_set():
                            if log_cb:
                                log_cb("[!] Download cancelled by user.")
                            try:
                                f.close()
                                if os.path.exists(tmp_path):
                                    os.remove(tmp_path)
                            except OSError:
                                pass
                            return False, "Download cancelled by user."

                        if chunk:
                            f.write(chunk)
                            hasher.update(chunk)
                            file_bytes_done += len(chunk)
                            accumulated_bytes += len(chunk)

                            if progress_cb:
                                pct = min(99.0, (accumulated_bytes / TOTAL_MODEL_BYTES_APPROX) * 100)
                                elapsed = time.time() - t_start
                                speed_mb = (file_bytes_done / (1024 * 1024)) / max(0.1, elapsed)
                                mb_done = round(accumulated_bytes / (1024 * 1024), 1)
                                progress_cb(pct, f"Downloading {filename} ({mb_done} MB, {speed_mb:.1f} MB/s)...")

                actual_size = os.path.getsize(tmp_path)
                if actual_size < expected_size:
                    raise RuntimeError(
                        f"Downloaded file {filename} is too small ({actual_size} bytes, expected at least {expected_size} bytes)."
                    )

                if expected_hash:
                    actual_hash = hasher.hexdigest()
                    if actual_hash.lower() != expected_hash.lower():
                        raise RuntimeError(
                            f"Integrity check failed for {filename}: SHA256 mismatch (got {actual_hash[:8]}..., expected {expected_hash[:8]}...)."
                        )

                # Atomic move
                if os.path.exists(final_path):
                    os.remove(final_path)
                shutil.move(tmp_path, final_path)
                if log_cb:
                    log_cb(f"[✓] {filename} verified and saved.")

            except Exception:
                # Clean up incomplete temp file on any error
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass
                raise

        if check_madlad_installed(dest_dir):
            if log_cb:
                log_cb("[✓] MADLAD-400 3B model successfully installed and ready.")
            if progress_cb:
                progress_cb(100.0, "MADLAD-400 3B model installed successfully!")
            return True, "MADLAD-400 3B model installed successfully!"
        else:
            return False, "Model verification failed after download."

    except requests.exceptions.SSLError as ssl_err:
        err_msg = (
            f"SSL certificate verification failed while contacting Hugging Face: {ssl_err}\n"
            "This typically happens on networks with TLS inspection or corporate proxies. "
            "Ensure system root certificates are installed or set the SSL_CERT_FILE / REQUESTS_CA_BUNDLE environment variable."
        )
        if log_cb:
            log_cb(f"[!] {err_msg}")
        return False, err_msg
    except Exception as e:
        err_msg = str(e)
        if log_cb:
            log_cb(f"[!] Error downloading MADLAD model: {err_msg}")
        return False, f"Download failed: {err_msg}"
    finally:
        session.close()


def import_local_madlad_folder(
    source_folder: str, target_dir: str | None = None, log_cb: Callable[[str], None] | None = None
) -> tuple[bool, str]:
    """
    Imports model files from a local directory (for offline/air-gapped environments).
    Validates presence of required files, then copies them to application model directory.
    """
    src_abs = os.path.abspath(source_folder)
    if not os.path.isdir(src_abs):
        return False, f"Source folder not found: {src_abs}"

    if not check_madlad_installed(src_abs):
        info = get_madlad_model_info(src_abs)
        missing_str = ", ".join(info.get("missing_files", []))
        return False, f"Folder is missing required files or files are truncated: {missing_str}"

    dest_dir = target_dir or get_madlad_model_dir()
    os.makedirs(dest_dir, exist_ok=True)

    try:
        # Copy core files
        for filename in ("model.bin", "config.json"):
            src_file = os.path.join(src_abs, filename)
            dst_file = os.path.join(dest_dir, filename)
            if log_cb:
                log_cb(f"[*] Copying {filename}...")
            shutil.copy2(src_file, dst_file)

        # Copy SentencePiece model (whichever is present)
        for sp_name in ("spiece.model", "sentencepiece.model", "sentencepiece.bpe.model"):
            src_file = os.path.join(src_abs, sp_name)
            if os.path.isfile(src_file):
                dst_file = os.path.join(dest_dir, sp_name)
                if log_cb:
                    log_cb(f"[*] Copying {sp_name}...")
                shutil.copy2(src_file, dst_file)

        # Copy vocabulary (whichever is present)
        for vocab_name in ("shared_vocabulary.json", "shared_vocabulary.txt"):
            src_file = os.path.join(src_abs, vocab_name)
            if os.path.isfile(src_file):
                dst_file = os.path.join(dest_dir, vocab_name)
                if log_cb:
                    log_cb(f"[*] Copying {vocab_name}...")
                shutil.copy2(src_file, dst_file)

        if check_madlad_installed(dest_dir):
            if log_cb:
                log_cb(f"[✓] MADLAD model imported successfully to {dest_dir}")
            return True, f"Successfully imported model to {dest_dir}"
        return False, "Model verification failed after copying."
    except Exception as e:
        return False, f"Failed to copy model files: {e}"
