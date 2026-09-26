"""
engine/nllb_manager.py
======================
Model lifecycle and storage manager for NLLB-200 1.3B (INT8 / CTranslate2).
Handles model discovery, offline verification, chunked streaming download with progress,
and local directory imports for air-gapped systems.
"""

import os
import shutil
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

import requests

APP_FOLDER_NAME = "OfflineDocumentTranslator"
MODELS_SUBDIR = "models"
NLLB_1_3B_DIR_NAME = "nllb-200-distilled-1.3B-ct2-int8"

# Files required for CTranslate2 + SentencePiece execution
REQUIRED_MODEL_FILES = {
    "model.bin": 500 * 1024 * 1024,  # minimum 500 MB (actual ~1.4 GB)
    "shared_vocabulary.json": 50 * 1024,  # minimum 50 KB (actual ~2.5 MB)
    "sentencepiece.bpe.model": 1 * 1024 * 1024,  # minimum 1 MB (actual ~4.9 MB)
}

# Hugging Face CDN endpoints (JustFrederik/nllb-200-distilled-1.3B-ct2-int8)
HF_BASE_URL = "https://huggingface.co/JustFrederik/nllb-200-distilled-1.3B-ct2-int8/resolve/main"
NLLB_DOWNLOAD_MANIFEST = {
    "sentencepiece.bpe.model": f"{HF_BASE_URL}/sentencepiece.bpe.model",
    "shared_vocabulary.json": f"{HF_BASE_URL}/shared_vocabulary.json",
    "model.bin": f"{HF_BASE_URL}/model.bin",
}


def get_models_root_dir() -> str:
    """
    Returns the absolute path to the application's models directory.
    Priority:
      1. NLLB_MODELS_DIR environment variable (for custom installations/tests)
      2. %LOCALAPPDATA%/OfflineDocumentTranslator/models (Windows standard)
      3. ~/.offline-translator/models (POSIX / fallback)
    """
    env_dir = os.environ.get("NLLB_MODELS_DIR")
    if env_dir:
        models_dir = os.path.abspath(env_dir)
    elif sys.platform == "win32" and os.environ.get("LOCALAPPDATA"):
        models_dir = os.path.join(os.environ["LOCALAPPDATA"], APP_FOLDER_NAME, MODELS_SUBDIR)
    else:
        user_home = os.path.expanduser("~")
        models_dir = os.path.join(user_home, ".offline-translator", MODELS_SUBDIR)

    os.makedirs(models_dir, exist_ok=True)
    return models_dir


def get_nllb_model_dir(custom_path: str | None = None) -> str:
    """Returns the dedicated directory path for NLLB-200 1.3B."""
    if custom_path:
        return os.path.abspath(custom_path)
    return os.path.join(get_models_root_dir(), NLLB_1_3B_DIR_NAME)


def check_nllb_installed(model_dir: str | None = None) -> bool:
    """
    Verifies if NLLB-200 1.3B model files are present and valid on disk.
    Checks existence and non-trivial file sizes for all required CTranslate2 files.
    """
    target_dir = model_dir or get_nllb_model_dir()
    if not os.path.isdir(target_dir):
        return False

    for filename, min_bytes in REQUIRED_MODEL_FILES.items():
        file_path = os.path.join(target_dir, filename)
        if not os.path.isfile(file_path):
            return False
        try:
            if os.path.getsize(file_path) < min_bytes:
                return False
        except OSError:
            return False

    return True


def get_nllb_model_info(model_dir: str | None = None) -> dict[str, Any]:
    """Returns detailed diagnostic info about local NLLB model status."""
    target_dir = model_dir or get_nllb_model_dir()
    installed = check_nllb_installed(target_dir)
    total_bytes = 0
    missing = []

    if os.path.isdir(target_dir):
        for filename, min_bytes in REQUIRED_MODEL_FILES.items():
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
    else:
        missing = list(REQUIRED_MODEL_FILES.keys())

    return {
        "installed": installed,
        "path": target_dir,
        "size_mb": round(total_bytes / (1024 * 1024), 1),
        "missing_files": missing,
    }


def download_nllb_model(
    target_dir: str | None = None,
    progress_cb: Callable[[float, str], None] | None = None,
    cancel_event: threading.Event | None = None,
    log_cb: Callable[[str], None] | None = None,
) -> bool:
    """
    Downloads NLLB-200 1.3B INT8 model files from Hugging Face with chunked streaming.
    Streams to .tmp files and renames atomically upon completion.

    Args:
        target_dir: Destination folder. Defaults to get_nllb_model_dir().
        progress_cb: Callback receiving (percentage: float 0.0-100.0, status_str: str).
        cancel_event: Optional threading.Event to abort download.
        log_cb: Optional logging callback.
    """
    dest_dir = target_dir or get_nllb_model_dir()
    os.makedirs(dest_dir, exist_ok=True)

    if check_nllb_installed(dest_dir):
        if log_cb:
            log_cb("[✓] NLLB-200 1.3B model is already installed.")
        if progress_cb:
            progress_cb(100.0, "Model installed.")
        return True

    if log_cb:
        log_cb(f"[*] Starting NLLB-200 1.3B download (~1.4 GB) to: {dest_dir}")

    accumulated_bytes = 0

    try:
        for filename, url in NLLB_DOWNLOAD_MANIFEST.items():
            final_path = os.path.join(dest_dir, filename)
            tmp_path = final_path + ".tmp"

            # Skip if already downloaded and valid
            if os.path.exists(final_path) and os.path.getsize(final_path) >= REQUIRED_MODEL_FILES[filename]:
                accumulated_bytes += os.path.getsize(final_path)
                continue

            if log_cb:
                log_cb(f"[*] Downloading {filename} from Hugging Face...")

            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            file_bytes_done = 0
            t_start = time.time()

            with open(tmp_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):  # 1 MB chunks
                    if cancel_event and cancel_event.is_set():
                        if log_cb:
                            log_cb("[!] Download cancelled by user.")
                        try:
                            f.close()
                            os.remove(tmp_path)
                        except OSError:
                            pass
                        return False

                    if chunk:
                        f.write(chunk)
                        file_bytes_done += len(chunk)
                        accumulated_bytes += len(chunk)

                        if progress_cb:
                            pct = min(99.0, (accumulated_bytes / (1.45 * 1024 * 1024 * 1024)) * 100)
                            elapsed = time.time() - t_start
                            speed_mb = (file_bytes_done / (1024 * 1024)) / max(0.1, elapsed)
                            mb_done = round(accumulated_bytes / (1024 * 1024), 1)
                            progress_cb(pct, f"Downloading {filename} ({mb_done} MB, {speed_mb:.1f} MB/s)...")

            # Validate size floor before committing
            if os.path.getsize(tmp_path) < REQUIRED_MODEL_FILES[filename]:
                raise RuntimeError(f"Downloaded file {filename} is too small ({os.path.getsize(tmp_path)} bytes).")

            # Atomic move
            if os.path.exists(final_path):
                os.remove(final_path)
            shutil.move(tmp_path, final_path)
            if log_cb:
                log_cb(f"[✓] {filename} verified and saved.")

        if check_nllb_installed(dest_dir):
            if log_cb:
                log_cb("[✓] NLLB-200 1.3B model successfully installed and ready.")
            if progress_cb:
                progress_cb(100.0, "NLLB 1.3B model installed successfully!")
            return True
        else:
            raise RuntimeError("Model verification failed after download.")

    except Exception as e:
        if log_cb:
            log_cb(f"[!] Error downloading NLLB model: {e}")
        return False


def import_local_model_folder(
    source_folder: str, target_dir: str | None = None, log_cb: Callable[[str], None] | None = None
) -> tuple[bool, str]:
    """
    Imports model files from a local directory (for offline/air-gapped environments).
    Validates presence of required files, then copies them to application model directory.
    """
    src_abs = os.path.abspath(source_folder)
    if not os.path.isdir(src_abs):
        return False, f"Source folder not found: {src_abs}"

    missing = []
    for filename, min_bytes in REQUIRED_MODEL_FILES.items():
        src_file = os.path.join(src_abs, filename)
        if not os.path.isfile(src_file) or os.path.getsize(src_file) < min_bytes:
            missing.append(filename)

    if missing:
        return False, f"Folder is missing required files or files are truncated: {', '.join(missing)}"

    dest_dir = target_dir or get_nllb_model_dir()
    os.makedirs(dest_dir, exist_ok=True)

    try:
        for filename in REQUIRED_MODEL_FILES:
            src_file = os.path.join(src_abs, filename)
            dst_file = os.path.join(dest_dir, filename)
            if log_cb:
                log_cb(f"[*] Copying {filename}...")
            shutil.copy2(src_file, dst_file)

        if check_nllb_installed(dest_dir):
            if log_cb:
                log_cb(f"[✓] NLLB model imported successfully to {dest_dir}")
            return True, f"Successfully imported model to {dest_dir}"
        return False, "Model verification failed after copying."
    except Exception as e:
        return False, f"Failed to copy model files: {e}"
