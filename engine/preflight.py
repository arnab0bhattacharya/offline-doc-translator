"""
engine/preflight.py
===================
System and environment diagnostics before running translation.
Detects Ollama connectivity, model availability, RAM, and disk constraints.
"""

import os
import shutil
import psutil
import requests
from typing import List, Tuple, Optional
from .errors import ErrorCode, TranslatorError


def check_ollama_status(ollama_url: str = "http://localhost:11434") -> bool:
    """Fast health check to determine if the local Ollama server is responsive."""
    try:
        res = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
        return res.status_code == 200
    except Exception:
        return False


def list_installed_models(ollama_url: str = "http://localhost:11434") -> List[str]:
    """Retrieves all locally pulled models from Ollama."""
    try:
        res = requests.get(f"{ollama_url.rstrip('/')}/api/tags", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return [m.get("name", "") for m in data.get("models", []) if "name" in m]
    except Exception:
        pass
    return []


def check_model_installed(model_name: str, ollama_url: str = "http://localhost:11434") -> bool:
    """Verifies whether the requested model is pulled and available in Ollama."""
    models = list_installed_models(ollama_url)
    if not models:
        return False
    # Check exact match or base name match (e.g., 'gemma4:e2b-it-qat' or 'gemma4:e2b-it-qat:latest')
    for m in models:
        if m == model_name or m.startswith(f"{model_name}:") or model_name.startswith(f"{m}:"):
            return True
    return False


def check_ram(min_free_mb: int = 150) -> Tuple[bool, float]:
    """Checks if available system RAM clears the minimum safe operating floor."""
    try:
        available_mb = psutil.virtual_memory().available / (1024 * 1024)
        return (available_mb >= min_free_mb, available_mb)
    except Exception:
        return (True, 9999.0)


def check_disk_space(path: str = ".", min_free_mb: int = 500) -> Tuple[bool, float]:
    """Checks if free disk space on the given drive/path clears the minimum threshold."""
    try:
        target_dir = os.path.abspath(path)
        if not os.path.exists(target_dir):
            target_dir = os.path.dirname(target_dir) or "."
        total, used, free = shutil.disk_usage(target_dir)
        free_mb = free / (1024 * 1024)
        return (free_mb >= min_free_mb, free_mb)
    except Exception:
        return (True, 9999.0)


def check_nmt_ready(direction: str) -> bool:
    """Checks that the Argos runtime and requested local language package are usable."""
    try:
        from .backend_nmt import NMTBackend
        return NMTBackend().is_ready(direction)
    except Exception:
        return False


def run_nmt_preflight(direction: str) -> None:
    """Raises a user-facing error when strict Fast NMT mode cannot run locally."""
    if not check_nmt_ready(direction):
        raise TranslatorError(
            ErrorCode.E08,
            detail=f"No usable local Argos package was found for {direction}."
        )




def run_preflight(
    model_name: str,
    input_path: Optional[str] = None,
    output_path: Optional[str] = None,
    ollama_url: str = "http://localhost:11434",
    min_free_ram_mb: int = 150,
    min_free_disk_mb: int = 500,
    check_model: bool = True,
    require_ollama: bool = True,
) -> None:
    """
    Executes complete preflight validation.
    Raises TranslatorError immediately upon finding a fatal environment issue.
    """
    # 1. Validate local paths first so file errors are not masked by backend status.
    if input_path is not None:
        if not os.path.exists(input_path):
            raise TranslatorError(ErrorCode.E04, detail=f"Input file not found at: {input_path}")
        valid_exts = {".pptx", ".xlsx", ".docx", ".pdf"}
        ext = os.path.splitext(input_path)[1].lower()
        if ext not in valid_exts:
            raise TranslatorError(
                ErrorCode.E04,
                detail=f"Unsupported format '{ext}'. Supported: {', '.join(sorted(valid_exts))}"
            )

    if output_path is not None and os.path.exists(output_path):
        try:
            with open(output_path, "a+b"):
                pass
        except PermissionError as pe:
            raise TranslatorError(
                ErrorCode.E05,
                detail=f"Output file '{output_path}' is locked by another program.",
                original_exc=pe
            )

    # 2. Check RAM (E03)
    has_ram, ram_mb = check_ram(min_free_ram_mb)
    if not has_ram:
        raise TranslatorError(
            ErrorCode.E03,
            detail=f"Only {ram_mb:.0f} MB available (minimum safety floor is {min_free_ram_mb} MB)."
        )

    # 3. Check Disk Space (E07)
    target_check_dir = os.path.dirname(output_path) if output_path else "."
    has_disk, disk_mb = check_disk_space(target_check_dir, min_free_disk_mb)
    if not has_disk:
        raise TranslatorError(
            ErrorCode.E07,
            detail=f"Only {disk_mb:.0f} MB free disk space (minimum required is {min_free_disk_mb} MB)."
        )

    # 4. Check Ollama server availability (E01)
    if require_ollama and not check_ollama_status(ollama_url):
        raise TranslatorError(ErrorCode.E01, detail=f"Could not connect to Ollama at {ollama_url}")

    # 5. Check model existence (E02)
    if require_ollama and check_model and not check_model_installed(model_name, ollama_url):
        available = list_installed_models(ollama_url)
        available_str = ", ".join(available) if available else "None found"
        raise TranslatorError(
            ErrorCode.E02,
            detail=f"Model '{model_name}' is not in local Ollama inventory. Available models: [{available_str}]"
        )
