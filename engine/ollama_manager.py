"""
engine/ollama_manager.py
========================
Lifecycle and memory management for the local Ollama background service.
Provides headless process launching, loaded model discovery (/api/ps),
instant VRAM eviction (keep_alive: 0), and clean shutdown cleanup.
"""

import os
import shutil
import subprocess
import sys
import threading
import time

import requests

from .preflight import check_ollama_status


def find_ollama_binary() -> str | None:
    """
    Locates the Ollama executable on the system.
    Searches system PATH first, then well-known installation directories on Windows.
    """
    # 1. Search PATH
    path_bin = shutil.which("ollama")
    if path_bin:
        return os.path.abspath(path_bin)

    # 2. Search Windows standard install locations
    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        program_files = os.environ.get("PROGRAMFILES", "")
        program_files_x86 = os.environ.get("PROGRAMFILES(X86)", "")

        candidates = [
            os.path.join(local_app_data, "Programs", "Ollama", "ollama.exe"),
            os.path.join(local_app_data, "Programs", "Ollama", "ollama app.exe"),
            os.path.join(program_files, "Ollama", "ollama.exe"),
            os.path.join(program_files_x86, "Ollama", "ollama.exe"),
        ]

        for cand in candidates:
            if cand and os.path.isfile(cand):
                return os.path.abspath(cand)

    return None


class OllamaManager:
    """
    Manages Ollama background service lifecycle and model memory eviction.
    """

    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url.rstrip("/")
        self._process: subprocess.Popen | None = None
        self._spawned_by_app: bool = False
        self._lock = threading.Lock()

    @property
    def spawned_by_app(self) -> bool:
        """Returns True if the current background process was spawned by this manager."""
        return self._spawned_by_app

    def is_running(self) -> bool:
        """Checks if Ollama service is responsive to HTTP requests."""
        return check_ollama_status(self.ollama_url)

    def start_service(self, timeout: float = 15.0) -> tuple[bool, str]:
        """
        Launches Ollama service in the background without opening a terminal window.
        Polls until the service is responsive or timeout is reached.
        Thread-safe against concurrent invocations.
        """
        with self._lock:
            if self.is_running():
                return True, "Ollama is already running."

            if self._process is not None and self._process.poll() is None:
                # Already spawned and initializing in another thread
                pass
            else:
                binary = find_ollama_binary()
                if not binary:
                    return False, "Ollama executable not found. Please install Ollama from https://ollama.com."

                creationflags = 0
                if sys.platform == "win32":
                    creationflags = subprocess.CREATE_NO_WINDOW

                try:
                    self._process = subprocess.Popen(
                        [binary, "serve"],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                        creationflags=creationflags,
                    )
                    self._spawned_by_app = True
                except Exception as e:
                    self._process = None
                    self._spawned_by_app = False
                    return False, f"Failed to spawn Ollama process: {e}"

            # Wait for service to become responsive
            deadline = time.time() + timeout
            while time.time() < deadline:
                if self.is_running():
                    return True, "Ollama started successfully in background."
                if self._process is not None and self._process.poll() is not None:
                    rc = self._process.returncode
                    self._process = None
                    self._spawned_by_app = False
                    return False, f"Ollama process exited immediately with code {rc}."
                time.sleep(0.5)

            return False, f"Ollama failed to respond within {int(timeout)} seconds."

    def get_loaded_models(self) -> list[str]:
        """
        Queries Ollama's /api/ps endpoint to discover models currently resident in VRAM/RAM.
        Returns a list of model names.
        """
        try:
            res = requests.get(f"{self.ollama_url}/api/ps", timeout=3.0)
            if res.status_code == 200:
                data = res.json()
                models = []
                for m in data.get("models", []):
                    name = m.get("name") or m.get("model")
                    if name:
                        models.append(name)
                return models
        except Exception:
            pass
        return []

    def unload_model(self, model_name: str) -> bool:
        """
        Evicts a specific model from GPU VRAM and system RAM immediately
        by sending keep_alive: 0 to Ollama's /api/generate (or /api/chat fallback).
        """
        payload = {"model": model_name, "keep_alive": 0}
        # 1. Try /api/generate
        try:
            res = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=5.0)
            if res.status_code == 200:
                return True
        except Exception:
            pass

        # 2. Fallback to /api/chat
        try:
            res = requests.post(f"{self.ollama_url}/api/chat", json=payload, timeout=5.0)
            if res.status_code == 200:
                return True
        except Exception:
            pass

        return False

    def unload_all_models(self) -> list[str]:
        """
        Discovers all models currently held in memory and unloads each one.
        Returns list of model names that were unloaded.
        """
        loaded = self.get_loaded_models()
        return [model for model in loaded if self.unload_model(model)]

    def stop_service(self) -> tuple[bool, str]:
        """
        Evicts models from memory and, if the Ollama service was spawned by our application,
        terminates the background process.
        """
        with self._lock:
            # Always unload models first
            self.unload_all_models()

            if self._spawned_by_app and self._process is not None:
                if self._process.poll() is None:
                    try:
                        self._process.terminate()
                        try:
                            self._process.wait(timeout=2.0)
                        except subprocess.TimeoutExpired:
                            self._process.kill()
                            self._process.wait(timeout=1.0)
                    except Exception as e:
                        return False, f"Failed to terminate Ollama process: {e}"
                    finally:
                        self._process = None
                        self._spawned_by_app = False
                else:
                    self._process = None
                    self._spawned_by_app = False
                return True, "Ollama background service stopped."

            return False, "Ollama was not launched by this application; model memory evicted."

    def cleanup_on_exit(self) -> None:
        """
        Safety cleanup hook called during application shutdown.
        Evicts loaded models to release VRAM/RAM, and terminates the Ollama
        process if it was started by this app.
        Guaranteed non-crashing and non-blocking with tight timeouts.
        """
        try:
            self.stop_service()
        except Exception:
            pass


_manager: OllamaManager | None = None


def get_ollama_manager() -> OllamaManager:
    """Returns the process-wide singleton instance of OllamaManager."""
    global _manager
    if _manager is None:
        _manager = OllamaManager()
    return _manager
