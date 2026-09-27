"""
engine/backend_madlad.py
========================
High-fidelity Machine Translation (MT) backend powered by Google Research's MADLAD-400 3B (Apache 2.0).
Executes INT8 quantized inference via CTranslate2 (T5 architecture) and SentencePiece.

Architectural Guarantees:
- Model Pinning: Every device runs identical model weights and quantization for enterprise-wide consistency.
- Adaptive Runtime: Auto-detects NVIDIA CUDA GPUs with sufficient VRAM, gracefully falling back to CPU.
- Strict Lazy Loading: Model weights and SentencePiece tables are never loaded until first user translation.
- Deterministic Unload: Memory is completely evicted from RAM/VRAM on unload() or process exit.
- Clean Text Pipeline: Operates natively on clean natural text without synthetic bracket fragmentation.
"""

import gc
import os
import time
from collections.abc import Callable
from typing import Any

from .backend_base import TranslationBackend
from .madlad_manager import check_madlad_installed, get_madlad_model_dir
from .system_specs import calculate_optimal_cpu_threads

# Supported language codes for MADLAD-400
MADLAD_LANG_MAP = {
    "ja": "ja",
    "en": "en",
    "es": "es",
    "de": "de",
    "fr": "fr",
    "zh": "zh",
    "ko": "ko",
    "it": "it",
    "pt": "pt",
    "ru": "ru",
}


def detect_hardware() -> tuple[str, str, str]:
    """
    Detects system compute hardware and determines optimal CTranslate2 runtime settings.
    Returns: (device, compute_type, description)
      - device: "cuda" or "cpu"
      - compute_type: "int8_float16" for CUDA, "int8" for CPU
      - description: human-readable hardware descriptor
    """
    try:
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            free_vram_mb = 0
            gpu_name = "NVIDIA GPU"
            try:
                import torch

                if torch.cuda.is_available():
                    gpu_name = torch.cuda.get_device_name(0)
                    total_vram = torch.cuda.get_device_properties(0).total_memory
                    free_vram_mb = int(total_vram / (1024 * 1024))
            except Exception:
                pass

            if free_vram_mb >= 3500:
                return "cuda", "int8_float16", f"{gpu_name} ({free_vram_mb} MB VRAM) — CUDA Accelerated"
            elif free_vram_mb == 0 and ctranslate2.get_cuda_device_count() > 0:
                return "cuda", "int8_float16", "CUDA Hardware Acceleration"
    except Exception:
        pass

    # Dynamic adaptive CPU allocation based on host system RAM and core count
    cpu_cores = os.cpu_count() or 4
    threads = calculate_optimal_cpu_threads(logical_threads=cpu_cores)
    return "cpu", "int8", f"CPU Execution (Allocated {threads} of {cpu_cores} threads to preserve RAM stability)"


class MADLADBackend(TranslationBackend):
    """
    Polymorphic translation backend for Google MADLAD-400 3B (INT8) via CTranslate2.
    """

    name = "madlad"

    def __init__(
        self,
        model_dir: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
        intra_threads: int | None = None,
    ):
        self.model_dir = model_dir or get_madlad_model_dir()
        self._user_device = device
        self._user_compute_type = compute_type
        self._user_intra_threads = intra_threads
        self.intra_threads = intra_threads or calculate_optimal_cpu_threads()

        # Hardware runtime selection
        self._active_device = "cpu"
        self._active_compute_type = "int8"
        self._hardware_desc = "CPU"
        self._refresh_hardware_config()

        # STRICT LAZY LOADING: zero memory allocated at initialization
        self._translator: Any | None = None
        self._sp_processor: Any | None = None
        self._is_loaded = False

    def _refresh_hardware_config(self) -> None:
        """Determines active device and compute type based on user override or auto-detection."""
        if self._user_device:
            self._active_device = self._user_device
            self._active_compute_type = self._user_compute_type or (
                "int8_float16" if self._user_device == "cuda" else "int8"
            )
            self._hardware_desc = f"{self._active_device.upper()} (Manual Override)"
        else:
            dev, comp, desc = detect_hardware()
            self._active_device = dev
            self._active_compute_type = comp
            self._hardware_desc = desc

    @property
    def active_device(self) -> str:
        """Returns the active execution device ('cpu' or 'cuda')."""
        return self._active_device

    @property
    def active_device_description(self) -> str:
        """Returns a human-readable description of the compute backend."""
        return self._hardware_desc

    def is_available(self) -> bool:
        """Returns True if CTranslate2, SentencePiece, and model weights are present."""
        try:
            import ctranslate2  # noqa: F401
            import sentencepiece  # noqa: F401

            return check_madlad_installed(self.model_dir)
        except ImportError:
            return False

    def is_ready(self, direction: str) -> bool:
        """Returns True if the backend is installed and direction language codes are supported."""
        if not self.is_available():
            return False
        parts = direction.split("2")
        if len(parts) != 2:
            return False
        src_code, tgt_code = parts
        return src_code in MADLAD_LANG_MAP and tgt_code in MADLAD_LANG_MAP

    def is_model_loaded(self) -> bool:
        """Returns True if the model weights are currently resident in RAM/VRAM."""
        return self._is_loaded and self._translator is not None

    def _get_spiece_path(self) -> str:
        """Locates the SentencePiece model file in the model directory."""
        for fname in ("spiece.model", "sentencepiece.model", "sentencepiece.bpe.model"):
            fpath = os.path.join(self.model_dir, fname)
            if os.path.isfile(fpath):
                return fpath
        raise FileNotFoundError(f"SentencePiece model file missing in: {self.model_dir}")

    def _ensure_loaded(self) -> None:
        """Loads CTranslate2 translator and SentencePiece tokenizer into memory on first use."""
        if self._is_loaded and self._translator is not None:
            return

        if not check_madlad_installed(self.model_dir):
            raise RuntimeError(
                f"MADLAD-400 3B model files not found or corrupted in: {self.model_dir}. "
                "Please download the model before translating."
            )

        # Enforce mutual exclusivity: evict any resident Ollama models from RAM
        try:
            from engine.ollama_manager import get_ollama_manager

            get_ollama_manager().unload_all_models()
        except Exception:
            pass

        import ctranslate2
        import sentencepiece as spm

        sp_path = self._get_spiece_path()
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)
        self._sp_processor = sp

        self._refresh_hardware_config()
        device = self._active_device
        compute_type = self._active_compute_type

        # Re-sample real-time available memory headroom right before allocating CTranslate2 buffers
        if self._user_intra_threads is None and device == "cpu":
            self.intra_threads = calculate_optimal_cpu_threads()

        try:
            self._translator = ctranslate2.Translator(
                self.model_dir,
                device=device,
                compute_type=compute_type,
                intra_threads=self.intra_threads if device == "cpu" else 1,
                inter_threads=1,
            )
        except Exception as e:
            # If CUDA initialization failed, gracefully fall back to CPU execution
            if device == "cuda":
                device = "cpu"
                compute_type = "int8"
                self._active_device = "cpu"
                self._active_compute_type = "int8"
                self._hardware_desc = f"CPU Execution ({self.intra_threads} threads) [CUDA fallback: {e}]"
                self._translator = ctranslate2.Translator(
                    self.model_dir,
                    device=device,
                    compute_type=compute_type,
                    intra_threads=self.intra_threads,
                    inter_threads=1,
                )
            else:
                raise

        self._is_loaded = True

    def unload(self) -> None:
        """
        Evicts the CTranslate2 model and SentencePiece tokenizer from system memory immediately.
        Calls garbage collection to reclaim memory.
        """
        if self._translator is not None:
            del self._translator
            self._translator = None

        if self._sp_processor is not None:
            del self._sp_processor
            self._sp_processor = None

        self._is_loaded = False
        gc.collect()

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: dict[str, str] | None = None,
        context: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> tuple[str | None, float]:
        """
        Unified TranslationBackend protocol method.
        Translates a single text unit using MADLAD 3B and returns (result, elapsed).
        """
        if not text or not text.strip():
            return text, 0.0

        t0 = time.time()
        try:
            result = self.translate_single(text, direction=direction, log_cb=log_cb)
            elapsed = time.time() - t0
            return result, elapsed
        except Exception as e:
            if "mkl_malloc" in str(e).lower() or isinstance(e, MemoryError):
                raise
            if log_cb:
                log_cb(f"[!] MADLAD translation error: {e}")
            elapsed = time.time() - t0
            return None, elapsed

    def translate_single(
        self,
        text: str,
        direction: str,
        log_cb: Callable[[str], None] | None = None,
    ) -> str:
        """Translates a single clean text string."""
        results = self.translate_batch([text], direction=direction, log_cb=log_cb)
        return results[0] if results else text

    def translate_batch(
        self,
        texts: list[str],
        direction: str,
        log_cb: Callable[[str], None] | None = None,
    ) -> list[str]:
        """
        Translates a batch of texts using CTranslate2 vectorization.
        Prepares token inputs formatted as: <2{tgt_lang}> {text} </s>
        """
        if not texts:
            return []

        parts = direction.split("2")
        if len(parts) != 2:
            raise ValueError(f"Invalid direction format '{direction}'. Expected 'src2tgt'.")
        _src_code, tgt_code = parts

        if tgt_code not in MADLAD_LANG_MAP:
            raise ValueError(f"Unsupported target language for MADLAD: {tgt_code}")

        self._ensure_loaded()
        sp = self._sp_processor
        translator = self._translator

        tgt_tag = f"<2{tgt_code}>"
        tokenized_batch = []
        for t in texts:
            # MADLAD-400 expects prefix: <2{tgt_lang}> {text}
            prompt_str = f"{tgt_tag} {t.strip()}"
            pieces = sp.encode_as_pieces(prompt_str)
            pieces.append("</s>")
            tokenized_batch.append(pieces)

        step_results = translator.translate_batch(
            tokenized_batch,
            batch_type="examples",
            max_batch_size=len(texts),
            beam_size=4,
        )

        translated_texts = []
        for res in step_results:
            hyp = res.hypotheses[0]
            # Strip eos/bos if present
            cleaned_pieces = [p for p in hyp if p not in ("<s>", "</s>")]
            decoded = sp.decode_pieces(cleaned_pieces)
            translated_texts.append(decoded.strip())

        return translated_texts
