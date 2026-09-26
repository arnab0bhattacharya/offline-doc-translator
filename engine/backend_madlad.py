"""
engine/backend_madlad.py
========================
High-fidelity Neural Machine Translation (NMT) backend powered by Google Research's MADLAD-400 3B.
Executes INT8 quantized inference via CTranslate2 (T5 architecture) and SentencePiece.

Architectural Guarantees:
- Strict Lazy Loading: Model weights and SentencePiece tables are never loaded until first user translation.
- Deterministic Unload: Memory is completely evicted from RAM/C++ runtime on unload() or process exit.
- Clean Text Pipeline: Operates natively on clean natural text without synthetic bracket fragmentation.
"""

import gc
import os
import time
from collections.abc import Callable
from typing import Any

from .backend_base import TranslationBackend
from .madlad_manager import check_madlad_installed, get_madlad_model_dir

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


class MADLADBackend(TranslationBackend):
    """
    Polymorphic translation backend for MADLAD-400 3B (INT8) via CTranslate2.
    """

    name = "madlad"

    def __init__(
        self,
        model_dir: str | None = None,
        compute_type: str = "int8",
        intra_threads: int | None = None,
    ):
        self.model_dir = model_dir or get_madlad_model_dir()
        self.compute_type = compute_type
        self.intra_threads = intra_threads or min(4, os.cpu_count() or 4)

        # STRICT LAZY LOADING: zero memory allocated at initialization
        self._translator: Any | None = None
        self._sp_processor: Any | None = None
        self._is_loaded = False

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
        """Returns True if the model weights are currently resident in RAM."""
        return self._is_loaded and self._translator is not None

    def _get_spiece_path(self) -> str:
        """Locates the SentencePiece model file in the model directory."""
        for fname in ("spiece.model", "sentencepiece.model", "sentencepiece.bpe.model"):
            fpath = os.path.join(self.model_dir, fname)
            if os.path.isfile(fpath):
                return fpath
        raise FileNotFoundError(f"SentencePiece model file missing in: {self.model_dir}")

    def _ensure_loaded(self) -> None:
        """Loads CTranslate2 translator and SentencePiece tokenizer into RAM on first use."""
        if self._is_loaded and self._translator is not None:
            return

        if not check_madlad_installed(self.model_dir):
            raise RuntimeError(
                f"MADLAD-400 3B model files not found or corrupted in: {self.model_dir}. "
                "Please download the model before translating."
            )

        import ctranslate2
        import sentencepiece as spm

        sp_path = self._get_spiece_path()
        sp = spm.SentencePieceProcessor()
        sp.load(sp_path)
        self._sp_processor = sp

        self._translator = ctranslate2.Translator(
            self.model_dir,
            device="cpu",
            compute_type=self.compute_type,
            intra_threads=self.intra_threads,
            inter_threads=1,
        )
        self._is_loaded = True

    def unload(self) -> None:
        """
        Evicts the CTranslate2 model and SentencePiece tokenizer from system RAM immediately.
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
