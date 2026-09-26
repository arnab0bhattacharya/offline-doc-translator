"""
engine/backend_nllb.py
======================
High-fidelity Neural Machine Translation (NMT) backend powered by Meta's NLLB-200 3.3B.
Executes INT8 quantized inference via CTranslate2 and SentencePiece with zero external server dependencies.

Architectural Guarantees:
- Strict Lazy Loading: Model weights and SentencePiece tables are never loaded until first user translation.
- Deterministic Unload: Memory is completely evicted from RAM/C++ runtime on unload() or process exit.
- Clean Text Pipeline: Operates natively on clean natural text; no synthetic bracket fragmentation.
"""

import gc
import os
import time
from collections.abc import Callable
from typing import Any

from .backend_base import TranslationBackend
from .nllb_manager import check_nllb_installed, get_nllb_model_dir

# FLORES-200 BCP-47 language tag mappings
NLLB_FLORES_MAP = {
    "ja": "jpn_Jpan",
    "en": "eng_Latn",
    "es": "spa_Latn",
    "de": "deu_Latn",
    "fr": "fra_Latn",
    "zh": "zho_Hans",
    "ko": "kor_Hang",
    "it": "ita_Latn",
    "pt": "por_Latn",
    "ru": "rus_Cyrl",
}


class NLLBBackend(TranslationBackend):
    """
    Polymorphic translation backend for NLLB-200 3.3B (INT8) via CTranslate2.
    """

    name = "nllb"

    def __init__(
        self,
        model_dir: str | None = None,
        compute_type: str = "int8",
        intra_threads: int | None = None,
    ):
        self.model_dir = model_dir or get_nllb_model_dir()
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

            return check_nllb_installed(self.model_dir)
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
        return src_code in NLLB_FLORES_MAP and tgt_code in NLLB_FLORES_MAP

    def is_model_loaded(self) -> bool:
        """Returns True if the model weights are currently resident in RAM."""
        return self._is_loaded and self._translator is not None

    def _ensure_loaded(self) -> None:
        """Loads CTranslate2 translator and SentencePiece tokenizer into RAM on first use."""
        if self._is_loaded and self._translator is not None:
            return

        if not check_nllb_installed(self.model_dir):
            raise RuntimeError(
                f"NLLB-200 3.3B model files not found or corrupted in: {self.model_dir}. "
                "Please download the model before translating."
            )

        import ctranslate2
        import sentencepiece as spm

        sp_path = os.path.join(self.model_dir, "sentencepiece.bpe.model")
        if not os.path.isfile(sp_path):
            raise FileNotFoundError(f"SentencePiece model file missing at: {sp_path}")

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
        Translates a single text unit using NLLB 1.3B and returns (result, elapsed).
        """
        t0 = time.time()
        try:
            res = self.translate_single(text, direction)
            elapsed = time.time() - t0
            return res, elapsed
        except Exception as e:
            if "mkl_malloc" in str(e).lower() or isinstance(e, MemoryError):
                raise
            if log_cb:
                log_cb(f"  [-] NLLB backend error: {e}")
            return None, time.time() - t0

    def translate_single(self, text: str, direction: str) -> str:
        """Translates a single string using NLLB-200 1.3B."""
        batch_res = self.translate_batch([text], direction)
        return batch_res[0] if batch_res else ""

    def translate_batch(self, texts: list[str], direction: str) -> list[str]:
        """
        Translates a batch of clean text strings using CTranslate2 beam search.
        Automatically formats input with FLORES-200 prefix and target language prefix.
        """
        if not texts:
            return []

        parts = direction.split("2")
        if len(parts) != 2:
            raise ValueError(f"Invalid direction format: '{direction}'. Expected 'src2tgt'.")
        src_code, tgt_code = parts
        src_lang = NLLB_FLORES_MAP.get(src_code)
        tgt_lang = NLLB_FLORES_MAP.get(tgt_code)

        if not src_lang or not tgt_lang:
            raise ValueError(f"Unsupported direction for NLLB: '{direction}' ({src_code} -> {tgt_code})")

        self._ensure_loaded()
        sp = self._sp_processor
        translator = self._translator

        # Prepare batch inputs
        non_empty_indices = []
        tokenized_batch = []
        results = [""] * len(texts)

        for i, text in enumerate(texts):
            stripped = text.strip()
            if not stripped:
                results[i] = text
                continue
            non_empty_indices.append(i)
            # NLLB format: [src_lang, ...tokens..., "</s>"]
            tokens = [src_lang, *sp.encode_as_pieces(stripped), "</s>"]
            tokenized_batch.append(tokens)

        if not tokenized_batch:
            return results

        # Run CTranslate2 generation
        target_prefixes = [[tgt_lang]] * len(tokenized_batch)
        translations = translator.translate_batch(
            tokenized_batch,
            target_prefix=target_prefixes,
            batch_type="tokens",
            max_batch_size=1024,
            beam_size=4,
        )

        for idx, trans in zip(non_empty_indices, translations, strict=False):
            hyp = trans.hypotheses[0]
            # Strip target language prefix token if present
            if hyp and hyp[0] == tgt_lang:
                hyp = hyp[1:]
            decoded = sp.decode_pieces(hyp).strip()
            results[idx] = decoded

        return results
