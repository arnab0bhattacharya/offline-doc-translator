"""
engine/run_job.py
=================
Shared translation job execution pipeline.
Used by both CLI (main.py) and GUI (queue_manager.py).
"""

import os
import threading
from collections.abc import Callable
from typing import Any

from formats.registry import get_handler

from .cache import CachePolicy
from .cache_locations import get_job_cache_path
from .core import TranslationEngine, TranslationMode
from .errors import ErrorCode, TranslatorError
from .logging import TranslationLogger
from .preflight import run_nmt_preflight, run_preflight


def execute_translation(
    input_path: str,
    output_path: str,
    direction: str,
    mode: TranslationMode | str,
    model_name: str,
    glossary: dict[str, str],
    progress_cb: Callable[[int, int, str], None] | None = None,
    log_cb: Callable[[str], None] | None = None,
    include_source_text: bool = False,
    policy: Any | None = None,
    cancel_event: threading.Event | None = None,
    logger: TranslationLogger | None = None,
    cache_policy: CachePolicy | str = CachePolicy.ENCRYPTED_PERSISTENT,
) -> dict[str, Any]:
    """
    Runs the full translation pipeline: preflight -> engine -> handler -> translate.
    Returns stats dict: {"total", "translated", "reverted", "skipped"}.
    """
    if not isinstance(mode, TranslationMode):
        mode = TranslationMode(mode)

    if not isinstance(cache_policy, CachePolicy):
        try:
            effective_cache_policy = CachePolicy(cache_policy)
        except (ValueError, TypeError):
            effective_cache_policy = CachePolicy.ENCRYPTED_PERSISTENT
    else:
        effective_cache_policy = cache_policy

    if cancel_event and cancel_event.is_set():
        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled before execution.")

    llm_available = mode == TranslationMode.PURE_LLM
    review_log_path = f"{output_path}.needs_review.log"

    if logger is not None and log_cb is not None:
        logger.add_legacy_callback(log_cb)
    effective_log_cb = log_cb if log_cb is not None else (logger.as_log_cb() if logger is not None else None)

    # 1. Preflight
    if effective_log_cb:
        effective_log_cb("Running system diagnostics & preflight...")
    run_preflight(
        model_name=model_name,
        input_path=input_path,
        output_path=output_path,
        check_model=llm_available,
        require_ollama=llm_available,
    )
    if mode == TranslationMode.FAST_NMT:
        run_nmt_preflight(direction)

    if cancel_event and cancel_event.is_set():
        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled after preflight.")

    # 2. Clear old review log if it exists
    if os.path.exists(review_log_path):
        try:
            os.remove(review_log_path)
        except OSError:
            pass

    # 3. Engine + Cache
    if effective_log_cb:
        effective_log_cb(f"Initializing {mode.value.upper()} engine & persistent cache...")

    target_cache_file = get_job_cache_path(
        output_path=output_path,
        mode=mode,
        cache_policy=effective_cache_policy,
    )
    out_dir = os.path.dirname(os.path.abspath(output_path))
    legacy_candidates = [
        os.path.join(out_dir, ".translation_cache.enc"),
        os.path.join(out_dir, ".translation_cache.json"),
        os.path.join(out_dir, "translation_cache.json"),
        os.path.join(out_dir, "translation_cache.enc"),
    ]

    engine = TranslationEngine(
        model_name=model_name,
        mode=mode,
        glossary=glossary,
        cache_file=target_cache_file,
        cache_policy=effective_cache_policy,
        legacy_cache_candidates=legacy_candidates,
        allow_llm=llm_available,
        include_source_text=include_source_text,
        logger=logger,
    )
    engine.load_cache(direction)

    if cancel_event and cancel_event.is_set():
        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled before handler dispatch.")

    # 4. Handler dispatch
    ext = os.path.splitext(input_path)[1].lower()
    handler = get_handler(ext, engine, policy=policy) if policy is not None else get_handler(ext, engine)

    # 5. Translate
    if effective_log_cb:
        effective_log_cb(f"Translating {ext.upper()} document...")
    stats = handler.translate(
        input_path=input_path,
        output_path=output_path,
        direction=direction,
        review_log_path=review_log_path,
        progress_cb=progress_cb,
        log_cb=effective_log_cb,
        cancel_event=cancel_event,
    )

    return stats
