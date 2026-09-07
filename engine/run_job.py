"""
engine/run_job.py
=================
Shared translation job execution pipeline.
Used by both CLI (main.py) and GUI (queue_manager.py).
"""
import os
from typing import Dict, Optional, Callable, Any, Union
from .core import TranslationEngine, TranslationMode
from .preflight import run_preflight, run_nmt_preflight
from formats.registry import get_handler


def execute_translation(
    input_path: str,
    output_path: str,
    direction: str,
    mode: Union[TranslationMode, str],
    model_name: str,
    glossary: Dict[str, str],
    progress_cb: Optional[Callable[[int, int, str], None]] = None,
    log_cb: Optional[Callable[[str], None]] = None,
    include_source_text: bool = False,
) -> Dict[str, Any]:
    """
    Runs the full translation pipeline: preflight -> engine -> handler -> translate.
    Returns stats dict: {"total", "translated", "reverted", "skipped"}.
    """
    if not isinstance(mode, TranslationMode):
        mode = TranslationMode(mode)

    llm_available = (mode == TranslationMode.PURE_LLM)
    review_log_path = f"{output_path}.needs_review.log"

    # 1. Preflight
    if log_cb:
        log_cb("Running system diagnostics & preflight...")
    run_preflight(
        model_name=model_name,
        input_path=input_path,
        output_path=output_path,
        check_model=llm_available,
        require_ollama=llm_available,
    )
    if mode == TranslationMode.FAST_NMT:
        run_nmt_preflight(direction)

    # 2. Clear old review log if it exists
    if os.path.exists(review_log_path):
        try:
            os.remove(review_log_path)
        except OSError:
            pass

    # 3. Engine + Cache
    if log_cb:
        log_cb(f"Initializing {mode.value.upper()} engine & persistent cache...")
    engine = TranslationEngine(
        model_name=model_name,
        mode=mode,
        glossary=glossary,
        cache_file=os.path.join(
            os.path.dirname(os.path.abspath(output_path)),
            ".translation_cache.json",
        ),
        allow_llm=llm_available,
        include_source_text=include_source_text,
    )
    engine.load_cache(direction)


    # 4. Handler dispatch
    ext = os.path.splitext(input_path)[1].lower()
    handler = get_handler(ext, engine)

    # 5. Translate
    if log_cb:
        log_cb(f"Translating {ext.upper()} document...")
    stats = handler.translate(
        input_path=input_path,
        output_path=output_path,
        direction=direction,
        review_log_path=review_log_path,
        progress_cb=progress_cb,
        log_cb=log_cb,
    )

    return stats

