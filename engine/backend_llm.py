"""
engine/backend_llm.py
=====================
Ollama LLM backend supporting single-chunk translation and Macro-Batch Polishing
with custom Glossary injection and automatic draft fallback.
"""

import re
import json
import time
import requests
from typing import Dict, List, Tuple, Optional, Any, Callable

from engine.core import verify_placeholders, clean_llm_response, build_prompts


class LLMBackend:
    """
    Handles Ollama inference, single-prompt isomorphic retries,
    and multi-line Macro Polishing with Glossary enforcement.
    """
    name: str = "llm"

    def __init__(
        self,
        model_name: str = "gemma4:e2b-it-qat",
        ollama_url: str = "http://localhost:11434",
        context_window: int = 2048
    ):
        self.model_name = model_name
        self.ollama_url = ollama_url.rstrip("/")
        self.generate_url = f"{self.ollama_url}/api/generate"
        self.context_window = context_window

    def is_available(self) -> bool:
        """Returns True if Ollama service is reachable."""
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=2.0)
            return r.status_code == 200
        except Exception:
            return False

    def is_ready(self, direction: str) -> bool:
        """Returns True if Ollama is reachable and ready for translation."""
        return self.is_available()

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: Optional[Dict[str, str]] = None,
        context: Optional[str] = None,
        log_cb: Optional[Callable[[str], None]] = None,
    ) -> Tuple[Optional[str], float]:
        """
        Unified TranslationBackend protocol method.
        Executes standard 2-attempt translation with isomorphic few-shot fallback.
        """
        return self.translate_single(
            masked_text=text,
            number_map=placeholder_map or {},
            direction=direction,
            context=context,
            log_cb=log_cb,
        )

    def translate_single(
        self,
        masked_text: str,
        number_map: Dict[str, str],
        direction: str,
        context: Optional[str] = None,
        log_cb: Optional[Callable[[str], None]] = None
    ) -> Tuple[Optional[str], float]:
        """
        Executes standard 2-attempt translation with isomorphic few-shot fallback.
        Returns: (translated_masked_text, elapsed_seconds)
        """
        prompt_1, prompt_2 = build_prompts(direction, masked_text, number_map, context)

        # Attempt 1 (T=0.1)
        t0 = time.time()
        payload_1 = {
            "model": self.model_name,
            "prompt": prompt_1,
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": self.context_window, "temperature": 0.1},
        }

        try:
            res_1 = requests.post(self.generate_url, json=payload_1, timeout=120)
            elapsed_1 = time.time() - t0
            if res_1.status_code == 200:
                raw_res = res_1.json().get("response", "").strip()
                result = clean_llm_response(raw_res)
                if verify_placeholders(result, number_map, masked_text):
                    return result, elapsed_1
        except Exception:
            pass

        # Attempt 2 (T=0.0 strict isomorphic few-shot)
        t1 = time.time()
        payload_2 = {
            "model": self.model_name,
            "prompt": prompt_2,
            "stream": False,
            "keep_alive": "10m",
            "options": {"num_ctx": self.context_window, "temperature": 0.0},
        }

        try:
            res_2 = requests.post(self.generate_url, json=payload_2, timeout=120)
            elapsed_2 = time.time() - t1
            if res_2.status_code == 200:
                raw_res = res_2.json().get("response", "").strip()
                result = clean_llm_response(raw_res)
                if verify_placeholders(result, number_map, masked_text):
                    return result, elapsed_2
        except Exception:
            pass

        return None, 0.0
