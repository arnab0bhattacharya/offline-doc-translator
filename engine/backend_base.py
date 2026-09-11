"""
engine/backend_base.py
======================
Formal protocol definition for translation backends.
Enables polymorphic dispatch across NMT, LLM, and future custom backends.
"""

from collections.abc import Callable
from typing import Protocol, runtime_checkable


@runtime_checkable
class TranslationBackend(Protocol):
    """
    Formal interface defining the contract that all translation backends must implement.
    """

    name: str

    def is_available(self) -> bool:
        """Returns True if the backend dependencies/services are installed or reachable."""
        ...

    def is_ready(self, direction: str) -> bool:
        """Returns True if the backend is fully prepared to execute translation for the specified direction."""
        ...

    def translate(
        self,
        text: str,
        direction: str,
        placeholder_map: dict[str, str] | None = None,
        context: str | None = None,
        log_cb: Callable[[str], None] | None = None,
    ) -> tuple[str | None, float]:
        """
        Translates a single text unit (optionally containing placeholders).

        Args:
            text: Source text to translate (may contain [[N0]], [[GLOSSARY_A]], etc.).
            direction: Translation direction ('ja2en', 'en2ja').
            placeholder_map: Optional mapping of placeholder tokens to original values.
            context: Optional contextual reference string.
            log_cb: Optional callback for diagnostic telemetry.

        Returns:
            Tuple of (translated_text_or_None, elapsed_seconds).
        """
        ...
