"""
engine/logging.py
=================
Structured logging subsystem for the offline document translator.
Defines typed TranslationLogEvent and the TranslationLogger event bus,
supporting subscriber registration, level filtering, CLI/GUI formatters,
and seamless bridging to legacy string callbacks.
"""

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from typing import Any

# Standard log levels
LOG_LEVELS = ("debug", "info", "warning", "error")

# Standard event categories
CATEGORIES = ("system", "cache", "translation", "preflight", "format", "backend")


@dataclass
class TranslationLogEvent:
    """
    Typed event payload representing a single logging or telemetry occurrence.
    """

    level: str  # "debug", "info", "warning", "error"
    category: str  # "system", "cache", "translation", "preflight", "format", "backend"
    message: str
    location: str | None = None
    elapsed: float | None = None
    details: dict[str, Any] | None = None
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def to_cli_string(self) -> str:
        """Formats the event into a clean human-readable console string."""
        loc_str = f" {self.location}:" if self.location else ""

        if self.category == "cache":
            return f"  [⚡ Cache]{loc_str} {self.message}"
        elif self.level == "error":
            return f"  [-] Error{loc_str}: {self.message}"
        elif self.level == "warning":
            return f"  [⚠ Warning]{loc_str}: {self.message}"
        elif self.category == "translation":
            if self.elapsed is not None and self.elapsed > 0:
                return f"  [✓ Done in {self.elapsed:.2f}s]{loc_str} {self.message}"
            return f"  [Translation]{loc_str} {self.message}"
        elif self.category == "preflight":
            return f"  [*] Preflight: {self.message}"
        elif self.category == "format":
            return f"  [*] Format{loc_str}: {self.message}"
        else:
            return f"  [*]{loc_str} {self.message}"

    def to_gui_string(self) -> str:
        """Formats the event for GUI log views."""
        return self.to_cli_string().strip()


class TranslationLogger:
    """
    Publisher-subscriber event dispatcher for TranslationLogEvents.
    Supports typed subscribers, standard logging forwarding, and legacy string callbacks.
    """

    def __init__(self, name: str = "translator", min_level: str = "info"):
        self.name = name
        self.min_level = min_level
        self._subscribers: list[Callable[[TranslationLogEvent], None]] = []

    def subscribe(self, callback: Callable[[TranslationLogEvent], None]) -> None:
        """Registers a listener for structured log events."""
        if callback not in self._subscribers:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[TranslationLogEvent], None]) -> None:
        """Removes a registered listener."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def clear_subscribers(self) -> None:
        """Removes all registered listeners."""
        self._subscribers.clear()

    def emit(self, event: TranslationLogEvent) -> None:
        """Broadcasts an event to all subscribers."""
        for cb in list(self._subscribers):
            try:
                cb(event)
            except Exception:
                pass

    def log(
        self,
        level: str,
        category: str,
        message: str,
        location: str | None = None,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranslationLogEvent:
        event = TranslationLogEvent(
            level=level,
            category=category,
            message=message,
            location=location,
            elapsed=elapsed,
            details=details,
        )
        self.emit(event)
        return event

    def info(
        self,
        message: str,
        category: str = "system",
        location: str | None = None,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranslationLogEvent:
        return self.log("info", category, message, location=location, elapsed=elapsed, details=details)

    def warning(
        self,
        message: str,
        category: str = "system",
        location: str | None = None,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranslationLogEvent:
        return self.log("warning", category, message, location=location, elapsed=elapsed, details=details)

    def error(
        self,
        message: str,
        category: str = "system",
        location: str | None = None,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranslationLogEvent:
        return self.log("error", category, message, location=location, elapsed=elapsed, details=details)

    def debug(
        self,
        message: str,
        category: str = "system",
        location: str | None = None,
        elapsed: float | None = None,
        details: dict[str, Any] | None = None,
    ) -> TranslationLogEvent:
        return self.log("debug", category, message, location=location, elapsed=elapsed, details=details)

    def add_legacy_callback(self, cb: Callable[[str], None]) -> None:
        """Bridges a legacy string callback function(msg) by subscribing a formatter wrapper."""

        def adapter(event: TranslationLogEvent):
            cb(event.to_cli_string())

        self.subscribe(adapter)

    def as_log_cb(self, category: str = "system") -> Callable[[str], None]:
        """Returns a legacy log_cb(str) function that converts raw strings into structured events."""

        def legacy_cb(raw_msg: str):
            msg = raw_msg.strip()
            level = "info"
            cat = category
            if "[!]" in msg or "Warning" in msg or "[⚠" in msg:
                level = "warning"
            elif "[-]" in msg or "error" in msg.lower() or "[❌" in msg:
                level = "error"
            elif "[⚡ Cache]" in msg:
                cat = "cache"
            elif "[*]" in msg:
                level = "info"

            self.emit(
                TranslationLogEvent(
                    level=level,
                    category=cat,
                    message=raw_msg,
                )
            )

        return legacy_cb


# Global default logger instance
_default_logger: TranslationLogger | None = None


def get_logger(name: str = "translator") -> TranslationLogger:
    """Returns the global or a named TranslationLogger instance."""
    global _default_logger
    if _default_logger is None:
        _default_logger = TranslationLogger(name=name)
    return _default_logger


def set_default_logger(logger: TranslationLogger) -> None:
    """Overrides the default global TranslationLogger."""
    global _default_logger
    _default_logger = logger
