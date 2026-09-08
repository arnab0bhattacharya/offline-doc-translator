"""
engine/errors.py
================
Standardized error codes (E01-E07) and user-facing exception definitions.
Designed to provide clear, actionable feedback for non-technical users.
"""

from enum import Enum
from typing import Optional


class ErrorCode(str, Enum):
    E01 = "E01"  # Translation engine (Ollama) is not running
    E02 = "E02"  # Required AI model is missing / not pulled
    E03 = "E03"  # Insufficient free memory (RAM)
    E04 = "E04"  # File unreadable, corrupted, or unsupported
    E05 = "E05"  # Output file cannot be saved (permission / locked)
    E06 = "E06"  # Translation finished with items needing manual review
    E07 = "E07"  # Insufficient disk space for workspace/cache
    E08 = "E08"  # Required local NMT package is unavailable
    E09 = "E09"  # Translation was cancelled by user


ERROR_MESSAGES = {
    ErrorCode.E01: {
        "title": "Translation Engine Offline",
        "message": "The local AI engine (Ollama) is not running.",
        "action": "Please start the Ollama application or run 'ollama serve' in your terminal."
    },
    ErrorCode.E02: {
        "title": "AI Model Not Found",
        "message": "The specified translation model is not installed.",
        "action": "Open terminal and run: ollama pull <model_name>"
    },
    ErrorCode.E03: {
        "title": "Low System Memory",
        "message": "Available RAM is below the safety threshold required for translation.",
        "action": "Close heavy applications to free up RAM before starting."
    },
    ErrorCode.E04: {
        "title": "Invalid or Corrupt File",
        "message": "Could not read or parse the selected document.",
        "action": "Check that the file is not password-protected or corrupted, and has a supported extension (.pptx, .xlsx, .docx, .pdf)."
    },
    ErrorCode.E05: {
        "title": "File Save Error",
        "message": "Cannot write the translated document to the output path.",
        "action": "Make sure the output file is not currently open in PowerPoint, Excel, Word, or another app."
    },
    ErrorCode.E06: {
        "title": "Items Need Review",
        "message": "Translation completed, but some complex elements failed validation.",
        "action": "Review the generated 'needs_review.log' file to see items kept in original language."
    },
    ErrorCode.E07: {
        "title": "Low Disk Space",
        "message": "Not enough free disk space for temporary workspace and cache.",
        "action": "Free up at least 500 MB of space on your main drive."
    },
    ErrorCode.E08: {
        "title": "Offline Translation Model Not Ready",
        "message": "The required local Japanese-English NMT package is not installed or cannot be used.",
        "action": "Install the required Argos Translate language package, then retry Fast NMT mode."
    },
    ErrorCode.E09: {
        "title": "Translation Cancelled",
        "message": "The translation was cancelled by user request.",
        "action": "You can re-add the document to the queue if you wish to translate it later."
    },
}


class TranslatorError(Exception):
    """Custom exception containing an error code, technical details, and user guidance."""

    def __init__(self, code: ErrorCode, detail: str = "", original_exc: Optional[Exception] = None):
        self.code = code
        self.detail = detail
        self.original_exc = original_exc
        meta = ERROR_MESSAGES.get(code, {})
        self.title = meta.get("title", "Unknown Error")
        self.user_message = meta.get("message", "An unexpected error occurred.")
        self.action = meta.get("action", "Please check system logs.")
        super().__init__(f"[{code.value}] {self.title}: {self.user_message} ({detail})")

    def format_user_dialog(self) -> str:
        """Formats a clean message string suitable for GUI error dialogs."""
        return (
            f"Error Code: {self.code.value} - {self.title}\n\n"
            f"{self.user_message}\n\n"
            f"Suggested Action: {self.action}\n"
            + (f"\nTechnical details: {self.detail}" if self.detail else "")
        )
