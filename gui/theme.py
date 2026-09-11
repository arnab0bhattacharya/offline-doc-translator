"""
gui/theme.py
============
Theme palette, language pairs, model presets, and shared GUI parsing utilities.
"""

# ── Color System (Dual Light/Dark Mode Tuples) ────────────────────
THEME = {
    "bg": ("gray94", "#0F172A"),
    "sidebar_bg": ("gray90", "#0B1120"),
    "card_bg": ("white", "#1E293B"),
    "card_border": ("gray80", "#334155"),
    "staging_bg": ("gray96", "#0F172A"),
    "text_primary": ("gray10", "#F8FAFC"),
    "text_secondary": ("gray45", "#94A3B8"),
    "primary": ("#2563EB", "#3B82F6"),
    "primary_hover": ("#1D4ED8", "#2563EB"),
    "success": ("#16A34A", "#22C55E"),
    "warning": ("#D97706", "#F59E0B"),
    "error": ("#DC2626", "#EF4444"),
    "btn_secondary": ("gray85", "#334155"),
    "btn_sec_hover": ("gray75", "#475569"),
    "log_bg": ("gray96", "#090D16"),
    "log_fg": ("gray20", "#CBD5E1"),
    # Badges
    "badge_pptx": "#EA580C",
    "badge_xlsx": "#16A34A",
    "badge_docx": "#2563EB",
    "badge_pdf": "#DC2626",
}

LANGUAGE_PAIRS = [
    ("Japanese → English", "ja2en"),
    ("English → Japanese", "en2ja"),
]

GEMMA_PRESETS = [
    "gemma4:e2b-it-qat",
    "gemma4:12b-it-qat",
    "gemma4:27b-it-qat",
]

import logging
from collections.abc import Callable

logger = logging.getLogger("offline_translator.glossary")

MAX_GLOSSARY_ENTRIES: int = 10_000
MAX_TERM_LENGTH: int = 200


def parse_glossary_text(
    text: str,
    on_warning: Callable[[str], None] | None = None,
) -> dict[str, str]:
    """
    Parses key-value glossary lines supporting ->, :, or = delimiters.
    Enforces security & performance limits from Codex Finding 3:
      - Max 10,000 entries
      - Max 200 characters per term (source or target)
      - Logs warning when limits are exceeded.
    """
    glossary: dict[str, str] = {}
    skipped_length = 0
    limit_reached = False

    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "->" in line:
            parts = line.split("->", 1)
        elif ":" in line:
            parts = line.split(":", 1)
        elif "=" in line:
            parts = line.split("=", 1)
        else:
            continue
        src = parts[0].strip()
        tgt = parts[1].strip()
        if not src or not tgt:
            continue

        if len(src) > MAX_TERM_LENGTH or len(tgt) > MAX_TERM_LENGTH:
            skipped_length += 1
            continue

        if len(glossary) >= MAX_GLOSSARY_ENTRIES:
            if not limit_reached:
                limit_reached = True
                warn_msg = f"Glossary exceeded maximum limit of {MAX_GLOSSARY_ENTRIES} entries. Remaining entries were ignored."
                logger.warning(warn_msg)
                if on_warning:
                    on_warning(warn_msg)
            break

        glossary[src] = tgt

    if skipped_length > 0:
        warn_msg = f"Skipped {skipped_length} glossary term(s) exceeding {MAX_TERM_LENGTH} characters."
        logger.warning(warn_msg)
        if on_warning:
            on_warning(warn_msg)

    return glossary


def format_glossary_text(glossary: dict[str, str]) -> str:
    """Formats a dictionary back to standard glossary text representation."""
    lines = ["# Term -> Translation (one per line)"]
    for k, v in glossary.items():
        lines.append(f"{k} -> {v}")
    return "\n".join(lines) + "\n"
