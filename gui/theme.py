"""
gui/theme.py
============
Theme palette, language pairs, model presets, and shared GUI parsing utilities.
"""

# ── Color System (Dual Light/Dark Mode Tuples) ────────────────────
# Each value is a (light_mode, dark_mode) tuple compatible with
# CustomTkinter's native dual-mode color handling.
THEME = {
    "bg": ("#F8FAFC", "#0F172A"),
    "sidebar_bg": ("#FFFFFF", "#0B1120"),
    "card_bg": ("#FFFFFF", "#1E293B"),
    "card_border": ("#E2E8F0", "#334155"),
    "staging_bg": ("#F1F5F9", "#0F172A"),
    "text_primary": ("#0F172A", "#F8FAFC"),
    "text_secondary": ("#64748B", "#94A3B8"),
    "primary": ("#2563EB", "#3B82F6"),
    "primary_hover": ("#1D4ED8", "#2563EB"),
    "success": ("#16A34A", "#22C55E"),
    "warning": ("#D97706", "#F59E0B"),
    "error": ("#DC2626", "#EF4444"),
    "btn_secondary": ("#E2E8F0", "#334155"),
    "btn_sec_hover": ("#CBD5E1", "#475569"),
    "btn_sec_text": ("#334155", "#E2E8F0"),
    "log_bg": ("#F8FAFC", "#090D16"),
    "log_fg": ("#334155", "#CBD5E1"),
    "review_hover": ("#B45309", "#B45309"),
    "cancel_hover": ("#991B1B", "#991B1B"),
    # Badges (single color — high contrast on white text in both modes)
    "badge_pptx": "#EA580C",
    "badge_xlsx": "#16A34A",
    "badge_docx": "#2563EB",
    "badge_pdf": "#DC2626",
}

# ── Spacing Constants ─────────────────────────────────────────────
SPACING = {
    "xs": 4,
    "sm": 8,
    "md": 16,
    "lg": 24,
    "xl": 32,
    "section_gap": 20,
    "card_pad": 16,
    "sidebar_pad": 16,
}

LANGUAGE_PAIRS = [
    ("Japanese → English", "ja2en"),
    ("English → Japanese", "en2ja"),
]

PINNED_OLLAMA_MODEL = "gemma4:e2b-it-qat"

GEMMA_PRESETS = [
    PINNED_OLLAMA_MODEL,
]

import logging
import os
from collections.abc import Callable

logger = logging.getLogger("offline_translator.glossary")

MAX_GLOSSARY_ENTRIES: int = 10_000
MAX_TERM_LENGTH: int = 200
GLOSSARY_DIR: str = os.path.expanduser("~/.offline-translator")
LAST_GLOSSARY_PATH: str = os.path.join(GLOSSARY_DIR, "last_glossary.txt")


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
