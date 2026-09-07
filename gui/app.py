"""
gui/app.py
==========
Modern, Professional Desktop Interface for Offline Document Translator.
Designed with DeepL / Native Desktop aesthetics:
  - Sidebar + Fluid Workspace layout
  - Multi-document batch selection & directory scanning
  - True Dynamic Theming (Native Dark & Light mode support)
  - Proportional, centered cards (no awkward stretched bars)
  - Dedicated Start Translation action button
  - Interactive queue with live progress, format badges, and result openers
  - Quick Translate split-pane powered by Ollama
  - System & Diagnostics tab with 1-click Argos Package Installer
  - Zero-load Ollama startup (only queries metadata on launch)
  - Windows AppUserModelID & custom taskbar / window icons
"""

import os
import sys
import time
import shutil
import threading
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox
from typing import Optional, Dict, List

import requests
import customtkinter as ctk

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Set Windows AppUserModelID so taskbar displays our custom icon
if sys.platform == "win32":
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("offline.document.translator.v2")
    except Exception:
        pass

try:
    from engine.errors import ErrorCode, TranslatorError
    from engine.preflight import (
        check_ollama_status,
        list_installed_models,
        run_preflight,
        run_nmt_preflight,
        check_ram,
        check_disk_space,
    )
    from engine.core import TranslationEngine, TranslationMode, DIRECTIONS, clean_llm_response
    from engine.queue_manager import TranslationQueue, TranslationJob, JobStatus
    from engine.backend_nmt import NMTBackend
    from formats.registry import get_handler, SUPPORTED_EXTENSIONS
except (ImportError, ValueError):
    from ..engine.errors import ErrorCode, TranslatorError
    from ..engine.preflight import (
        check_ollama_status,
        list_installed_models,
        run_preflight,
        run_nmt_preflight,
        check_ram,
        check_disk_space,
    )
    from ..engine.core import TranslationEngine, TranslationMode, DIRECTIONS, clean_llm_response
    from ..engine.queue_manager import TranslationQueue, TranslationJob, JobStatus
    from ..engine.backend_nmt import NMTBackend
    from ..formats.registry import get_handler, SUPPORTED_EXTENSIONS


# ── Color System (Dual Light/Dark Mode Tuples) ────────────────────
THEME = {
    "bg":            ("gray94", "#0F172A"),
    "sidebar_bg":    ("gray90", "#0B1120"),
    "card_bg":       ("white",  "#1E293B"),
    "card_border":   ("gray80", "#334155"),
    "staging_bg":    ("gray96", "#0F172A"),
    "text_primary":  ("gray10", "#F8FAFC"),
    "text_secondary":("gray45", "#94A3B8"),
    "primary":       ("#2563EB", "#3B82F6"),
    "primary_hover": ("#1D4ED8", "#2563EB"),
    "success":       ("#16A34A", "#22C55E"),
    "warning":       ("#D97706", "#F59E0B"),
    "error":         ("#DC2626", "#EF4444"),
    "btn_secondary": ("gray85", "#334155"),
    "btn_sec_hover": ("gray75", "#475569"),
    "log_bg":        ("gray96", "#090D16"),
    "log_fg":        ("gray20", "#CBD5E1"),
    # Badges
    "badge_pptx":    "#EA580C",
    "badge_xlsx":    "#16A34A",
    "badge_docx":    "#2563EB",
    "badge_pdf":     "#DC2626",
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


def parse_glossary_text(text: str) -> Dict[str, str]:
    glossary = {}
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
        if src and tgt:
            glossary[src] = tgt
    return glossary


class TranslatorApp:
    """Modern Desktop UI for Offline Document Translator with Sidebar Navigation."""

    def __init__(self, root: ctk.CTk):
        self.root = root
        self.root.title("Offline Document Translator")
        self.root.geometry("1060x820")
        self.root.minsize(860, 680)

        # Set default appearance
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        # Setup icon
        self._setup_window_icon()

        # State
        self.nmt_backend = NMTBackend()
        self.selected_files: List[str] = []
        self._last_output_path: Optional[str] = None
        self._last_review_log: Optional[str] = None
        self._job_widgets: Dict[str, Dict] = {}
        self.active_tab = "docs"

        # Translation Queue
        self.translation_queue = TranslationQueue(
            on_job_update=self._on_job_update,
            on_log=self._on_job_log
        )

        self._build_layout()

        # Initial check (Zero-load metadata only)
        self.root.after(100, self._check_system_status)

    def _setup_window_icon(self):
        """Discovers and attaches application icon across all environments."""
        search_dirs = [
            PROJECT_ROOT,
            os.path.join(PROJECT_ROOT, "assets"),
            getattr(sys, "_MEIPASS", ""),
            os.path.join(getattr(sys, "_MEIPASS", ""), "assets"),
            os.path.dirname(sys.executable),
            os.path.join(os.path.dirname(sys.executable), "assets"),
            os.path.join(os.path.dirname(sys.executable), "_internal", "assets"),
        ]

        # 1. Try native .ico
        for d in search_dirs:
            if not d:
                continue
            ico = os.path.join(d, "icon.ico") if not d.endswith("icon.ico") else d
            if os.path.isfile(ico):
                try:
                    self.root.iconbitmap(ico)
                    break
                except Exception:
                    pass

        # 2. Try PNG fallback
        for d in search_dirs:
            if not d:
                continue
            png = os.path.join(d, "icon.png") if not d.endswith("icon.png") else d
            if os.path.isfile(png):
                try:
                    from PIL import ImageTk, Image
                    img = ImageTk.PhotoImage(Image.open(png))
                    self.root.iconphoto(True, img)
                    break
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════
    #  MAIN LAYOUT (Sidebar + Content Workspace)
    # ══════════════════════════════════════════════════════════════

    def _build_layout(self):
        self.root.configure(fg_color=THEME["bg"])

        # Main horizontal container
        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)

        # Left Navigation Sidebar
        self._build_sidebar()

        # Right Content Area
        self.content_area = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_area.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)

        # Tab Frames
        self.tab_frames: Dict[str, ctk.CTkFrame] = {}

        self._build_documents_view()
        self._build_quick_view()
        self._build_system_view()

        # Show initial tab
        self._switch_tab("docs")

    # ─────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(
            self.main_container, width=230, corner_radius=0,
            fg_color=THEME["sidebar_bg"]
        )
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        # ── App Branding ──
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(20, 24))

        ctk.CTkLabel(
            brand_frame, text="🌐  DocTranslator",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=THEME["text_primary"], anchor="w"
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_frame, text="Offline Neural & Local LLM",
            font=ctk.CTkFont(size=11), text_color=THEME["text_secondary"], anchor="w"
        ).pack(anchor="w", pady=(2, 0))

        # ── Navigation Buttons ──
        self.nav_buttons = {}

        nav_items = [
            ("docs",   "📄  Documents",        "Batch translate PPTX, XLSX, DOCX, PDF"),
            ("quick",  "⚡  Quick Translate",  "Instant side-by-side text lookup"),
            ("system", "⚙  System & AI",       "Engines, models, and diagnostics"),
        ]

        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(fill="x", padx=12, pady=0)

        for tab_id, label, desc in nav_items:
            btn = ctk.CTkButton(
                nav_frame, text=f"  {label}", height=42, anchor="w",
                font=ctk.CTkFont(size=13, weight="bold"),
                corner_radius=8,
                fg_color="transparent",
                text_color=THEME["text_secondary"],
                hover_color=THEME["btn_secondary"],
                command=lambda t=tab_id: self._switch_tab(t)
            )
            btn.pack(fill="x", pady=4)
            self.nav_buttons[tab_id] = btn

        # ── Spacer ──
        ctk.CTkFrame(self.sidebar, fg_color="transparent").pack(fill="both", expand=True)

        # ── Sidebar Status Card ──
        status_card = ctk.CTkFrame(self.sidebar, fg_color=THEME["card_bg"], corner_radius=8)
        status_card.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            status_card, text="AI Engine Status",
            font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["text_secondary"]
        ).pack(anchor="w", padx=10, pady=(8, 4))

        # Argos indicator
        self.side_argos_btn = ctk.CTkButton(
            status_card, text="● Argos: Checking...", height=24, anchor="w",
            font=ctk.CTkFont(size=11), fg_color="transparent", text_color=THEME["warning"],
            hover_color=THEME["btn_secondary"], command=lambda: self._switch_tab("system")
        )
        self.side_argos_btn.pack(fill="x", padx=6, pady=2)

        # Ollama indicator
        self.side_ollama_btn = ctk.CTkButton(
            status_card, text="● Ollama: Checking...", height=24, anchor="w",
            font=ctk.CTkFont(size=11), fg_color="transparent", text_color=THEME["warning"],
            hover_color=THEME["btn_secondary"], command=self._refresh_ollama_status
        )
        self.side_ollama_btn.pack(fill="x", padx=6, pady=(0, 6))

        # ── Theme Toggle in Sidebar Footer ──
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(0, 16))

        self.theme_switch = ctk.CTkSwitch(
            footer, text="Dark Theme", font=ctk.CTkFont(size=11),
            command=self._toggle_theme
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="left")

    def _switch_tab(self, tab_id: str):
        self.active_tab = tab_id
        for tid, frame in self.tab_frames.items():
            if tid == tab_id:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        # Update button highlights
        for tid, btn in self.nav_buttons.items():
            if tid == tab_id:
                btn.configure(
                    fg_color=THEME["primary"],
                    text_color="#FFFFFF"
                )
            else:
                btn.configure(
                    fg_color="transparent",
                    text_color=THEME["text_secondary"]
                )

    # ─────────────────────────────────────────────────────────────
    #  TAB 1: DOCUMENT TRANSLATION WORKSPACE
    # ─────────────────────────────────────────────────────────────

    def _build_documents_view(self):
        tab = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.tab_frames["docs"] = tab

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── Page Header ──
        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            header, text="Document Translation",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            header, text="Select single or batch documents (PPTX, XLSX, DOCX, PDF) to translate offline.",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        # ── Card 1: Staging & Document Selection ──
        card_sel = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_sel.pack(fill="x", pady=(0, 14))

        # Drop/Upload Target Area
        upload_area = ctk.CTkFrame(card_sel, fg_color=THEME["staging_bg"], corner_radius=10)
        upload_area.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            upload_area, text="📂", font=ctk.CTkFont(size=28)
        ).pack(pady=(12, 4))

        ctk.CTkLabel(
            upload_area, text="Add Documents to Translate",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_primary"]
        ).pack()

        ctk.CTkLabel(
            upload_area, text="Choose multiple individual files or scan an entire folder",
            font=ctk.CTkFont(size=11), text_color=THEME["text_secondary"]
        ).pack(pady=(2, 10))

        # Buttons inside target area
        target_btn_row = ctk.CTkFrame(upload_area, fg_color="transparent")
        target_btn_row.pack(pady=(0, 14))

        ctk.CTkButton(
            target_btn_row, text="  Select Files...  ", height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"], hover_color=THEME["primary_hover"],
            command=self._browse_multi_files
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            target_btn_row, text="  Select Folder...  ", height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._browse_folder
        ).pack(side="left", padx=6)

        # Selected files container
        self.staged_container = ctk.CTkFrame(card_sel, fg_color="transparent")
        self.staged_container.pack(fill="x", padx=16, pady=(0, 14))

        self.staged_header = ctk.CTkFrame(self.staged_container, fg_color="transparent")
        self.staged_header.pack(fill="x", pady=(0, 6))

        self.staged_count_label = ctk.CTkLabel(
            self.staged_header, text="0 documents staged",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]
        ).pack(side="left")

        ctk.CTkButton(
            self.staged_header, text="Clear All", width=70, height=22,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color=THEME["text_secondary"], hover_color=THEME["btn_secondary"],
            command=self._clear_selected_files
        ).pack(side="right")

        self.staged_list_frame = ctk.CTkFrame(self.staged_container, fg_color=THEME["staging_bg"], corner_radius=8)
        self.staged_list_frame.pack(fill="x")

        self.empty_staged_label = ctk.CTkLabel(
            self.staged_list_frame, text="No documents selected yet.",
            font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"], pady=12
        )
        self.empty_staged_label.pack()

        # ── Card 2: Configuration & Direction Bar ──
        card_opt = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_opt.pack(fill="x", pady=(0, 14))

        opt_inner = ctk.CTkFrame(card_opt, fg_color="transparent")
        opt_inner.pack(fill="x", padx=16, pady=14)

        # Controls Row
        ctrl_row = ctk.CTkFrame(opt_inner, fg_color="transparent")
        ctrl_row.pack(fill="x")

        # Engine Segmented Switch
        engine_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        engine_box.pack(side="left", fill="y", padx=(0, 16))

        ctk.CTkLabel(
            engine_box, text="ENGINE", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        self.mode_var = ctk.StringVar(value=TranslationMode.FAST_NMT.value)
        self.mode_seg = ctk.CTkSegmentedButton(
            engine_box,
            values=["⚡ Fast NMT (Offline)", "🧠 Pure LLM (Ollama)"],
            command=self._on_mode_seg_changed,
            font=ctk.CTkFont(size=12, weight="bold")
        )
        self.mode_seg.set("⚡ Fast NMT (Offline)")
        self.mode_seg.pack()

        # Direction Box
        dir_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        dir_box.pack(side="left", fill="y", padx=(0, 16))

        ctk.CTkLabel(
            dir_box, text="DIRECTION", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        dir_sub = ctk.CTkFrame(dir_box, fg_color="transparent")
        dir_sub.pack()

        self.direction_var = ctk.StringVar(value="ja2en")
        dir_display_values = [p[0] for p in LANGUAGE_PAIRS]
        self.direction_combo = ctk.CTkComboBox(
            dir_sub, values=dir_display_values,
            variable=ctk.StringVar(value=dir_display_values[0]),
            width=180, state="readonly", font=ctk.CTkFont(size=12),
            command=self._on_direction_changed
        )
        self.direction_combo.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            dir_sub, text="⇄", width=32, height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._swap_doc_direction
        ).pack(side="left")

        # Model Box (Dims when Fast NMT)
        self.model_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        self.model_box.pack(side="left", fill="y")

        ctk.CTkLabel(
            self.model_box, text="OLLAMA MODEL", font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"]
        ).pack(anchor="w", pady=(0, 4))

        self.model_var = ctk.StringVar(value="gemma4:e2b-it-qat")
        self.model_combo = ctk.CTkComboBox(
            self.model_box, values=GEMMA_PRESETS,
            variable=self.model_var, width=190, state="disabled",
            font=ctk.CTkFont(size=12)
        )
        self.model_combo.pack()

        # Collapsible Glossary in Card 2
        glossary_toggle_frame = ctk.CTkFrame(opt_inner, fg_color="transparent")
        glossary_toggle_frame.pack(fill="x", pady=(12, 0))

        self.glossary_btn = ctk.CTkButton(
            glossary_toggle_frame, text="▶  Custom Glossary (Optional)",
            anchor="w", fg_color="transparent", hover_color=THEME["btn_secondary"],
            text_color=THEME["text_secondary"], font=ctk.CTkFont(size=12),
            command=self._toggle_glossary
        )
        self.glossary_btn.pack(side="left")

        self.glossary_drawer = ctk.CTkFrame(opt_inner, fg_color="transparent")
        self.glossary_text = ctk.CTkTextbox(
            self.glossary_drawer, height=70, font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=THEME["staging_bg"]
        )
        self.glossary_text.pack(fill="x", pady=(6, 0))
        self.glossary_text.insert("0.0", "# Term -> Translation (one per line)\n")
        self._glossary_open = False

        # ── Primary Action Row ──
        action_bar = ctk.CTkFrame(scroll, fg_color="transparent")
        action_bar.pack(fill="x", pady=(2, 14))

        self.start_btn = ctk.CTkButton(
            action_bar, text="▶   Start Translation",
            width=220, height=44, corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["primary"], hover_color=THEME["primary_hover"],
            command=self._start_translation_flow
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.add_queue_btn = ctk.CTkButton(
            action_bar, text="➕  Add to Queue Only",
            width=160, height=44, corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._add_to_queue_only
        )
        self.add_queue_btn.pack(side="left")

        # ── Card 3: Queue & Active Jobs ──
        card_q = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_q.pack(fill="x", pady=(0, 14))

        q_head = ctk.CTkFrame(card_q, fg_color="transparent")
        q_head.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            q_head, text="Translation Queue & Progress",
            font=ctk.CTkFont(size=14, weight="bold"), text_color=THEME["text_primary"]
        ).pack(side="left")

        ctk.CTkButton(
            q_head, text="Clear Finished", width=90, height=24,
            font=ctk.CTkFont(size=11), fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._clear_completed_jobs
        ).pack(side="right")

        self.queue_container = ctk.CTkFrame(card_q, fg_color="transparent")
        self.queue_container.pack(fill="x", padx=16, pady=(0, 12))

        self.queue_empty_label = ctk.CTkLabel(
            self.queue_container, text="No documents in queue. Select files above and click 'Start Translation'.",
            font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"], pady=14
        )
        self.queue_empty_label.pack()

        # ── Card 4: Post-Actions & Activity Drawer ──
        bottom_box = ctk.CTkFrame(scroll, fg_color="transparent")
        bottom_box.pack(fill="x", pady=(0, 16))

        self.open_file_btn = ctk.CTkButton(
            bottom_box, text="📄  Open Output File", width=140, height=32, state="disabled",
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._open_last_output
        )
        self.open_file_btn.pack(side="left", padx=(0, 8))

        self.open_dir_btn = ctk.CTkButton(
            bottom_box, text="📁  Open Folder", width=120, height=32, state="disabled",
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._open_last_dir
        )
        self.open_dir_btn.pack(side="left", padx=(0, 8))

        self.review_btn = ctk.CTkButton(
            bottom_box, text="⚠  Review Log", width=120, height=32, state="disabled",
            fg_color=THEME["warning"], hover_color="#B45309",
            command=self._view_review_log
        )
        self.review_btn.pack(side="left")

        self.log_drawer_btn = ctk.CTkButton(
            bottom_box, text="Activity Log ▶", width=110, height=32,
            fg_color="transparent", text_color=THEME["text_secondary"], hover_color=THEME["btn_secondary"],
            command=self._toggle_log
        )
        self.log_drawer_btn.pack(side="right")

        # Collapsible log
        self.log_drawer = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        self.log_text = ctk.CTkTextbox(
            self.log_drawer, height=130, font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=THEME["log_bg"], text_color=THEME["log_fg"], state="disabled"
        )
        self.log_text.pack(fill="x", padx=12, pady=12)
        self._log_open = False

    # ─────────────────────────────────────────────────────────────
    #  TAB 2: QUICK TRANSLATE WORKSPACE
    # ─────────────────────────────────────────────────────────────

    def _build_quick_view(self):
        tab = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.tab_frames["quick"] = tab

        # Header
        header = ctk.CTkFrame(tab, fg_color="transparent")
        header.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            header, text="Quick Text Translation",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"]
        ).pack(side="left")

        # Top Control Bar
        toolbar = ctk.CTkFrame(
            tab, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        toolbar.pack(fill="x", pady=(0, 12))

        tb_inner = ctk.CTkFrame(toolbar, fg_color="transparent")
        tb_inner.pack(fill="x", padx=12, pady=8)

        # Direction
        self.quick_dir_var = ctk.StringVar(value="ja2en")
        dir_display_values = [p[0] for p in LANGUAGE_PAIRS]
        self.quick_dir_combo = ctk.CTkComboBox(
            tb_inner, values=dir_display_values,
            variable=ctk.StringVar(value=dir_display_values[0]),
            width=180, state="readonly", font=ctk.CTkFont(size=12),
            command=self._on_quick_dir_changed
        )
        self.quick_dir_combo.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            tb_inner, text="⇄", width=32, height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._swap_quick_dir
        ).pack(side="left", padx=(0, 14))

        # Model
        ctk.CTkLabel(
            tb_inner, text="Model:", font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME["text_secondary"]
        ).pack(side="left", padx=(0, 4))

        self.quick_model_combo = ctk.CTkComboBox(
            tb_inner, values=GEMMA_PRESETS,
            variable=self.model_var, width=180, font=ctk.CTkFont(size=12)
        )
        self.quick_model_combo.pack(side="left")

        # Copy & Clear
        ctk.CTkButton(
            tb_inner, text="📋 Copy", width=70, height=28,
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._copy_quick_translation
        ).pack(side="right", padx=(4, 0))

        ctk.CTkButton(
            tb_inner, text="🗑 Clear", width=70, height=28,
            fg_color="transparent", text_color=THEME["text_secondary"], hover_color=THEME["btn_secondary"],
            command=self._clear_quick_text
        ).pack(side="right", padx=4)

        # Split Text Pane
        pane = ctk.CTkFrame(tab, fg_color="transparent")
        pane.pack(fill="both", expand=True)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(0, weight=1)

        # Left Source Box
        src_card = ctk.CTkFrame(
            pane, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        src_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        src_card.rowconfigure(0, weight=0)
        src_card.rowconfigure(1, weight=1)
        src_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            src_card, text="Source Text", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_secondary"]
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.quick_source = ctk.CTkTextbox(
            src_card, font=ctk.CTkFont(size=13), wrap="word",
            fg_color="transparent", border_width=0
        )
        self.quick_source.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.quick_source.bind("<Control-Return>", lambda e: self._start_quick_translate())

        # Right Target Box
        tgt_card = ctk.CTkFrame(
            pane, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        tgt_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        tgt_card.rowconfigure(0, weight=0)
        tgt_card.rowconfigure(1, weight=1)
        tgt_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tgt_card, text="Translation Output (Ollama)", font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_secondary"]
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.quick_target = ctk.CTkTextbox(
            tgt_card, font=ctk.CTkFont(size=13), wrap="word",
            fg_color="transparent", border_width=0, state="disabled"
        )
        self.quick_target.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Bottom Bar
        bottom = ctk.CTkFrame(tab, fg_color="transparent")
        bottom.pack(fill="x", pady=(12, 0))

        self.quick_translate_btn = ctk.CTkButton(
            bottom, text="▶   Translate Text", width=160, height=38, corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["primary"], hover_color=THEME["primary_hover"],
            command=self._start_quick_translate
        )
        self.quick_translate_btn.pack(side="left", padx=(0, 10))

        self.quick_status = ctk.CTkLabel(
            bottom, text="Press Ctrl+Enter or click 'Translate Text'",
            font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"]
        )
        self.quick_status.pack(side="left")

    # ─────────────────────────────────────────────────────────────
    #  TAB 3: SYSTEM & AI WORKSPACE
    # ─────────────────────────────────────────────────────────────

    def _build_system_view(self):
        tab = ctk.CTkFrame(self.content_area, fg_color="transparent")
        self.tab_frames["system"] = tab

        scroll = ctk.CTkScrollableFrame(tab, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            header, text="System & AI Diagnostics",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"]
        ).pack(anchor="w")

        ctk.CTkLabel(
            header, text="Manage offline neural models, Ollama connectivity, and hardware constraints.",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"]
        ).pack(anchor="w", pady=(2, 0))

        # Card: Argos Engine
        card_argos = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_argos.pack(fill="x", pady=(0, 14))

        ca_inner = ctk.CTkFrame(card_argos, fg_color="transparent")
        ca_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            ca_inner, text="📦  Argos Offline Neural Engine",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).pack(anchor="w")

        self.sys_argos_desc = ctk.CTkLabel(
            ca_inner, text="Verifying installed language packages on disk...",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"], justify="left"
        )
        self.sys_argos_desc.pack(anchor="w", pady=(6, 12))

        argos_action_row = ctk.CTkFrame(ca_inner, fg_color="transparent")
        argos_action_row.pack(fill="x")

        self.sys_argos_btn = ctk.CTkButton(
            argos_action_row, text="⬇   Download / Reinstall Language Packages (JA ↔ EN)",
            height=34, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"], hover_color=THEME["primary_hover"],
            command=self._install_argos_packages_gui
        )
        self.sys_argos_btn.pack(side="left", padx=(0, 10))

        self.sys_argos_msg = ctk.CTkLabel(argos_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_argos_msg.pack(side="left")

        # Card: Ollama Engine
        card_ollama = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_ollama.pack(fill="x", pady=(0, 14))

        co_inner = ctk.CTkFrame(card_ollama, fg_color="transparent")
        co_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            co_inner, text="🧠  Local Ollama LLM Service",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).pack(anchor="w")

        self.sys_ollama_desc = ctk.CTkLabel(
            co_inner, text="Probing Ollama at http://localhost:11434...",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"], justify="left"
        )
        self.sys_ollama_desc.pack(anchor="w", pady=(6, 12))

        ctk.CTkButton(
            co_inner, text="↻  Refresh Ollama Connection", width=180, height=32,
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._refresh_ollama_status
        ).pack(anchor="w")

        # Card: Hardware
        card_hw = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_hw.pack(fill="x", pady=(0, 14))

        ch_inner = ctk.CTkFrame(card_hw, fg_color="transparent")
        ch_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            ch_inner, text="💻  System Hardware & Memory Diagnostics",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).pack(anchor="w")

        self.sys_hw_desc = ctk.CTkLabel(
            ch_inner, text="Checking system RAM and disk...",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"], justify="left"
        )
        self.sys_hw_desc.pack(anchor="w", pady=(6, 0))

        # Card: Translation Cache
        card_cache = ctk.CTkFrame(
            scroll, fg_color=THEME["card_bg"],
            border_color=THEME["card_border"], border_width=1, corner_radius=12
        )
        card_cache.pack(fill="x", pady=(0, 14))

        cc_inner = ctk.CTkFrame(card_cache, fg_color="transparent")
        cc_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            cc_inner, text="💾  Translation Cache",
            font=ctk.CTkFont(size=15, weight="bold"), text_color=THEME["text_primary"]
        ).pack(anchor="w")

        self.sys_cache_desc = ctk.CTkLabel(
            cc_inner, text="Persistent cache stores previously translated text chunks to accelerate future runs.",
            font=ctk.CTkFont(size=12), text_color=THEME["text_secondary"], justify="left"
        )
        self.sys_cache_desc.pack(anchor="w", pady=(6, 12))

        cache_action_row = ctk.CTkFrame(cc_inner, fg_color="transparent")
        cache_action_row.pack(fill="x")

        self.sys_clear_cache_btn = ctk.CTkButton(
            cache_action_row, text="🗑   Clear Translation Cache",
            height=32, font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["btn_secondary"], hover_color=THEME["btn_sec_hover"],
            command=self._clear_cache_gui
        )
        self.sys_clear_cache_btn.pack(side="left", padx=(0, 10))

        self.sys_cache_msg = ctk.CTkLabel(cache_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_cache_msg.pack(side="left")

    def _clear_cache_gui(self):
        confirm = messagebox.askyesno(
            "Clear Translation Cache",
            "Are you sure you want to clear the local translation cache? All cached sentence pairs will be removed."
        )
        if not confirm:
            return

        engine = TranslationEngine()
        engine.clear_cache()
        for cp in [
            os.path.abspath("translation_cache.json"),
            os.path.abspath(".translation_cache.json"),
        ]:
            if os.path.exists(cp):
                try:
                    os.remove(cp)
                except OSError:
                    pass

        self.sys_cache_msg.configure(text="✓ Cache cleared successfully", text_color=THEME["success"])
        self.root.after(3000, lambda: self.sys_cache_msg.configure(text=""))


    # ══════════════════════════════════════════════════════════════
    #  MULTI-DOCUMENT BATCH ACTIONS
    # ══════════════════════════════════════════════════════════════

    def _browse_multi_files(self):
        file_types = [
            ("Supported Documents", "*.pptx;*.xlsx;*.docx;*.pdf"),
            ("PowerPoint", "*.pptx"),
            ("Excel", "*.xlsx"),
            ("Word", "*.docx"),
            ("PDF", "*.pdf"),
            ("All Files", "*.*"),
        ]
        chosen = filedialog.askopenfilenames(title="Select Document(s)", filetypes=file_types)
        if chosen:
            for f in chosen:
                norm = os.path.abspath(f)
                if norm not in self.selected_files:
                    self.selected_files.append(norm)
            self._render_staged_files()

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select Folder of Documents")
        if folder:
            valid_exts = SUPPORTED_EXTENSIONS
            added = 0
            for root_dir, _, files in os.walk(folder):
                for file in files:
                    ext = os.path.splitext(file)[1].lower()
                    if ext in valid_exts and not file.startswith("~$"):
                        norm = os.path.abspath(os.path.join(root_dir, file))
                        if norm not in self.selected_files:
                            self.selected_files.append(norm)
                            added += 1
            if added > 0:
                self._log(f"[Folder] Imported {added} documents from {folder}")
            else:
                messagebox.showinfo("No Supported Files", "No PPTX, XLSX, DOCX, or PDF files found in folder.")
            self._render_staged_files()

    def _render_staged_files(self):
        for widget in self.staged_list_frame.winfo_children():
            widget.destroy()

        count = len(self.selected_files)
        # Update header and button
        self.staged_header.winfo_children()[0].configure(
            text=f"{count} document{'s' if count != 1 else ''} staged for translation"
        )
        self.start_btn.configure(
            text=f"▶   Start Translation ({count})" if count > 0 else "▶   Start Translation"
        )

        if count == 0:
            self.empty_staged_label = ctk.CTkLabel(
                self.staged_list_frame, text="No documents selected yet.",
                font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"], pady=12
            )
            self.empty_staged_label.pack()
            return

        for fpath in self.selected_files:
            chip = ctk.CTkFrame(self.staged_list_frame, fg_color=THEME["card_bg"], corner_radius=6)
            chip.pack(fill="x", padx=8, pady=3)

            ext = os.path.splitext(fpath)[1].lower().replace(".", "").upper()
            bcolor = THEME.get(f"badge_{ext.lower()}", THEME["primary"])

            # Format badge
            ctk.CTkLabel(
                chip, text=f" {ext} ", font=ctk.CTkFont(size=9, weight="bold"),
                fg_color=bcolor, corner_radius=4, text_color="#FFFFFF"
            ).pack(side="left", padx=(8, 6), pady=6)

            # Name & size
            name = os.path.basename(fpath)
            try:
                sz = os.path.getsize(fpath) / 1024
                sz_str = f"{sz / 1024:.1f} MB" if sz > 1024 else f"{sz:.0f} KB"
            except Exception:
                sz_str = ""

            ctk.CTkLabel(
                chip, text=f"{name}  ({sz_str})", font=ctk.CTkFont(size=11),
                text_color=THEME["text_primary"], anchor="w"
            ).pack(side="left", fill="x", expand=True, padx=4)

            # Remove chip button
            ctk.CTkButton(
                chip, text="✕", width=22, height=20, font=ctk.CTkFont(size=10),
                fg_color="transparent", text_color=THEME["text_secondary"],
                hover_color=THEME["btn_secondary"],
                command=lambda p=fpath: self._remove_staged_file(p)
            ).pack(side="right", padx=6)

    def _remove_staged_file(self, fpath: str):
        if fpath in self.selected_files:
            self.selected_files.remove(fpath)
            self._render_staged_files()

    def _clear_selected_files(self):
        self.selected_files.clear()
        self._render_staged_files()

    # ══════════════════════════════════════════════════════════════
    #  TRANSLATION FLOW
    # ══════════════════════════════════════════════════════════════

    def _start_translation_flow(self):
        if not self.selected_files:
            messagebox.showwarning("No Documents", "Please select one or more documents to translate.")
            return

        direction = self.direction_var.get()
        mode = TranslationMode(self.mode_var.get())
        model = self.model_var.get().strip() or "gemma4:e2b-it-qat"
        raw_glossary = self.glossary_text.get("0.0", "end")
        glossary = parse_glossary_text(raw_glossary)

        if mode == TranslationMode.PURE_LLM and not check_ollama_status():
            if not messagebox.askyesno(
                "Ollama Offline",
                "Pure LLM mode requires Ollama, but Ollama is offline.\n\nContinue anyway?"
            ):
                return

        added = 0
        for input_path in list(self.selected_files):
            if not os.path.exists(input_path):
                continue
            base, ext = os.path.splitext(input_path)
            output_path = f"{base}_{direction}{ext}"

            self.translation_queue.add_job(
                input_path=input_path,
                output_path=output_path,
                direction=direction,
                mode=mode,
                model_name=model,
                glossary=glossary
            )
            self._last_output_path = output_path
            self._last_review_log = f"{output_path}.needs_review.log"
            added += 1

        self._log(f"[Queue] Dispatched {added} document(s) via {mode.value.upper()} ({direction}).")
        self._clear_selected_files()

    def _add_to_queue_only(self):
        if not self.selected_files:
            messagebox.showwarning("No Documents", "Please select one or more documents first.")
            return
        self._start_translation_flow()

    # ══════════════════════════════════════════════════════════════
    #  QUEUE EVENT HANDLING
    # ══════════════════════════════════════════════════════════════

    def _on_job_update(self, job: TranslationJob):
        self.root.after(0, self._update_job_widget, job)

    def _on_job_log(self, job_id: str, message: str):
        self.root.after(0, self._log, message)

    def _update_job_widget(self, job: TranslationJob):
        if job.id not in self._job_widgets:
            self.queue_empty_label.pack_forget()

            row = ctk.CTkFrame(self.queue_container, fg_color=THEME["staging_bg"], corner_radius=8)
            row.pack(fill="x", pady=3)

            ext = os.path.splitext(job.input_path)[1].lower().replace(".", "").upper()
            bcolor = THEME.get(f"badge_{ext.lower()}", THEME["primary"])

            # Badge
            ctk.CTkLabel(
                row, text=f" {ext} ", font=ctk.CTkFont(size=9, weight="bold"),
                fg_color=bcolor, corner_radius=4, text_color="#FFFFFF"
            ).pack(side="left", padx=(8, 6), pady=6)

            # Name
            name = os.path.basename(job.input_path)
            name_label = ctk.CTkLabel(
                row, text=name, font=ctk.CTkFont(size=11, weight="bold"),
                width=170, anchor="w", text_color=THEME["text_primary"]
            )
            name_label.pack(side="left", padx=4)

            # Progress bar
            pbar = ctk.CTkProgressBar(row, height=8, corner_radius=4)
            pbar.set(0)
            pbar.pack(side="left", fill="x", expand=True, padx=8)

            # Status label
            st_label = ctk.CTkLabel(
                row, text="Queued", font=ctk.CTkFont(size=10),
                width=130, anchor="w", text_color=THEME["text_secondary"]
            )
            st_label.pack(side="left", padx=4)

            # Cancel button
            cancel_btn = ctk.CTkButton(
                row, text="✕", width=24, height=22, font=ctk.CTkFont(size=10),
                fg_color=THEME["error"], hover_color="#991B1B",
                command=lambda jid=job.id: self.translation_queue.cancel_job(jid)
            )
            cancel_btn.pack(side="right", padx=8)

            self._job_widgets[job.id] = {
                "row": row,
                "progress": pbar,
                "status_label": st_label,
                "cancel_btn": cancel_btn,
            }

        w = self._job_widgets[job.id]
        pct = job.progress / 100.0
        w["progress"].set(pct)

        if job.status == JobStatus.QUEUED:
            w["status_label"].configure(text="⏳ Queued", text_color=THEME["text_secondary"])
        elif job.status == JobStatus.RUNNING:
            msg = job.progress_message
            if len(msg) > 20:
                msg = msg[:18] + "..."
            w["status_label"].configure(text=f"🔄 {job.progress:.0f}% {msg}", text_color=THEME["primary"])
        elif job.status == JobStatus.COMPLETED:
            elapsed = ""
            if job.started_at and job.completed_at:
                secs = int(job.completed_at - job.started_at)
                m, s = divmod(secs, 60)
                elapsed = f" ({m:02d}:{s:02d})"
            w["status_label"].configure(text=f"✅ Done{elapsed}", text_color=THEME["success"])
            w["cancel_btn"].configure(state="disabled")
            w["progress"].set(1.0)
            self.open_file_btn.configure(state="normal")
            self.open_dir_btn.configure(state="normal")
            if self._last_review_log and os.path.exists(self._last_review_log) and os.path.getsize(self._last_review_log) > 0:
                self.review_btn.configure(state="normal")
        elif job.status == JobStatus.FAILED:
            err = job.error or "Failed"
            if len(err) > 22:
                err = err[:20] + "..."
            w["status_label"].configure(text=f"❌ {err}", text_color=THEME["error"])
            w["cancel_btn"].configure(state="disabled")
        elif job.status == JobStatus.CANCELLED:
            w["status_label"].configure(text="⛔ Cancelled", text_color=THEME["text_secondary"])
            w["cancel_btn"].configure(state="disabled")

    def _clear_completed_jobs(self):
        self.translation_queue.clear_completed()
        to_remove = []
        for jid, w in self._job_widgets.items():
            if self.translation_queue.get_job(jid) is None:
                w["row"].destroy()
                to_remove.append(jid)
        for jid in to_remove:
            del self._job_widgets[jid]
        if not self._job_widgets:
            self.queue_empty_label.pack(pady=14)

    # ══════════════════════════════════════════════════════════════
    #  QUICK TRANSLATE (OLLAMA)
    # ══════════════════════════════════════════════════════════════

    def _start_quick_translate(self):
        source_text = self.quick_source.get("0.0", "end").strip()
        if not source_text:
            self.quick_status.configure(text="Please enter or paste text to translate.", text_color=THEME["warning"])
            return

        direction = self.quick_dir_var.get()
        model_name = self.model_var.get().strip() or "gemma4:e2b-it-qat"

        self.quick_translate_btn.configure(state="disabled", text="Translating...")
        self.quick_status.configure(text=f"Connecting to Ollama ({model_name})...", text_color=THEME["primary"])

        thread = threading.Thread(
            target=self._quick_translate_worker,
            args=(source_text, direction, model_name),
            daemon=True
        )
        thread.start()

    def _quick_translate_worker(self, text: str, direction: str, model: str):
        try:
            if not check_ollama_status():
                self.root.after(0, self._set_quick_result, "", "⚠ Ollama is offline. Please start Ollama service.")
                return

            lang_names = {
                "ja": "Japanese",
                "en": "English",
                "zh": "Chinese",
                "ko": "Korean",
                "es": "Spanish",
                "fr": "French",
                "de": "German",
            }
            parts = direction.split("2") if "2" in direction else ("ja", "en")
            src_lang = lang_names.get(parts[0], parts[0])
            tgt_lang = lang_names.get(parts[1], parts[1])

            prompt = (
                f"You are an expert professional translator. Translate the following text from {src_lang} to {tgt_lang}.\n"
                f"Requirements:\n"
                f"- Output ONLY the direct translation.\n"
                f"- Preserve formatting, line breaks, numbers, and proper nouns.\n"
                f"- Do NOT include notes, explanations, commentary, or markdown quotes.\n\n"
                f"Source Text:\n{text}"
            )

            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "keep_alive": "10m",
                "options": {"temperature": 0.1, "num_ctx": 4096}
            }

            t0 = time.time()
            res = requests.post("http://localhost:11434/api/generate", json=payload, timeout=120)
            elapsed = time.time() - t0

            if res.status_code == 200:
                raw = res.json().get("response", "").strip()
                cleaned = clean_llm_response(raw)
                self.root.after(0, self._set_quick_result, cleaned, f"✓ Translated in {elapsed:.1f}s via {model}")
            elif res.status_code == 404:
                self.root.after(0, self._set_quick_result, "", f"⚠ Model '{model}' not found. Run 'ollama pull {model}'.")
            else:
                self.root.after(0, self._set_quick_result, "", f"⚠ Ollama returned HTTP {res.status_code}")

        except requests.exceptions.ConnectionError:
            self.root.after(0, self._set_quick_result, "", "⚠ Could not connect to Ollama at http://localhost:11434.")
        except requests.exceptions.Timeout:
            self.root.after(0, self._set_quick_result, "", "⚠ Request timed out (120s).")
        except Exception as e:
            self.root.after(0, self._set_quick_result, "", f"⚠ Translation error: {str(e)[:50]}")
        finally:
            def reenable():
                self.quick_translate_btn.configure(state="normal", text="▶   Translate Text")
            self.root.after(0, reenable)

    def _set_quick_result(self, text: str, status: str):
        self.quick_target.configure(state="normal")
        self.quick_target.delete("0.0", "end")
        if text:
            self.quick_target.insert("0.0", text)
        self.quick_target.configure(state="disabled")
        self.quick_status.configure(
            text=status,
            text_color=THEME["success"] if "✓" in status else THEME["warning"]
        )

    def _copy_quick_translation(self):
        text = self.quick_target.get("0.0", "end").strip()
        if text:
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.quick_status.configure(text="📋 Copied to clipboard!", text_color=THEME["success"])

    def _clear_quick_text(self):
        self.quick_source.delete("0.0", "end")
        self.quick_target.configure(state="normal")
        self.quick_target.delete("0.0", "end")
        self.quick_target.configure(state="disabled")
        self.quick_status.configure(text="Cleared. Paste text and click 'Translate Text'.", text_color=THEME["text_secondary"])

    def _on_quick_dir_changed(self, display_value: str):
        for label, code in LANGUAGE_PAIRS:
            if label == display_value:
                self.quick_dir_var.set(code)
                break

    def _swap_quick_dir(self):
        current = self.quick_dir_var.get()
        idx = next((i for i, p in enumerate(LANGUAGE_PAIRS) if p[1] == current), 0)
        new_idx = (idx + 1) % len(LANGUAGE_PAIRS)
        self.quick_dir_var.set(LANGUAGE_PAIRS[new_idx][1])
        self.quick_dir_combo.set(LANGUAGE_PAIRS[new_idx][0])

        target_text = self.quick_target.get("0.0", "end").strip()
        if target_text:
            self.quick_source.delete("0.0", "end")
            self.quick_source.insert("0.0", target_text)
            self.quick_target.configure(state="normal")
            self.quick_target.delete("0.0", "end")
            self.quick_target.configure(state="disabled")
            self.quick_status.configure(text="Direction swapped. Click 'Translate Text'.", text_color=THEME["text_secondary"])

    # ══════════════════════════════════════════════════════════════
    #  SYSTEM & DIAGNOSTICS LOGIC
    # ══════════════════════════════════════════════════════════════

    def _check_system_status(self):
        self._refresh_ollama_status()
        self._refresh_argos_status()
        self._refresh_hw_status()

    def _refresh_ollama_status(self):
        alive = check_ollama_status()
        if alive:
            models = list_installed_models()
            self.side_ollama_btn.configure(
                text=f"● Ollama: Online ({len(models)})",
                text_color=THEME["success"]
            )
            if models:
                all_models = list(dict.fromkeys(GEMMA_PRESETS + models))
                self.model_combo.configure(values=all_models)
                if hasattr(self, "quick_model_combo"):
                    self.quick_model_combo.configure(values=all_models)
                if self.model_var.get() not in all_models:
                    self.model_var.set(all_models[0])
            self.sys_ollama_desc.configure(
                text=f"Service: Online at http://localhost:11434\nInstalled Models: {', '.join(models) if models else 'None'}\n(Zero-Load Guarantee: Model weights load strictly on-demand)",
                text_color=THEME["success"]
            )
        else:
            self.side_ollama_btn.configure(
                text="● Ollama: Offline",
                text_color=THEME["error"]
            )
            self.sys_ollama_desc.configure(
                text="Service: Offline\nOllama is not running. Fast NMT mode will operate 100% offline without it.\nStart the Ollama app to enable Pure LLM mode.",
                text_color=THEME["error"]
            )

    def _refresh_argos_status(self):
        ja_en = self.nmt_backend.is_ready("ja2en")
        en_ja = self.nmt_backend.is_ready("en2ja")

        if ja_en and en_ja:
            self.side_argos_btn.configure(
                text="● Argos: Ready (JA↔EN)",
                text_color=THEME["success"]
            )
            self.sys_argos_desc.configure(
                text="Status: All language packages installed and verified.\n- Japanese → English: Ready\n- English → Japanese: Ready\nOffline neural translation is ready.",
                text_color=THEME["success"]
            )
        else:
            missing = []
            if not ja_en:
                missing.append("Japanese → English")
            if not en_ja:
                missing.append("English → Japanese")

            self.side_argos_btn.configure(
                text="● Argos: Missing Packages",
                text_color=THEME["warning"]
            )
            self.sys_argos_desc.configure(
                text=f"Status: Missing packages: {', '.join(missing)}\nClick the button below to download and install them directly (~100MB each).",
                text_color=THEME["warning"]
            )

    def _install_argos_packages_gui(self):
        self.sys_argos_btn.configure(state="disabled", text="Installing Packages...")
        self.sys_argos_msg.configure(text="Updating package index...", text_color=THEME["primary"])

        def worker():
            try:
                def cb(msg):
                    self.root.after(0, self.sys_argos_msg.configure, {"text": msg, "text_color": THEME["primary"]})

                ok1 = self.nmt_backend.install_language_pair("ja", "en", log_cb=cb)
                ok2 = self.nmt_backend.install_language_pair("en", "ja", log_cb=cb)

                if ok1 and ok2:
                    self.root.after(0, self._on_argos_complete, True, "✓ Packages installed successfully!")
                else:
                    self.root.after(0, self._on_argos_complete, False, "⚠ Could not install all packages. Check internet.")
            except Exception as e:
                self.root.after(0, self._on_argos_complete, False, f"⚠ Error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_argos_complete(self, success: bool, msg: str):
        self.sys_argos_btn.configure(state="normal", text="⬇   Download / Reinstall Language Packages (JA ↔ EN)")
        self.sys_argos_msg.configure(text=msg, text_color=THEME["success"] if success else THEME["error"])
        self._refresh_argos_status()

    def _refresh_hw_status(self):
        ram_ok, ram_avail = check_ram()
        disk_ok, disk_free = check_disk_space()

        txt = (
            f"Available System RAM: {ram_avail:.2f} GB ({'Healthy' if ram_ok else 'Low'})\n"
            f"Available Disk Space: {disk_free:.2f} GB ({'Healthy' if disk_ok else 'Low'})\n"
            f"Active Memory Guard: Dynamic model flush active."
        )
        self.sys_hw_desc.configure(text=txt)

    # ══════════════════════════════════════════════════════════════
    #  INTERACTIONS & TOGGLES
    # ══════════════════════════════════════════════════════════════

    def _on_mode_seg_changed(self, value: str):
        if "Pure LLM" in value:
            self.mode_var.set(TranslationMode.PURE_LLM.value)
            self.model_combo.configure(state="normal")
        else:
            self.mode_var.set(TranslationMode.FAST_NMT.value)
            self.model_combo.configure(state="disabled")

    def _on_direction_changed(self, display_value: str):
        for label, code in LANGUAGE_PAIRS:
            if label == display_value:
                self.direction_var.set(code)
                break

    def _swap_doc_direction(self):
        current = self.direction_var.get()
        idx = next((i for i, p in enumerate(LANGUAGE_PAIRS) if p[1] == current), 0)
        new_idx = (idx + 1) % len(LANGUAGE_PAIRS)
        self.direction_var.set(LANGUAGE_PAIRS[new_idx][1])
        self.direction_combo.set(LANGUAGE_PAIRS[new_idx][0])

    def _toggle_glossary(self):
        if self._glossary_open:
            self.glossary_drawer.pack_forget()
            self.glossary_btn.configure(text="▶  Custom Glossary (Optional)")
            self._glossary_open = False
        else:
            self.glossary_drawer.pack(fill="x", pady=(6, 0))
            self.glossary_btn.configure(text="▼  Custom Glossary (Optional)")
            self._glossary_open = True

    def _toggle_log(self):
        if self._log_open:
            self.log_drawer.pack_forget()
            self.log_drawer_btn.configure(text="Activity Log ▶")
            self._log_open = False
        else:
            self.log_drawer.pack(fill="x", pady=(0, 16))
            self.log_drawer_btn.configure(text="Activity Log ▼")
            self._log_open = True

    def _toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Dark Theme")
        else:
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Light Theme")

    def _log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _open_last_output(self):
        if self._last_output_path and os.path.exists(self._last_output_path):
            try:
                os.startfile(self._last_output_path)
            except Exception as e:
                messagebox.showerror("Cannot Open File", str(e))

    def _open_last_dir(self):
        if self._last_output_path:
            d = os.path.dirname(os.path.abspath(self._last_output_path))
            if os.path.exists(d):
                try:
                    subprocess.Popen(["explorer", d])
                except Exception:
                    pass

    def _view_review_log(self):
        if self._last_review_log and os.path.exists(self._last_review_log):
            try:
                os.startfile(self._last_review_log)
            except Exception as e:
                messagebox.showerror("Cannot Open File", str(e))

    def cleanup(self):
        self.translation_queue.shutdown()


def launch_gui():
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    app = TranslatorApp(root)

    def on_close():
        app.cleanup()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
