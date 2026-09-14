"""
gui/views/documents_view.py
===========================
Main workspace view for multi-document batch translation, staging, and queue monitoring.
"""

import os
import subprocess
import sys
import time
from collections.abc import Callable
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from engine.cache import CachePolicy
from engine.core import TranslationMode
from engine.preflight import check_ollama_status
from engine.queue_manager import JobStatus, TranslationJob, format_eta
from gui.controllers.translation_controller import TranslationController
from gui.dnd_helper import WindowsDropHook, is_point_in_widget
from gui.theme import (
    GEMMA_PRESETS,
    LANGUAGE_PAIRS,
    THEME,
    parse_glossary_text,
)
from gui.widgets.job_row import JobRow
from gui.widgets.staged_file_list import StagedFileList

GLOSSARY_DIR = os.path.expanduser("~/.offline-translator")
LAST_GLOSSARY_PATH = os.path.join(GLOSSARY_DIR, "last_glossary.txt")


class DocumentsView(ctk.CTkFrame):
    """
    Renders the document staging, configuration, queue, and post-action panels.
    """

    glossary_btn: Any = None
    glossary_load_btn: Any = None
    glossary_save_btn: Any = None
    glossary_drawer: Any = None
    glossary_text: Any = None
    queue_empty_label: Any = None
    batch_summary_card: Any = None
    summary_files_lbl: Any = None
    summary_chunks_lbl: Any = None
    summary_time_lbl: Any = None
    summary_review_lbl: Any = None
    summary_open_folder_btn: Any = None
    summary_copy_btn: Any = None
    bottom_box: Any = None
    _current_batch_ids: Any = None
    _batch_summary_shown: Any = None
    _last_batch_summary: Any = None

    def __init__(
        self,
        master,
        controller: TranslationController | None = None,
        on_log: Callable[[str], None] | None = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller or TranslationController()
        self.on_log_cb = on_log

        self._job_widgets: dict[str, JobRow | dict] = {}
        self._completed_outputs: list[str] = []
        self._completed_review_logs: list[str] = []
        self._last_output_path: str | None = None
        self._last_review_log: str | None = None
        self._current_batch_ids: set[str] = set()
        self._batch_summary_shown: bool = False
        self._last_batch_summary: dict[str, Any] | None = None
        self._glossary_open = False
        self._log_open = False
        self._drop_hook: WindowsDropHook | None = None

        self._build_ui()

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── Page Header ──
        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            header,
            text="Document Translation",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Select single or batch documents (PPTX, XLSX, DOCX, PDF) to translate offline.",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(2, 0))

        # ── Card 1: Staging & Document Selection ──
        card_sel = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_sel.pack(fill="x", pady=(0, 14))

        # Drop/Upload Target Area
        upload_area = ctk.CTkFrame(card_sel, fg_color=THEME["staging_bg"], corner_radius=10)
        upload_area.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(upload_area, text="📂", font=ctk.CTkFont(size=28)).pack(pady=(12, 4))

        ctk.CTkLabel(
            upload_area,
            text="Add Documents to Translate",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack()

        ctk.CTkLabel(
            upload_area,
            text="Choose multiple individual files or scan an entire folder",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"],
        ).pack(pady=(2, 10))

        # Target button row
        target_btn_row = ctk.CTkFrame(upload_area, fg_color="transparent")
        target_btn_row.pack(pady=(0, 14))

        ctk.CTkButton(
            target_btn_row,
            text="  Select Files...  ",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._browse_multi_files,
        ).pack(side="left", padx=6)

        ctk.CTkButton(
            target_btn_row,
            text="  Select Folder...  ",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._browse_folder,
        ).pack(side="left", padx=6)

        # Staged files container
        self.staged_list = StagedFileList(
            card_sel,
            on_files_changed=self._on_staged_files_changed,
        )
        self.staged_list.pack(fill="x", padx=16, pady=(0, 14))

        # ── Card 2: Configuration & Direction Bar ──
        card_opt = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
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
            engine_box,
            text="ENGINE",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        self.mode_var = ctk.StringVar(value=TranslationMode.FAST_NMT.value)
        self.mode_seg = ctk.CTkSegmentedButton(
            engine_box,
            values=["⚡ Fast NMT (Offline)", "🧠 Pure LLM (Ollama)"],
            command=self._on_mode_seg_changed,
            font=ctk.CTkFont(size=12, weight="bold"),
        )
        self.mode_seg.set("⚡ Fast NMT (Offline)")
        self.mode_seg.pack()

        # Direction Box
        dir_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        dir_box.pack(side="left", fill="y", padx=(0, 16))

        ctk.CTkLabel(
            dir_box,
            text="DIRECTION",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        dir_sub = ctk.CTkFrame(dir_box, fg_color="transparent")
        dir_sub.pack()

        self.direction_var = ctk.StringVar(value="ja2en")
        dir_display_values = [p[0] for p in LANGUAGE_PAIRS]
        self.direction_combo = ctk.CTkComboBox(
            dir_sub,
            values=dir_display_values,
            variable=ctk.StringVar(value=dir_display_values[0]),
            width=180,
            state="readonly",
            font=ctk.CTkFont(size=12),
            command=self._on_direction_changed,
        )
        self.direction_combo.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            dir_sub,
            text="⇄",
            width=32,
            height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._swap_doc_direction,
        ).pack(side="left")

        # Model Box (Dims when Fast NMT)
        self.model_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        self.model_box.pack(side="left", fill="y")

        ctk.CTkLabel(
            self.model_box,
            text="OLLAMA MODEL",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        self.model_var = ctk.StringVar(value=GEMMA_PRESETS[0])
        self.model_combo = ctk.CTkComboBox(
            self.model_box,
            values=GEMMA_PRESETS,
            variable=self.model_var,
            width=190,
            state="disabled",
            font=ctk.CTkFont(size=12),
        )
        self.model_combo.pack()

        # Cache Policy Box
        self.cache_box = ctk.CTkFrame(ctrl_row, fg_color="transparent")
        self.cache_box.pack(side="left", fill="y", padx=(14, 0))

        ctk.CTkLabel(
            self.cache_box,
            text="CACHE POLICY",
            font=ctk.CTkFont(size=10, weight="bold"),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(0, 4))

        self.cache_policy_display = [
            "Encrypted (Default)",
            "In-Memory (Privacy)",
            "Plaintext (Compatibility)",
        ]
        self.cache_policy_var = ctk.StringVar(value=self.cache_policy_display[0])
        self.cache_policy_combo = ctk.CTkComboBox(
            self.cache_box,
            values=self.cache_policy_display,
            variable=self.cache_policy_var,
            width=180,
            state="readonly",
            font=ctk.CTkFont(size=12),
        )
        self.cache_policy_combo.pack()

        # Collapsible Glossary in Card 2
        glossary_toggle_frame = ctk.CTkFrame(opt_inner, fg_color="transparent")
        glossary_toggle_frame.pack(fill="x", pady=(12, 0))

        self.glossary_btn = ctk.CTkButton(
            glossary_toggle_frame,
            text="▶  Custom Glossary (Optional)",
            anchor="w",
            fg_color="transparent",
            hover_color=THEME["btn_secondary"],
            text_color=THEME["text_secondary"],
            font=ctk.CTkFont(size=12),
            command=self._toggle_glossary,
        )
        self.glossary_btn.pack(side="left")

        self.glossary_save_btn = ctk.CTkButton(
            glossary_toggle_frame,
            text="💾 Save",
            width=68,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=THEME["text_secondary"],
            hover_color=THEME["btn_secondary"],
            command=self._save_glossary_file,
        )
        self.glossary_save_btn.pack(side="right", padx=(4, 0))

        self.glossary_load_btn = ctk.CTkButton(
            glossary_toggle_frame,
            text="📥 Load",
            width=68,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=THEME["text_secondary"],
            hover_color=THEME["btn_secondary"],
            command=self._load_glossary_file,
        )
        self.glossary_load_btn.pack(side="right")

        self.glossary_drawer = ctk.CTkFrame(opt_inner, fg_color="transparent")
        self.glossary_text = ctk.CTkTextbox(
            self.glossary_drawer,
            height=70,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=THEME["staging_bg"],
        )
        self.glossary_text.pack(fill="x", pady=(6, 0))
        self.glossary_text.insert("0.0", "# Term -> Translation (one per line)\n")

        # ── Primary Action Row ──
        action_bar = ctk.CTkFrame(scroll, fg_color="transparent")
        action_bar.pack(fill="x", pady=(2, 14))

        self.start_btn = ctk.CTkButton(
            action_bar,
            text="▶   Start Translation",
            width=220,
            height=44,
            corner_radius=8,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self.start_translation_flow,
        )
        self.start_btn.pack(side="left", padx=(0, 10))

        self.add_queue_btn = ctk.CTkButton(
            action_bar,
            text="➕  Add to Queue Only",
            width=160,
            height=44,
            corner_radius=8,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self.add_to_queue_only,
        )
        self.add_queue_btn.pack(side="left")

        # ── Card 3: Queue & Active Jobs ──
        card_q = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_q.pack(fill="x", pady=(0, 14))

        q_head = ctk.CTkFrame(card_q, fg_color="transparent")
        q_head.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            q_head,
            text="Translation Queue & Progress",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(side="left")

        ctk.CTkButton(
            q_head,
            text="Clear Finished",
            width=90,
            height=24,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self.clear_completed_jobs,
        ).pack(side="right")

        self.queue_container = ctk.CTkFrame(card_q, fg_color="transparent")
        self.queue_container.pack(fill="x", padx=16, pady=(0, 12))

        self.queue_empty_label = ctk.CTkLabel(
            self.queue_container,
            text="No documents in queue. Select files above and click 'Start Translation'.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color=THEME["text_secondary"],
            pady=14,
        )
        self.queue_empty_label.pack()

        # ── Batch Summary Card (Initially Hidden) ──
        self._build_batch_summary_card(scroll)

        # ── Card 4: Post-Actions & Activity Drawer ──
        self.bottom_box = ctk.CTkFrame(scroll, fg_color="transparent")
        self.bottom_box.pack(fill="x", pady=(0, 16))
        bottom_box = self.bottom_box

        self.open_file_btn = ctk.CTkButton(
            bottom_box,
            text="📄  Open Output File",
            width=140,
            height=32,
            state="disabled",
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self.open_last_output,
        )
        self.open_file_btn.pack(side="left", padx=(0, 8))

        self.open_dir_btn = ctk.CTkButton(
            bottom_box,
            text="📁  Open Folder",
            width=120,
            height=32,
            state="disabled",
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self.open_last_dir,
        )
        self.open_dir_btn.pack(side="left", padx=(0, 8))

        self.review_btn = ctk.CTkButton(
            bottom_box,
            text="⚠  Review Log",
            width=120,
            height=32,
            state="disabled",
            fg_color=THEME["warning"],
            hover_color="#B45309",
            command=self.view_review_log,
        )
        self.review_btn.pack(side="left")

        self.log_drawer_btn = ctk.CTkButton(
            bottom_box,
            text="Activity Log ▶",
            width=110,
            height=32,
            fg_color="transparent",
            text_color=THEME["text_secondary"],
            hover_color=THEME["btn_secondary"],
            command=self._toggle_log,
        )
        self.log_drawer_btn.pack(side="right")

        # Collapsible log
        self.log_drawer = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=10,
        )
        self.log_text = ctk.CTkTextbox(
            self.log_drawer,
            height=130,
            font=ctk.CTkFont(family="Consolas", size=10),
            fg_color=THEME["log_bg"],
            text_color=THEME["log_fg"],
            state="disabled",
        )
        self.log_text.pack(fill="x", padx=12, pady=12)

        # Auto-load persisted glossary & initialize DND
        self._auto_load_glossary()
        self._setup_dnd()

    # ── Properties & Staged List Integration ──

    @property
    def selected_files(self) -> list[str]:
        return self.staged_list.selected_files

    def _browse_multi_files(self):
        self.staged_list.browse_files()

    def _browse_folder(self):
        self.staged_list.browse_folder(log_cb=self.log)

    def _on_staged_files_changed(self, files: list[str]):
        count = len(files)
        if count > 0:
            self.start_btn.configure(text=f"▶   Start Translation ({count})")
        else:
            self.start_btn.configure(text="▶   Start Translation")

    # ── Mode & Direction Handlers ──

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

    # ── Glossary Persistence & Management ──

    def _auto_load_glossary(self):
        """Auto-loads last-used glossary from ~/.offline-translator/last_glossary.txt on startup."""
        try:
            if os.path.isfile(LAST_GLOSSARY_PATH):
                with open(LAST_GLOSSARY_PATH, encoding="utf-8") as f:
                    content = f.read()
                if content.strip():
                    self.glossary_text.delete("0.0", "end")
                    self.glossary_text.insert("0.0", content)
        except Exception:
            pass

    def sync_glossary(self):
        """Synchronizes glossary content from persisted file if modified elsewhere."""
        try:
            if os.path.isfile(LAST_GLOSSARY_PATH):
                with open(LAST_GLOSSARY_PATH, encoding="utf-8") as f:
                    content = f.read()
                current = self.glossary_text.get("0.0", "end")
                if content.strip() and content.strip() != current.strip():
                    self.glossary_text.delete("0.0", "end")
                    self.glossary_text.insert("0.0", content)
        except Exception:
            pass

    def persist_glossary(self):
        """Auto-persists current glossary to ~/.offline-translator/last_glossary.txt."""
        try:
            os.makedirs(GLOSSARY_DIR, exist_ok=True)
            content = self.glossary_text.get("0.0", "end").rstrip()
            with open(LAST_GLOSSARY_PATH, "w", encoding="utf-8") as f:
                f.write(content + "\n" if content else "")
        except Exception:
            pass

    def _load_glossary_file(self, file_path: str | None = None):
        """Loads a glossary text/csv file into the glossary drawer."""
        if not file_path:
            file_path = filedialog.askopenfilename(
                title="Load Glossary File",
                filetypes=[
                    ("Glossary / Text Files", "*.txt *.csv"),
                    ("Text Files", "*.txt"),
                    ("CSV Files", "*.csv"),
                    ("All Files", "*.*"),
                ],
            )
        if not file_path or not os.path.isfile(file_path):
            return

        try:
            try:
                with open(file_path, encoding="utf-8") as f:
                    content = f.read()
            except UnicodeDecodeError:
                with open(file_path, encoding="latin-1") as f:
                    content = f.read()

            warnings: list[str] = []
            parsed = parse_glossary_text(content, on_warning=lambda w: warnings.append(w))

            self.glossary_text.delete("0.0", "end")
            self.glossary_text.insert("0.0", content)

            if not self._glossary_open:
                self._toggle_glossary()

            msg = f"[Glossary] Loaded {len(parsed)} entries from {os.path.basename(file_path)}."
            if warnings:
                msg += f" (Warnings: {'; '.join(warnings)})"
                messagebox.showwarning("Glossary Limits", "\n".join(warnings))
            self.log(msg)
            self.persist_glossary()
        except Exception as e:
            messagebox.showerror("Glossary Load Error", f"Failed to load glossary file:\n{e}")

    def _save_glossary_file(self):
        """Saves current glossary drawer content to a user-chosen text file."""
        content = self.glossary_text.get("0.0", "end").strip()
        if not content or content == "# Term -> Translation (one per line)":
            messagebox.showinfo("Empty Glossary", "There are no glossary entries to save.")
            return

        dest = filedialog.asksaveasfilename(
            title="Save Glossary As",
            defaultextension=".txt",
            filetypes=[("Text Files", "*.txt"), ("All Files", "*.*")],
            initialfile="glossary.txt",
        )
        if not dest:
            return

        try:
            with open(dest, "w", encoding="utf-8") as f:
                f.write(content + "\n")
            self.log(f"[Glossary] Saved glossary to {os.path.basename(dest)}.")
        except Exception as e:
            messagebox.showerror("Glossary Save Error", f"Failed to save glossary:\n{e}")

    def _setup_dnd(self):
        """Initializes drag-and-drop support on Windows and via Tkinter extensions."""
        for w in (self.glossary_drawer, self.glossary_text):
            if hasattr(w, "drop_target_register"):
                try:
                    w.drop_target_register("DND_Files")
                    w.dnd_bind("<<Drop>>", self._on_tkdnd_drop)
                except Exception:
                    pass

        if sys.platform == "win32":
            try:
                self._drop_hook = WindowsDropHook(self, self._on_window_drop)
                self.after(200, self._drop_hook.hook)
            except Exception:
                pass

        self.bind("<Destroy>", self._on_destroy_cleanup, add="+")

    def _on_destroy_cleanup(self, event=None):
        if event and event.widget != self:
            return
        self.persist_glossary()
        if self._drop_hook:
            self._drop_hook.unhook()
            self._drop_hook = None

    def _on_window_drop(self, files: list[str], screen_x: int, screen_y: int):
        """Handles dropped files from Windows Explorer."""
        if not files:
            return
        if not self.winfo_ismapped():
            try:
                for child in self.master.winfo_children():
                    if hasattr(child, "_on_window_drop") and child != self and child.winfo_ismapped():
                        child._on_window_drop(files, screen_x, screen_y)
                        break
            except Exception:
                pass
            return

        hit_glossary = is_point_in_widget(self.glossary_drawer, screen_x, screen_y) or is_point_in_widget(
            self.glossary_text, screen_x, screen_y
        )
        txt_files = [f for f in files if f.lower().endswith((".txt", ".csv"))]
        doc_files = [f for f in files if os.path.splitext(f.lower())[1] in [".docx", ".pptx", ".xlsx", ".pdf"]]

        if hit_glossary or (txt_files and not doc_files):
            target = txt_files[0] if txt_files else files[0]
            self._load_glossary_file(target)
        elif doc_files:
            self.staged_list.add_files(doc_files)
        elif os.path.isdir(files[0]):
            self._browse_folder_path(files[0])

    def _browse_folder_path(self, folder_path: str):
        from formats.registry import SUPPORTED_EXTENSIONS

        added = []
        for root_dir, _, filenames in os.walk(folder_path):
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext in SUPPORTED_EXTENSIONS:
                    added.append(os.path.join(root_dir, fn))
        if added:
            self.staged_list.add_files(added)
            self.log(f"Staged {len(added)} document(s) from dropped folder.")

    def _on_tkdnd_drop(self, event):
        try:
            files = self.tk.splitlist(event.data)
            if files:
                self._load_glossary_file(files[0])
        except Exception:
            pass

    def _toggle_log(self):
        if self._log_open:
            self.log_drawer.pack_forget()
            self.log_drawer_btn.configure(text="Activity Log ▶")
            self._log_open = False
        else:
            self.log_drawer.pack(fill="x", pady=(0, 16))
            self.log_drawer_btn.configure(text="Activity Log ▼")
            self._log_open = True

    def update_model_choices(self, models: list[str]):
        if models:
            all_models = list(dict.fromkeys(GEMMA_PRESETS + models))
            self.model_combo.configure(values=all_models)
            if self.model_var.get() not in all_models:
                self.model_var.set(all_models[0])

    # ── Job Dispatch & Execution ──

    def start_translation_flow(self):
        staged = self.staged_list.get_files()
        if not staged:
            messagebox.showwarning("No Documents", "Please select one or more documents to translate.")
            return

        direction = self.direction_var.get()
        mode = TranslationMode(self.mode_var.get())
        model = self.model_var.get().strip() or GEMMA_PRESETS[0]
        raw_glossary = self.glossary_text.get("0.0", "end")
        glossary_warnings: list[str] = []
        glossary = parse_glossary_text(raw_glossary, on_warning=lambda w: glossary_warnings.append(w))
        if glossary_warnings:
            self.log(f"[Glossary Warning] {'; '.join(glossary_warnings)}")
        self.persist_glossary()

        cache_sel = self.cache_policy_var.get()
        if "In-Memory" in cache_sel:
            cache_policy = CachePolicy.MEMORY_ONLY
        elif "Plaintext" in cache_sel:
            cache_policy = CachePolicy.PLAINTEXT_PERSISTENT
        else:
            cache_policy = CachePolicy.ENCRYPTED_PERSISTENT

        if mode == TranslationMode.PURE_LLM and not check_ollama_status():
            if not messagebox.askyesno(
                "Ollama Offline",
                "Pure LLM mode requires Ollama, but Ollama is offline.\n\nContinue anyway?",
            ):
                return

        dispatched = self.controller.start_batch(
            input_files=staged,
            direction=direction,
            mode=mode,
            model_name=model,
            glossary=glossary,
            cache_policy=cache_policy,
        )

        self._current_batch_ids = {job_id for job_id, _, _ in dispatched}
        self._batch_summary_shown = False
        if hasattr(self, "batch_summary_card") and self.batch_summary_card:
            self.batch_summary_card.pack_forget()

        for job_id, in_path, out_path in dispatched:
            self._last_output_path = out_path
            self._last_review_log = f"{out_path}.needs_review.log"

        self.log(f"[Queue] Dispatched {len(dispatched)} document(s) via {mode.value.upper()} ({direction}).")
        self.staged_list.clear()

    def add_to_queue_only(self):
        staged = self.staged_list.get_files()
        if not staged:
            messagebox.showwarning("No Documents", "Please select one or more documents first.")
            return
        self.start_translation_flow()

    # ── Job Updates & Queue Monitoring ──

    def update_job(self, job: TranslationJob):
        if job.id not in self._job_widgets:
            self.queue_empty_label.pack_forget()

            row = JobRow(
                self.queue_container,
                job=job,
                on_cancel=lambda jid: self.controller.cancel_job(jid),
            )
            row.pack(fill="x", pady=3)
            self._job_widgets[job.id] = row

        w = self._job_widgets[job.id]
        if isinstance(w, JobRow):
            w.update_job(job)
        else:
            # Fallback if dictionary-based mock
            pct = job.progress / 100.0
            if "progress" in w:
                w["progress"].set(pct)
            if "status_label" in w:
                st_lbl = w["status_label"]
                if job.status == JobStatus.QUEUED:
                    st_lbl.configure(text="⏳ Queued", text_color=THEME["text_secondary"], cursor="")
                elif job.status == JobStatus.RUNNING:
                    eta_str = getattr(job, "eta_str", "")
                    if not eta_str and job.started_at and job.progress > 0 and job.progress < 100.0:
                        elapsed = max(0.0, time.time() - job.started_at)
                        rem = elapsed * (100.0 - job.progress) / job.progress
                        eta_str = format_eta(rem)

                    msg = job.progress_message or ""
                    if len(msg) > 20:
                        msg = msg[:18] + "..."
                    if eta_str:
                        status_text = f"🔄 {job.progress:.0f}% ({eta_str}) {msg}".strip()
                    else:
                        status_text = f"🔄 {job.progress:.0f}% {msg}".strip()
                    st_lbl.configure(text=status_text, text_color=THEME["primary"], cursor="")
                elif job.status == JobStatus.COMPLETED:
                    elapsed = ""
                    if job.started_at and job.completed_at:
                        secs = max(0, int(job.completed_at - job.started_at))
                        m, s = divmod(secs, 60)
                        elapsed = f" ({m:02d}:{s:02d})"

                    stats = getattr(job, "result", None) or {}
                    reverted = stats.get("reverted", 0)
                    skipped = stats.get("skipped", 0)
                    translated = stats.get("translated", 0)

                    suffix_parts = []
                    if reverted > 0:
                        suffix_parts.append(f"⚠ {reverted} reverted")
                    if skipped > 0:
                        suffix_parts.append(f"{skipped} skipped")
                    suffix = f" — {', '.join(suffix_parts)}" if suffix_parts else ""

                    if "translated" in stats:
                        status_text = f"✅ Done{elapsed} — {translated} translated{suffix}"
                    else:
                        status_text = f"✅ Done{elapsed}"

                    st_lbl.configure(text=status_text, text_color=THEME["success"], cursor="")
                elif job.status == JobStatus.FAILED:
                    err_msg = (
                        job.error.title
                        if (job.error and hasattr(job.error, "title"))
                        else (job.error_message or "Failed")
                    )
                    st_lbl.configure(text=f"❌ {err_msg}", text_color=THEME["error"], cursor="hand2")
                    if job.error and hasattr(job.error, "format_user_dialog"):
                        eo = job.error
                        st_lbl.bind(
                            "<Button-1>",
                            lambda e, err=eo: messagebox.showerror(
                                f"Translation Error [{err.code.value}]", err.format_user_dialog()
                            ),
                        )
                    elif job.error_message:
                        em = job.error_message
                        st_lbl.bind("<Button-1>", lambda e, msg=em: messagebox.showerror("Translation Error", msg))
                elif job.status == JobStatus.CANCELLED:
                    st_lbl.configure(text="⛔ Cancelled", text_color=THEME["text_secondary"], cursor="")
            if "cancel_btn" in w:
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) or (
                    job.status == JobStatus.RUNNING and getattr(job, "cancel_event", None) and job.cancel_event.is_set()
                ):
                    w["cancel_btn"].configure(state="disabled")

        if job.status == JobStatus.COMPLETED:
            if job.output_path:
                self._last_output_path = job.output_path
                if job.output_path not in self._completed_outputs:
                    self._completed_outputs.append(job.output_path)
            if job.review_log_path:
                self._last_review_log = job.review_log_path
                if os.path.exists(job.review_log_path) and os.path.getsize(job.review_log_path) > 0:
                    if job.review_log_path not in self._completed_review_logs:
                        self._completed_review_logs.append(job.review_log_path)

        if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
            self._update_post_action_buttons()

            # Check if all jobs in current batch have reached a terminal state
            batch_ids = getattr(self, "_current_batch_ids", None)
            summary_shown = getattr(self, "_batch_summary_shown", False)
            if batch_ids and not summary_shown:
                ctrl = getattr(self, "controller", None)
                if ctrl and hasattr(ctrl, "is_batch_complete") and ctrl.is_batch_complete(batch_ids):
                    self._batch_summary_shown = True
                    self._show_batch_summary()

    def _update_post_action_buttons(self):
        num_out = len(self._completed_outputs)
        if num_out == 0:
            if self._last_output_path and os.path.exists(self._last_output_path):
                self.open_file_btn.configure(text="📄  Open Output File", state="normal")
                self.open_dir_btn.configure(text="📁  Open Folder", state="normal")
            else:
                self.open_file_btn.configure(text="📄  Open Output File", state="disabled")
                self.open_dir_btn.configure(text="📁  Open Folder", state="disabled")
        elif num_out == 1:
            self.open_file_btn.configure(text="📄  Open Output File", state="normal")
            self.open_dir_btn.configure(text="📁  Open Folder", state="normal")
        else:
            self.open_file_btn.configure(text=f"📄  Open All Outputs ({num_out})", state="normal")
            self.open_dir_btn.configure(text="📁  Open Output Folders", state="normal")

        num_rev = len(self._completed_review_logs)
        if num_rev == 0:
            if (
                self._last_review_log
                and os.path.exists(self._last_review_log)
                and os.path.getsize(self._last_review_log) > 0
            ):
                self.review_btn.configure(text="⚠  Review Log", state="normal")
            else:
                self.review_btn.configure(text="⚠  Review Log", state="disabled")
        elif num_rev == 1:
            self.review_btn.configure(text="⚠  Review Log", state="normal")
        else:
            self.review_btn.configure(text=f"⚠  Review Logs ({num_rev})", state="normal")

    # ── Batch Summary Card ──

    def _build_batch_summary_card(self, parent):
        self.batch_summary_card = ctk.CTkFrame(
            parent,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )

        # Header Frame
        head = ctk.CTkFrame(self.batch_summary_card, fg_color="transparent")
        head.pack(fill="x", padx=16, pady=(12, 6))

        ctk.CTkLabel(
            head,
            text="📊  Batch Summary",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(side="left")

        ctk.CTkButton(
            head,
            text="✕",
            width=26,
            height=26,
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=THEME["text_secondary"],
            hover_color=THEME["btn_secondary"],
            command=self._dismiss_batch_summary,
        ).pack(side="right")

        # Body Frame
        body = ctk.CTkFrame(self.batch_summary_card, fg_color="transparent")
        body.pack(fill="x", padx=16, pady=(2, 6))

        self.summary_files_lbl = ctk.CTkLabel(
            body,
            text="Files: 0 completed, 0 failed",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_primary"],
            anchor="w",
        )
        self.summary_files_lbl.pack(fill="x", pady=2)

        self.summary_chunks_lbl = ctk.CTkLabel(
            body,
            text="Chunks: 0 translated",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            anchor="w",
        )
        self.summary_chunks_lbl.pack(fill="x", pady=2)

        self.summary_time_lbl = ctk.CTkLabel(
            body,
            text="Total time: 0s",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            anchor="w",
        )
        self.summary_time_lbl.pack(fill="x", pady=2)

        self.summary_review_lbl = ctk.CTkLabel(
            body,
            text="Review logs: None",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            anchor="w",
        )
        self.summary_review_lbl.pack(fill="x", pady=2)

        # Button Frame
        btn_bar = ctk.CTkFrame(self.batch_summary_card, fg_color="transparent")
        btn_bar.pack(fill="x", padx=16, pady=(8, 12))

        self.summary_open_folder_btn = ctk.CTkButton(
            btn_bar,
            text="📁  Open Output Folder",
            font=ctk.CTkFont(size=12, weight="bold"),
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._open_batch_common_folder,
        )
        self.summary_open_folder_btn.pack(side="left", padx=(0, 8))

        self.summary_copy_btn = ctk.CTkButton(
            btn_bar,
            text="📋  Copy Summary",
            font=ctk.CTkFont(size=12),
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._copy_batch_summary,
        )
        self.summary_copy_btn.pack(side="left")

    def _dismiss_batch_summary(self):
        """Hides the batch summary card."""
        if hasattr(self, "batch_summary_card") and self.batch_summary_card:
            self.batch_summary_card.pack_forget()

    def _show_batch_summary(self):
        """Populates and displays the batch summary card."""
        summary = self.controller.get_batch_summary(self._current_batch_ids)
        self._last_batch_summary = summary

        # 1. Files line
        completed_c = summary.get("completed_count", 0)
        failed_c = summary.get("failed_count", 0)
        cancelled_c = summary.get("cancelled_count", 0)
        files_text = f"Files: {completed_c} completed, {failed_c} failed"
        if cancelled_c > 0:
            files_text += f", {cancelled_c} cancelled"
        self.summary_files_lbl.configure(text=files_text)

        # 2. Chunks line
        trans_c = summary.get("translated_chunks", 0)
        rev_c = summary.get("reverted_chunks", 0)
        skip_c = summary.get("skipped_chunks", 0)
        chunk_parts = [f"{trans_c} translated"]
        if rev_c > 0:
            chunk_parts.append(f"{rev_c} reverted")
        if skip_c > 0:
            chunk_parts.append(f"{skip_c} skipped")
        self.summary_chunks_lbl.configure(text=f"Chunks: {', '.join(chunk_parts)}")

        # 3. Total time line
        self.summary_time_lbl.configure(text=f"Total time: {summary.get('formatted_time', '0s')}")

        # 4. Review logs line
        rev_paths = summary.get("review_log_paths", [])
        if not rev_paths:
            self.summary_review_lbl.configure(
                text="Review logs: None (clean translation)", text_color=THEME["text_secondary"]
            )
        else:
            num = len(rev_paths)
            self.summary_review_lbl.configure(
                text=f"Review logs: {num} file{'s' if num != 1 else ''} {'have' if num != 1 else 'has'} items to review",
                text_color=THEME["warning"],
            )

        # Folder button state
        has_outputs = bool(summary.get("output_paths") or summary.get("common_dir"))
        self.summary_open_folder_btn.configure(state="normal" if has_outputs else "disabled")

        if getattr(self, "bottom_box", None):
            self.batch_summary_card.pack(fill="x", pady=(0, 14), before=self.bottom_box)
        else:
            self.batch_summary_card.pack(fill="x", pady=(0, 14))

    def _open_batch_common_folder(self):
        """Opens the folder containing the batch outputs."""
        if not self._last_batch_summary:
            return
        target_dir = self._last_batch_summary.get("common_dir")
        if not target_dir or not os.path.exists(target_dir):
            outputs = self._last_batch_summary.get("output_paths", [])
            for out_p in outputs:
                parent = os.path.dirname(os.path.abspath(out_p))
                if os.path.exists(parent):
                    target_dir = parent
                    break

        if not target_dir or not os.path.exists(target_dir):
            messagebox.showinfo("No Output Folder", "No output folders were found for this batch.")
            return

        try:
            if sys.platform == "win32":
                subprocess.Popen(["explorer", target_dir])
            else:
                subprocess.Popen(["xdg-open", target_dir])
        except Exception as e:
            messagebox.showerror("Cannot Open Folder", f"Could not open {target_dir}:\n{e}")

    def _copy_batch_summary(self):
        """Copies the summary text to the clipboard and provides visual feedback."""
        if not self._last_batch_summary:
            return
        text = self._last_batch_summary.get("formatted_text", "")
        if not text:
            return

        try:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.update()
        except Exception:
            pass

        self.summary_copy_btn.configure(text="✅  Copied!")
        self.after(2000, lambda: self.summary_copy_btn.configure(text="📋  Copy Summary"))

    def clear_completed_jobs(self):
        self.controller.clear_completed()
        to_remove = []
        for jid, w in self._job_widgets.items():
            if self.controller.get_job(jid) is None:
                if isinstance(w, JobRow):
                    w.destroy()
                elif isinstance(w, dict) and "row" in w:
                    w["row"].destroy()
                to_remove.append(jid)

        for jid in to_remove:
            del self._job_widgets[jid]

        self._completed_outputs.clear()
        self._completed_review_logs.clear()
        self._last_output_path = None
        self._last_review_log = None
        self._current_batch_ids.clear()
        self._batch_summary_shown = False
        if hasattr(self, "batch_summary_card") and self.batch_summary_card:
            self.batch_summary_card.pack_forget()
        self._update_post_action_buttons()

        if not self._job_widgets:
            self.queue_empty_label.pack(pady=14)

    # ── Post Actions ──

    def open_last_output(self):
        targets = [p for p in self._completed_outputs if os.path.exists(p)]
        if not targets and self._last_output_path and os.path.exists(self._last_output_path):
            targets = [self._last_output_path]
        if not targets:
            messagebox.showinfo("No Output Files", "No completed output files found.")
            return

        if len(targets) > 5:
            if not messagebox.askyesno("Open All Output Files", f"Open all {len(targets)} translated documents?"):
                return

        for p in targets:
            try:
                if sys.platform == "win32":
                    os.startfile(p)
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception as e:
                messagebox.showerror("Cannot Open File", f"Could not open {p}:\n{e}")

    def open_last_dir(self):
        targets = self._completed_outputs or ([self._last_output_path] if self._last_output_path else [])
        dirs = list(dict.fromkeys(os.path.dirname(os.path.abspath(p)) for p in targets if p))
        valid_dirs = [d for d in dirs if os.path.exists(d)]
        if not valid_dirs:
            messagebox.showinfo("No Folders", "No output folders found.")
            return

        for d in valid_dirs:
            try:
                if sys.platform == "win32":
                    subprocess.Popen(["explorer", d])
                else:
                    subprocess.Popen(["xdg-open", d])
            except Exception as e:
                messagebox.showerror("Cannot Open Folder", f"Could not open {d}:\n{e}")

    def view_review_log(self):
        targets = [p for p in self._completed_review_logs if os.path.exists(p) and os.path.getsize(p) > 0]
        if not targets and self._last_review_log and os.path.exists(self._last_review_log):
            targets = [self._last_review_log]
        if not targets:
            messagebox.showinfo("No Review Logs", "No review logs were generated for completed jobs.")
            return

        for p in targets:
            try:
                if sys.platform == "win32":
                    os.startfile(p)
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception as e:
                messagebox.showerror("Cannot Open Review Log", f"Could not open {p}:\n{e}")

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        if self.on_log_cb:
            self.on_log_cb(msg)
