"""
gui/views/quick_view.py
=======================
Quick translate view supporting side-by-side text lookup powered by Ollama.
Supports custom glossary rules, number masking, and placeholder verification.
"""

import os
import threading
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk
import requests

from engine.backend_llm import LLMBackend
from engine.core import (
    mask_glossary_terms,
    mask_numbers,
    should_translate,
    unmask_protected_text,
    verify_placeholders,
)
from engine.preflight import check_ollama_status
from gui.dnd_helper import is_point_in_widget
from gui.theme import (
    GEMMA_PRESETS,
    GLOSSARY_DIR,
    LANGUAGE_PAIRS,
    LAST_GLOSSARY_PATH,
    THEME,
    parse_glossary_text,
)


class QuickView(ctk.CTkFrame):
    """
    Renders the split-pane quick text translation workspace.
    """

    glossary_btn: Any = None
    glossary_load_btn: Any = None
    glossary_save_btn: Any = None
    glossary_drawer: Any = None
    glossary_text: Any = None
    pane: Any = None

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self._glossary_open = False

        self._build_ui()
        self._auto_load_glossary()
        self._setup_dnd()

    def _build_ui(self):
        # Header
        header = ctk.CTkFrame(self, fg_color="transparent")
        header.pack(fill="x", pady=(0, 12))

        ctk.CTkLabel(
            header,
            text="Quick Text Translation",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(side="left")

        # Top Control Bar
        toolbar = ctk.CTkFrame(
            self, fg_color=THEME["card_bg"], border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        toolbar.pack(fill="x", pady=(0, 12))

        tb_inner = ctk.CTkFrame(toolbar, fg_color="transparent")
        tb_inner.pack(fill="x", padx=12, pady=8)

        # Direction Selector
        self.quick_dir_var = ctk.StringVar(value="ja2en")
        dir_display_values = [p[0] for p in LANGUAGE_PAIRS]
        self.quick_dir_combo = ctk.CTkComboBox(
            tb_inner,
            values=dir_display_values,
            variable=ctk.StringVar(value=dir_display_values[0]),
            width=180,
            state="readonly",
            font=ctk.CTkFont(size=12),
            command=self._on_quick_dir_changed,
        )
        self.quick_dir_combo.pack(side="left", padx=(0, 4))

        ctk.CTkButton(
            tb_inner,
            text="⇄",
            width=32,
            height=28,
            font=ctk.CTkFont(size=14, weight="bold"),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._swap_quick_dir,
        ).pack(side="left", padx=(0, 14))

        # Model Selector
        ctk.CTkLabel(
            tb_inner, text="Model:", font=ctk.CTkFont(size=11, weight="bold"), text_color=THEME["text_secondary"]
        ).pack(side="left", padx=(0, 4))

        self.model_var = ctk.StringVar(value=GEMMA_PRESETS[0])
        self.quick_model_combo = ctk.CTkComboBox(
            tb_inner, values=GEMMA_PRESETS, variable=self.model_var, width=180, font=ctk.CTkFont(size=12)
        )
        self.quick_model_combo.pack(side="left")

        # Copy & Clear
        ctk.CTkButton(
            tb_inner,
            text="📋 Copy",
            width=70,
            height=28,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self._copy_quick_translation,
        ).pack(side="right", padx=(4, 0))

        ctk.CTkButton(
            tb_inner,
            text="🗑 Clear",
            width=70,
            height=28,
            fg_color="transparent",
            text_color=THEME["text_secondary"],
            hover_color=THEME["btn_secondary"],
            command=self._clear_quick_text,
        ).pack(side="right", padx=4)

        # Collapsible Glossary
        glossary_toggle_frame = ctk.CTkFrame(self, fg_color="transparent")
        glossary_toggle_frame.pack(fill="x", pady=(0, 6))

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

        self.glossary_drawer = ctk.CTkFrame(self, fg_color="transparent")
        self.glossary_text = ctk.CTkTextbox(
            self.glossary_drawer,
            height=70,
            font=ctk.CTkFont(family="Consolas", size=11),
            fg_color=THEME["staging_bg"],
        )
        self.glossary_text.pack(fill="x", pady=(0, 6))
        self.glossary_text.insert("0.0", "# Term -> Translation (one per line)\n")

        # Split Text Pane
        self.pane = ctk.CTkFrame(self, fg_color="transparent")
        self.pane.pack(fill="both", expand=True)
        self.pane.columnconfigure(0, weight=1)
        self.pane.columnconfigure(1, weight=1)
        self.pane.rowconfigure(0, weight=1)

        # Left Source Box
        src_card = ctk.CTkFrame(
            self.pane, fg_color=THEME["card_bg"], border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        src_card.grid(row=0, column=0, sticky="nsew", padx=(0, 6))

        src_card.rowconfigure(0, weight=0)
        src_card.rowconfigure(1, weight=1)
        src_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            src_card, text="Source Text", font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.quick_source = ctk.CTkTextbox(
            src_card, font=ctk.CTkFont(size=13), wrap="word", fg_color="transparent", border_width=0
        )
        self.quick_source.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))
        self.quick_source.bind("<Control-Return>", lambda e: self._start_quick_translate())

        # Right Target Box
        tgt_card = ctk.CTkFrame(
            self.pane, fg_color=THEME["card_bg"], border_color=THEME["card_border"], border_width=1, corner_radius=10
        )
        tgt_card.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        tgt_card.rowconfigure(0, weight=0)
        tgt_card.rowconfigure(1, weight=1)
        tgt_card.columnconfigure(0, weight=1)

        ctk.CTkLabel(
            tgt_card,
            text="Translation Output (Ollama)",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_secondary"],
        ).grid(row=0, column=0, sticky="w", padx=14, pady=(10, 4))

        self.quick_target = ctk.CTkTextbox(
            tgt_card, font=ctk.CTkFont(size=13), wrap="word", fg_color="transparent", border_width=0, state="disabled"
        )
        self.quick_target.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Bottom Action Bar
        bottom = ctk.CTkFrame(self, fg_color="transparent")
        bottom.pack(fill="x", pady=(12, 0))

        self.quick_translate_btn = ctk.CTkButton(
            bottom,
            text="▶   Translate Text",
            width=160,
            height=38,
            corner_radius=8,
            font=ctk.CTkFont(size=13, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._start_quick_translate,
        )
        self.quick_translate_btn.pack(side="left", padx=(0, 10))

        self.quick_status = ctk.CTkLabel(
            bottom,
            text="Press Ctrl+Enter or click 'Translate Text'",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color=THEME["text_secondary"],
        )
        self.quick_status.pack(side="left")

    def update_models(self, model_list: list[str]) -> None:
        """Updates available models in the combo dropdown."""
        if model_list:
            all_models = list(dict.fromkeys(GEMMA_PRESETS + model_list))
            self.quick_model_combo.configure(values=all_models)
            if self.model_var.get() not in all_models:
                self.model_var.set(all_models[0])

    def _start_quick_translate(self) -> None:
        source_text = self.quick_source.get("0.0", "end").strip()
        if not source_text:
            self.quick_status.configure(text="Please enter or paste text to translate.", text_color=THEME["warning"])
            return

        direction = self.quick_dir_var.get()
        model_name = self.model_var.get().strip() or "gemma4:e2b-it-qat"

        self.quick_translate_btn.configure(state="disabled", text="Translating...")
        self.quick_status.configure(text=f"Connecting to Ollama ({model_name})...", text_color=THEME["primary"])

        thread = threading.Thread(
            target=self._quick_translate_worker, args=(source_text, direction, model_name), daemon=True
        )
        thread.start()

    def _quick_translate_worker(self, text: str, direction: str, model: str) -> None:
        try:
            if not check_ollama_status():
                self.after(0, self._set_quick_result, "", "⚠ Ollama is offline. Please start Ollama service.")
                return

            if not should_translate(text, direction):
                self.after(
                    0,
                    self._set_quick_result,
                    text,
                    "ℹ Text does not require translation for selected direction.",
                )
                return

            glossary = parse_glossary_text(self.glossary_text.get("0.0", "end"))
            masked_text, glossary_map = mask_glossary_terms(text, glossary)
            masked_text, number_map = mask_numbers(masked_text)
            placeholder_map = {**number_map, **glossary_map}

            backend = LLMBackend(model_name=model, context_window=4096)
            result, elapsed = backend.translate_single(
                masked_text=masked_text,
                number_map=placeholder_map,
                direction=direction,
            )

            if result and verify_placeholders(result, placeholder_map, masked_text):
                final = unmask_protected_text(result, number_map, glossary_map)
                info_parts = [f"✓ Translated in {elapsed:.1f}s via {model}"]
                if glossary_map:
                    info_parts.append(f"({len(glossary_map)} glossary term{'s' if len(glossary_map) != 1 else ''})")
                self.after(
                    0,
                    self._set_quick_result,
                    final,
                    " ".join(info_parts),
                )
            else:
                self.after(
                    0,
                    self._set_quick_result,
                    "",
                    "⚠ Translation failed — model returned no valid output.",
                )

        except requests.exceptions.ConnectionError:
            self.after(0, self._set_quick_result, "", "⚠ Could not connect to Ollama at http://localhost:11434.")
        except requests.exceptions.Timeout:
            self.after(0, self._set_quick_result, "", "⚠ Request timed out (120s).")
        except Exception as e:
            self.after(0, self._set_quick_result, "", f"⚠ Translation error: {str(e)[:50]}")
        finally:
            self.after(
                0,
                lambda: self.quick_translate_btn.configure(state="normal", text="▶   Translate Text"),
            )

    def _set_quick_result(self, text: str, status: str) -> None:
        self.quick_target.configure(state="normal")
        self.quick_target.delete("0.0", "end")
        if text:
            self.quick_target.insert("0.0", text)
        self.quick_target.configure(state="disabled")
        self.quick_status.configure(text=status, text_color=THEME["success"] if "✓" in status else THEME["warning"])

    def _copy_quick_translation(self) -> None:
        text = self.quick_target.get("0.0", "end").strip()
        if text:
            self.clipboard_clear()
            self.clipboard_append(text)
            self.quick_status.configure(text="📋 Copied to clipboard!", text_color=THEME["success"])

    def _clear_quick_text(self) -> None:
        self.quick_source.delete("0.0", "end")
        self.quick_target.configure(state="normal")
        self.quick_target.delete("0.0", "end")
        self.quick_target.configure(state="disabled")
        self.quick_status.configure(
            text="Cleared. Paste text and click 'Translate Text'.", text_color=THEME["text_secondary"]
        )

    def _on_quick_dir_changed(self, display_value: str) -> None:
        for label, code in LANGUAGE_PAIRS:
            if label == display_value:
                self.quick_dir_var.set(code)
                break

    def _swap_quick_dir(self) -> None:
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
            self.quick_status.configure(
                text="Direction swapped. Click 'Translate Text'.", text_color=THEME["text_secondary"]
            )

    def _toggle_glossary(self):
        if self._glossary_open:
            self.glossary_drawer.pack_forget()
            self.glossary_btn.configure(text="▶  Custom Glossary (Optional)")
            self._glossary_open = False
        else:
            self.glossary_drawer.pack(fill="x", pady=(0, 6), before=self.pane)
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

            msg = f"Loaded {len(parsed)} glossary entries from {os.path.basename(file_path)}."
            if warnings:
                msg += f" (Warnings: {'; '.join(warnings)})"
                messagebox.showwarning("Glossary Limits", "\n".join(warnings))
            self.quick_status.configure(text=f"📋 {msg}", text_color=THEME["success"])
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
            self.quick_status.configure(
                text=f"💾 Saved glossary to {os.path.basename(dest)}.", text_color=THEME["success"]
            )
        except Exception as e:
            messagebox.showerror("Glossary Save Error", f"Failed to save glossary:\n{e}")

    def _setup_dnd(self):
        """Initializes drag-and-drop support via Tkinter extensions if available."""
        for w in (self.glossary_drawer, self.glossary_text):
            if hasattr(w, "drop_target_register"):
                try:
                    w.drop_target_register("DND_Files")
                    w.dnd_bind("<<Drop>>", self._on_tkdnd_drop)
                except Exception:
                    pass

    def _on_tkdnd_drop(self, event):
        """Handles dropped files from Tkinter DnD extension."""
        raw_data = getattr(event, "data", "")
        if not raw_data:
            return
        files = [p.strip().strip("{}") for p in raw_data.split()]
        if files:
            self._load_glossary_file(files[0])

    def _on_window_drop(self, files: list[str], screen_x: int, screen_y: int):
        """Handles dropped files forwarded from window drop hook."""
        if not self.winfo_ismapped() or not files:
            return

        hit_glossary = is_point_in_widget(self.glossary_drawer, screen_x, screen_y) or is_point_in_widget(
            self.glossary_text, screen_x, screen_y
        )
        if hit_glossary:
            glossary_files = [f for f in files if f.lower().endswith((".txt", ".csv"))]
            if glossary_files:
                self._load_glossary_file(glossary_files[0])
            return

        # If dropped onto quick view and it's a text/csv file, load into glossary
        first = files[0]
        if first.lower().endswith((".txt", ".csv")):
            self._load_glossary_file(first)
