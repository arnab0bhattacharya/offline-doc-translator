"""
gui/views/quick_view.py
=======================
Quick translate view supporting side-by-side text lookup powered by Ollama.
"""

import threading

import customtkinter as ctk
import requests

from engine.backend_llm import LLMBackend
from engine.core import mask_numbers, unmask_numbers
from engine.preflight import check_ollama_status
from gui.theme import GEMMA_PRESETS, LANGUAGE_PAIRS, THEME


class QuickView(ctk.CTkFrame):
    """
    Renders the split-pane quick text translation workspace.
    """

    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)

        self._build_ui()

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

        # Split Text Pane
        pane = ctk.CTkFrame(self, fg_color="transparent")
        pane.pack(fill="both", expand=True)
        pane.columnconfigure(0, weight=1)
        pane.columnconfigure(1, weight=1)
        pane.rowconfigure(0, weight=1)

        # Left Source Box
        src_card = ctk.CTkFrame(
            pane, fg_color=THEME["card_bg"], border_color=THEME["card_border"], border_width=1, corner_radius=10
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
            pane, fg_color=THEME["card_bg"], border_color=THEME["card_border"], border_width=1, corner_radius=10
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

            backend = LLMBackend(model_name=model, context_window=4096)

            masked_text, number_map = mask_numbers(text)
            result, elapsed = backend.translate_single(
                masked_text=masked_text,
                number_map=number_map,
                direction=direction,
            )

            if result:
                final = unmask_numbers(result, number_map)
                self.after(
                    0,
                    self._set_quick_result,
                    final,
                    f"✓ Translated in {elapsed:.1f}s via {model}",
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
