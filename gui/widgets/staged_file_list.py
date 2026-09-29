"""
gui/widgets/staged_file_list.py
===============================
Reusable widget for managing, browsing, and rendering staged translation documents
with per-document direction and engine selection.
"""

import os
from collections.abc import Callable
from dataclasses import dataclass
from tkinter import filedialog, messagebox
from typing import Any

import customtkinter as ctk

from formats.registry import SUPPORTED_EXTENSIONS
from gui.theme import PINNED_OLLAMA_MODEL, THEME


@dataclass
class StagedItem:
    """Represents a single document queued for translation with its individual configuration."""

    path: str
    direction: str = "ja2en"
    mode: str = "machine_translation"
    model_name: str = PINNED_OLLAMA_MODEL


class StagedFileList(ctk.CTkFrame):
    """
    Renders and manages the staged document list, batch selection dialogs,
    and individual per-document translation settings.
    """

    def __init__(self, master, on_files_changed: Callable[[list[str]], None] | None = None, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_files_changed = on_files_changed
        self.default_direction: str = "ja2en"
        self.default_mode: str = "machine_translation"
        self.default_model: str = PINNED_OLLAMA_MODEL
        self.staged_items: list[StagedItem] = []

        # ── Staged Header ──
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(0, 6))

        self.count_label = ctk.CTkLabel(
            self.header_frame,
            text="0 documents staged",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_secondary"],
        )
        self.count_label.pack(side="left")

        self.clear_btn = ctk.CTkButton(
            self.header_frame,
            text="Clear All",
            width=80,
            height=28,
            font=ctk.CTkFont(size=11),
            fg_color=THEME["btn_secondary"],
            text_color=THEME["btn_sec_text"],
            hover_color=THEME["btn_sec_hover"],
            command=self.clear,
        )
        self.clear_btn.pack(side="right")

        # ── List Container ──
        self.list_frame = ctk.CTkFrame(self, fg_color=THEME["staging_bg"], corner_radius=8)
        self.list_frame.pack(fill="x")

        self.empty_label = ctk.CTkLabel(
            self.list_frame,
            text="No documents selected yet.",
            font=ctk.CTkFont(size=11, slant="italic"),
            text_color=THEME["text_secondary"],
            pady=12,
        )
        self.empty_label.pack()

    @property
    def selected_files(self) -> list[str]:
        """Provides access to raw file paths for compatibility with existing views/tests."""
        return [item.path for item in self.staged_items]

    @selected_files.setter
    def selected_files(self, paths: list[str]) -> None:
        self.staged_items = [
            StagedItem(
                path=p,
                direction=self.default_direction,
                mode=self.default_mode,
                model_name=self.default_model,
            )
            for p in paths
        ]

    def get_files(self) -> list[str]:
        """Returns a copy of staged file paths."""
        return list(self.selected_files)

    def get_staged_specs(self) -> list[dict[str, Any]]:
        """
        Returns full per-document specifications including path, direction,
        engine mode, and model name.
        """
        return [
            {
                "path": item.path,
                "direction": item.direction,
                "mode": item.mode,
                "model_name": item.model_name,
            }
            for item in self.staged_items
        ]

    def set_default_preset(self, direction: str, mode: str, model_name: str) -> None:
        """Sets the baseline defaults for newly added documents."""
        self.default_direction = direction
        self.default_mode = mode
        self.default_model = model_name

    def apply_preset_to_all(self, direction: str, mode: str, model_name: str) -> None:
        """Updates all currently staged documents to match the provided preset."""
        self.default_direction = direction
        self.default_mode = mode
        self.default_model = model_name
        for item in self.staged_items:
            item.direction = direction
            item.mode = mode
            item.model_name = model_name
        self._render()
        self._notify_changed()

    def add_files(
        self,
        paths: list[str],
        direction: str | None = None,
        mode: str | None = None,
        model_name: str | None = None,
    ) -> int:
        """Adds unique normalized file paths and updates the view."""
        effective_dir = direction or self.default_direction
        effective_mode = mode or self.default_mode
        effective_model = model_name or self.default_model

        added = 0
        existing = {item.path for item in self.staged_items}
        for p in paths:
            norm = os.path.abspath(p)
            if norm not in existing and os.path.isfile(norm):
                self.staged_items.append(
                    StagedItem(
                        path=norm,
                        direction=effective_dir,
                        mode=effective_mode,
                        model_name=effective_model,
                    )
                )
                existing.add(norm)
                added += 1

        if added > 0:
            self._render()
            self._notify_changed()
        return added

    def remove_file(self, path: str) -> None:
        """Removes a file from the staged list."""
        self.staged_items = [item for item in self.staged_items if item.path != path]
        self._render()
        self._notify_changed()

    def clear(self) -> None:
        """Clears all staged documents."""
        if self.staged_items:
            self.staged_items.clear()
            self._render()
            self._notify_changed()

    def browse_files(self) -> None:
        """Opens native file dialog for multi-document selection."""
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
            self.add_files(list(chosen))

    def browse_folder(self, log_cb: Callable[[str], None] | None = None) -> None:
        """Opens folder dialog, scans for supported documents, and stages them."""
        folder = filedialog.askdirectory(title="Select Folder of Documents")
        if not folder:
            return

        added_paths: list[str] = []
        for root_dir, _, files in os.walk(folder):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in SUPPORTED_EXTENSIONS and not file.startswith("~$"):
                    norm = os.path.abspath(os.path.join(root_dir, file))
                    added_paths.append(norm)

        added = self.add_files(added_paths)
        if added > 0:
            if log_cb:
                log_cb(f"[Folder] Imported {added} documents from {folder}")
        else:
            messagebox.showinfo("No Supported Files", "No PPTX, XLSX, DOCX, or PDF files found in folder.")

    def _notify_changed(self) -> None:
        if self.on_files_changed:
            self.on_files_changed(self.get_files())

    def _render(self) -> None:
        for widget in self.list_frame.winfo_children():
            widget.destroy()

        count = len(self.staged_items)
        self.count_label.configure(text=f"{count} document{'s' if count != 1 else ''} staged for translation")

        if count == 0:
            self.empty_label = ctk.CTkLabel(
                self.list_frame,
                text="No documents selected yet.",
                font=ctk.CTkFont(size=11, slant="italic"),
                text_color=THEME["text_secondary"],
                pady=12,
            )
            self.empty_label.pack()
            return

        dir_map_display = {"ja2en": "JA → EN", "en2ja": "EN → JA"}
        dir_map_val = {"JA → EN": "ja2en", "EN → JA": "en2ja"}

        for item in self.staged_items:
            chip = ctk.CTkFrame(self.list_frame, fg_color=THEME["card_bg"], corner_radius=6)
            chip.pack(fill="x", padx=8, pady=4)

            ext = os.path.splitext(item.path)[1].lower().replace(".", "").upper()
            bcolor = THEME.get(f"badge_{ext.lower()}", THEME["primary"])

            # Format badge
            ctk.CTkLabel(
                chip,
                text=f" {ext} ",
                font=ctk.CTkFont(size=9, weight="bold"),
                fg_color=bcolor,
                corner_radius=4,
                text_color="#FFFFFF",
            ).pack(side="left", padx=(8, 6), pady=8)

            # Name & size
            name = os.path.basename(item.path)
            try:
                sz = os.path.getsize(item.path) / 1024
                sz_str = f"{sz / 1024:.1f} MB" if sz > 1024 else f"{sz:.0f} KB"
            except Exception:
                sz_str = ""

            ctk.CTkLabel(
                chip,
                text=f"{name}  ({sz_str})",
                font=ctk.CTkFont(size=11),
                text_color=THEME["text_primary"],
                anchor="w",
            ).pack(side="left", fill="x", expand=True, padx=4)

            # ── Inline Controls Frame (Right-Aligned) ──
            ctrls = ctk.CTkFrame(chip, fg_color="transparent")
            ctrls.pack(side="right", padx=6)

            # Inline Direction Selector
            initial_dir_disp = dir_map_display.get(item.direction, "JA → EN")

            def on_dir_change(new_disp: str, target_item=item):
                target_item.direction = dir_map_val.get(new_disp, "ja2en")
                self._notify_changed()

            dir_combo = ctk.CTkComboBox(
                ctrls,
                values=["JA → EN", "EN → JA"],
                variable=ctk.StringVar(value=initial_dir_disp),
                width=95,
                height=26,
                font=ctk.CTkFont(size=10, weight="bold"),
                state="readonly",
                command=on_dir_change,
            )
            dir_combo.pack(side="left", padx=(0, 4))

            # Inline Engine Selector
            is_mt = "machine" in item.mode.lower() or "nmt" in item.mode.lower()
            initial_mode_disp = "⚡ MT" if is_mt else "🧠 AI"

            def on_mode_change(new_disp: str, target_item=item):
                target_item.mode = "machine_translation" if "MT" in new_disp else "ai_translation"
                target_item.model_name = PINNED_OLLAMA_MODEL
                self._notify_changed()

            mode_combo = ctk.CTkComboBox(
                ctrls,
                values=["⚡ MT", "🧠 AI"],
                variable=ctk.StringVar(value=initial_mode_disp),
                width=80,
                height=26,
                font=ctk.CTkFont(size=10, weight="bold"),
                state="readonly",
                command=on_mode_change,
            )
            mode_combo.pack(side="left", padx=(0, 6))

            # Remove chip button
            ctk.CTkButton(
                ctrls,
                text="✕",
                width=24,
                height=24,
                font=ctk.CTkFont(size=10),
                fg_color="transparent",
                text_color=THEME["btn_sec_text"],
                hover_color=THEME["btn_secondary"],
                command=lambda p=item.path: self.remove_file(p),
            ).pack(side="left")
