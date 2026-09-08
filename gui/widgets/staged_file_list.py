"""
gui/widgets/staged_file_list.py
===============================
Reusable widget for managing, browsing, and rendering staged translation documents.
"""

import os
from tkinter import filedialog, messagebox
from typing import List, Callable, Optional
import customtkinter as ctk

from formats.registry import SUPPORTED_EXTENSIONS
from gui.theme import THEME


class StagedFileList(ctk.CTkFrame):
    """
    Renders and manages the staged document list and batch selection dialogs.
    """

    def __init__(
        self,
        master,
        on_files_changed: Optional[Callable[[List[str]], None]] = None,
        **kwargs
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.on_files_changed = on_files_changed
        self.selected_files: List[str] = []

        # ── Staged Header ──
        self.header_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.header_frame.pack(fill="x", pady=(0, 6))

        self.count_label = ctk.CTkLabel(
            self.header_frame, text="0 documents staged",
            font=ctk.CTkFont(size=12, weight="bold"), text_color=THEME["text_secondary"]
        )
        self.count_label.pack(side="left")

        self.clear_btn = ctk.CTkButton(
            self.header_frame, text="Clear All", width=70, height=22,
            font=ctk.CTkFont(size=11), fg_color="transparent",
            text_color=THEME["text_secondary"], hover_color=THEME["btn_secondary"],
            command=self.clear
        )
        self.clear_btn.pack(side="right")

        # ── List Container ──
        self.list_frame = ctk.CTkFrame(self, fg_color=THEME["staging_bg"], corner_radius=8)
        self.list_frame.pack(fill="x")

        self.empty_label = ctk.CTkLabel(
            self.list_frame, text="No documents selected yet.",
            font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"], pady=12
        )
        self.empty_label.pack()

    def get_files(self) -> List[str]:
        """Returns a copy of staged file paths."""
        return list(self.selected_files)

    def add_files(self, paths: List[str]) -> int:
        """Adds unique normalized file paths and updates the view."""
        added = 0
        for p in paths:
            norm = os.path.abspath(p)
            if norm not in self.selected_files and os.path.isfile(norm):
                self.selected_files.append(norm)
                added += 1
        if added > 0:
            self._render()
            self._notify_changed()
        return added

    def remove_file(self, path: str) -> None:
        """Removes a file from the staged list."""
        if path in self.selected_files:
            self.selected_files.remove(path)
            self._render()
            self._notify_changed()

    def clear(self) -> None:
        """Clears all staged documents."""
        if self.selected_files:
            self.selected_files.clear()
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

    def browse_folder(self, log_cb: Optional[Callable[[str], None]] = None) -> None:
        """Opens folder dialog, scans for supported documents, and stages them."""
        folder = filedialog.askdirectory(title="Select Folder of Documents")
        if not folder:
            return

        added_paths: List[str] = []
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

        count = len(self.selected_files)
        self.count_label.configure(
            text=f"{count} document{'s' if count != 1 else ''} staged for translation"
        )

        if count == 0:
            self.empty_label = ctk.CTkLabel(
                self.list_frame, text="No documents selected yet.",
                font=ctk.CTkFont(size=11, slant="italic"), text_color=THEME["text_secondary"], pady=12
            )
            self.empty_label.pack()
            return

        for fpath in self.selected_files:
            chip = ctk.CTkFrame(self.list_frame, fg_color=THEME["card_bg"], corner_radius=6)
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
                command=lambda p=fpath: self.remove_file(p)
            ).pack(side="right", padx=6)
