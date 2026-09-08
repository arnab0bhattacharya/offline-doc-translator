"""
gui/widgets/job_row.py
======================
Reusable widget representing a single active, queued, or completed translation job row.
"""

import os
from tkinter import messagebox
from typing import Callable, Optional
import customtkinter as ctk

from engine.queue_manager import TranslationJob, JobStatus
from gui.theme import THEME


class JobRow(ctk.CTkFrame):
    """
    Renders and manages a single job row in the translation queue.
    """

    def __init__(
        self,
        master,
        job: TranslationJob,
        on_cancel: Optional[Callable[[str], None]] = None,
        **kwargs
    ):
        super().__init__(master, fg_color=THEME["staging_bg"], corner_radius=8, **kwargs)
        self.job_id = job.id
        self.on_cancel = on_cancel

        ext = os.path.splitext(job.input_path)[1].lower().replace(".", "").upper()
        bcolor = THEME.get(f"badge_{ext.lower()}", THEME["primary"])

        # Format Badge
        self.badge_label = ctk.CTkLabel(
            self, text=f" {ext} ", font=ctk.CTkFont(size=9, weight="bold"),
            fg_color=bcolor, corner_radius=4, text_color="#FFFFFF"
        )
        self.badge_label.pack(side="left", padx=(8, 6), pady=6)

        # File Name
        name = os.path.basename(job.input_path)
        self.name_label = ctk.CTkLabel(
            self, text=name, font=ctk.CTkFont(size=11, weight="bold"),
            width=170, anchor="w", text_color=THEME["text_primary"]
        )
        self.name_label.pack(side="left", padx=4)

        # Progress Bar
        self.progress_bar = ctk.CTkProgressBar(self, height=8, corner_radius=4)
        self.progress_bar.set(0)
        self.progress_bar.pack(side="left", fill="x", expand=True, padx=8)

        # Status Label
        self.st_label = ctk.CTkLabel(
            self, text="Queued", font=ctk.CTkFont(size=10),
            width=130, anchor="w", text_color=THEME["text_secondary"]
        )
        self.st_label.pack(side="left", padx=4)

        # Cancel Button
        self.cancel_button = ctk.CTkButton(
            self, text="✕", width=24, height=22, font=ctk.CTkFont(size=10),
            fg_color=THEME["error"], hover_color="#991B1B",
            command=self._handle_cancel
        )
        self.cancel_button.pack(side="right", padx=8)

        # Aliases for dictionary-style compatibility
        self.row = self
        self.progress = self.progress_bar
        self.status_label = self.st_label
        self.cancel_btn = self.cancel_button

        self.update_job(job)

    def __getitem__(self, key: str):
        aliases = {
            "row": self.row,
            "progress": self.progress,
            "status_label": self.status_label,
            "cancel_btn": self.cancel_btn,
        }
        if key in aliases:
            return aliases[key]
        raise KeyError(key)

    def _handle_cancel(self):
        if self.on_cancel:
            self.on_cancel(self.job_id)

    def update_job(self, job: TranslationJob) -> None:
        """Applies updated status, progress, and error display for the job."""
        pct = job.progress / 100.0
        self.progress_bar.set(pct)

        if job.status == JobStatus.QUEUED:
            self.st_label.configure(text="⏳ Queued", text_color=THEME["text_secondary"], cursor="")
            self.st_label.unbind("<Button-1>")
        elif job.status == JobStatus.RUNNING:
            msg = job.progress_message
            if len(msg) > 20:
                msg = msg[:18] + "..."
            self.st_label.configure(text=f"🔄 {job.progress:.0f}% {msg}", text_color=THEME["primary"], cursor="")
            self.st_label.unbind("<Button-1>")
        elif job.status == JobStatus.COMPLETED:
            elapsed = ""
            if job.started_at and job.completed_at:
                secs = int(job.completed_at - job.started_at)
                m, s = divmod(secs, 60)
                elapsed = f" ({m:02d}:{s:02d})"
            self.st_label.configure(text=f"✅ Done{elapsed}", text_color=THEME["success"], cursor="")
            self.st_label.unbind("<Button-1>")
            self.cancel_button.configure(state="disabled")
            self.progress_bar.set(1.0)
        elif job.status == JobStatus.FAILED:
            if job.error and hasattr(job.error, "title"):
                err_display = job.error.title
            else:
                err_display = job.error_message or "Failed"
            if len(err_display) > 22:
                err_display = err_display[:20] + "..."
            self.st_label.configure(text=f"❌ {err_display}", text_color=THEME["error"])
            self.cancel_button.configure(state="disabled")

            if job.error and hasattr(job.error, "format_user_dialog"):
                self.st_label.configure(cursor="hand2")
                err_obj = job.error
                self.st_label.bind(
                    "<Button-1>",
                    lambda e, eo=err_obj: messagebox.showerror(
                        f"Translation Error [{eo.code.value}]",
                        eo.format_user_dialog(),
                    ),
                )
            elif job.error_message:
                self.st_label.configure(cursor="hand2")
                err_msg = job.error_message
                self.st_label.bind(
                    "<Button-1>",
                    lambda e, msg=err_msg: messagebox.showerror(
                        "Translation Error",
                        msg,
                    ),
                )
        elif job.status == JobStatus.CANCELLED:
            self.st_label.configure(text="⛔ Cancelled", text_color=THEME["text_secondary"], cursor="")
            self.st_label.unbind("<Button-1>")
            self.cancel_button.configure(state="disabled")
