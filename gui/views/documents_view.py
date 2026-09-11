"""
gui/views/documents_view.py
===========================
Main workspace view for multi-document batch translation, staging, and queue monitoring.
"""

import os
import time
import subprocess
from tkinter import messagebox
from typing import Callable, Dict, List, Optional, Union
import customtkinter as ctk

from engine.core import TranslationMode
from engine.cache import CachePolicy
from engine.preflight import check_ollama_status
from engine.queue_manager import TranslationJob, JobStatus, format_eta
from gui.controllers.translation_controller import TranslationController
from gui.theme import THEME, LANGUAGE_PAIRS, GEMMA_PRESETS, parse_glossary_text
from gui.widgets.job_row import JobRow
from gui.widgets.staged_file_list import StagedFileList


class DocumentsView(ctk.CTkFrame):
    """
    Renders the document staging, configuration, queue, and post-action panels.
    """

    def __init__(
        self,
        master,
        controller: Optional[TranslationController] = None,
        on_log: Optional[Callable[[str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.controller = controller or TranslationController()
        self.on_log_cb = on_log

        self._job_widgets: Dict[str, Union[JobRow, Dict]] = {}
        self._last_output_path: Optional[str] = None
        self._last_review_log: Optional[str] = None
        self._glossary_open = False
        self._log_open = False

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

        # ── Card 4: Post-Actions & Activity Drawer ──
        bottom_box = ctk.CTkFrame(scroll, fg_color="transparent")
        bottom_box.pack(fill="x", pady=(0, 16))

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

    # ── Properties & Staged List Integration ──

    @property
    def selected_files(self) -> List[str]:
        return self.staged_list.selected_files

    def _browse_multi_files(self):
        self.staged_list.browse_files()

    def _browse_folder(self):
        self.staged_list.browse_folder(log_cb=self.log)

    def _on_staged_files_changed(self, files: List[str]):
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

    def _toggle_log(self):
        if self._log_open:
            self.log_drawer.pack_forget()
            self.log_drawer_btn.configure(text="Activity Log ▶")
            self._log_open = False
        else:
            self.log_drawer.pack(fill="x", pady=(0, 16))
            self.log_drawer_btn.configure(text="Activity Log ▼")
            self._log_open = True

    def update_model_choices(self, models: List[str]):
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
        glossary = parse_glossary_text(raw_glossary)

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
                    err_msg = job.error.title if (job.error and hasattr(job.error, "title")) else (job.error_message or "Failed")
                    st_lbl.configure(text=f"❌ {err_msg}", text_color=THEME["error"], cursor="hand2")
                    if job.error and hasattr(job.error, "format_user_dialog"):
                        eo = job.error
                        st_lbl.bind("<Button-1>", lambda e, err=eo: messagebox.showerror(f"Translation Error [{err.code.value}]", err.format_user_dialog()))
                    elif job.error_message:
                        em = job.error_message
                        st_lbl.bind("<Button-1>", lambda e, msg=em: messagebox.showerror("Translation Error", msg))
                elif job.status == JobStatus.CANCELLED:
                    st_lbl.configure(text="⛔ Cancelled", text_color=THEME["text_secondary"], cursor="")
            if "cancel_btn" in w:
                if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED):
                    w["cancel_btn"].configure(state="disabled")
                elif job.status == JobStatus.RUNNING and getattr(job, "cancel_event", None) and job.cancel_event.is_set():
                    w["cancel_btn"].configure(state="disabled")

        if job.status == JobStatus.COMPLETED:
            self.open_file_btn.configure(state="normal")
            self.open_dir_btn.configure(state="normal")
            if self._last_review_log and os.path.exists(self._last_review_log) and os.path.getsize(self._last_review_log) > 0:
                self.review_btn.configure(state="normal")

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

        if not self._job_widgets:
            self.queue_empty_label.pack(pady=14)

    # ── Post Actions ──

    def open_last_output(self):
        if self._last_output_path and os.path.exists(self._last_output_path):
            try:
                os.startfile(self._last_output_path)
            except Exception as e:
                messagebox.showerror("Cannot Open File", str(e))

    def open_last_dir(self):
        if self._last_output_path:
            d = os.path.dirname(os.path.abspath(self._last_output_path))
            if os.path.exists(d):
                try:
                    subprocess.Popen(["explorer", d])
                except Exception:
                    pass

    def view_review_log(self):
        if self._last_review_log and os.path.exists(self._last_review_log):
            try:
                os.startfile(self._last_review_log)
            except Exception as e:
                messagebox.showerror("Cannot Open File", str(e))

    def log(self, msg: str):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{msg}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")
        if self.on_log_cb:
            self.on_log_cb(msg)
