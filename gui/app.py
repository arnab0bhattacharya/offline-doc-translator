"""
gui/app.py
==========
Modern, Professional Desktop Interface for Offline Document Translator.
Thin orchestration shell coordinating Views, Widgets, and Controllers.
"""

import os
import sys
import time
from tkinter import messagebox

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

from engine.backend_nmt import NMTBackend
from engine.queue_manager import JobStatus, TranslationJob
from gui.controllers import TranslationController
from gui.theme import THEME
from gui.views import DocumentsView, QuickView, SystemView


class TranslatorApp:
    """Modern Desktop UI orchestration shell with Sidebar Navigation."""

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

        # Engine & Controller state
        self.nmt_backend = NMTBackend()
        self.controller = TranslationController(
            on_job_update=self._on_job_update,
            on_log=self._on_job_log,
        )
        self.translation_queue = self.controller.queue
        self.active_tab = "docs"

        # Build UI layout
        self._build_layout()

        # Initial check (Zero-load metadata only)
        self.root.after(100, self.system_view.refresh_status)

    @property
    def selected_files(self) -> list[str]:
        if hasattr(self, "docs_view"):
            return self.docs_view.selected_files
        return []

    @property
    def _last_output_path(self) -> str | None:
        if hasattr(self, "docs_view"):
            return self.docs_view._last_output_path
        return None

    @_last_output_path.setter
    def _last_output_path(self, val: str | None):
        if hasattr(self, "docs_view"):
            self.docs_view._last_output_path = val

    @property
    def _last_review_log(self) -> str | None:
        if hasattr(self, "docs_view"):
            return self.docs_view._last_review_log
        return None

    @_last_review_log.setter
    def _last_review_log(self, val: str | None):
        if hasattr(self, "docs_view"):
            self.docs_view._last_review_log = val

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
                    from PIL import Image, ImageTk

                    img = ImageTk.PhotoImage(Image.open(png))
                    self.root.iconphoto(True, img)
                    break
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════
    #  MAIN LAYOUT & SIDEBAR
    # ══════════════════════════════════════════════════════════════

    def _build_layout(self):
        self.root.configure(fg_color=THEME["bg"])

        self.main_container = ctk.CTkFrame(self.root, fg_color="transparent")
        self.main_container.pack(fill="both", expand=True)

        # Left Navigation Sidebar
        self._build_sidebar()

        # Right Content Area
        self.content_area = ctk.CTkFrame(self.main_container, fg_color="transparent")
        self.content_area.pack(side="left", fill="both", expand=True, padx=(0, 16), pady=16)

        # View Instances
        self.tab_frames: dict[str, ctk.CTkFrame] = {}

        self.docs_view = DocumentsView(self.content_area, controller=self.controller)
        self.quick_view = QuickView(self.content_area)
        self.system_view = SystemView(
            self.content_area,
            nmt_backend=self.nmt_backend,
            on_ollama_status=self._on_ollama_status,
            on_argos_status=self._on_argos_status,
        )

        self.tab_frames["docs"] = self.docs_view
        self.tab_frames["quick"] = self.quick_view
        self.tab_frames["system"] = self.system_view

        self._job_widgets = self.docs_view._job_widgets

        # Show initial tab
        self._switch_tab("docs")

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.main_container, width=230, corner_radius=0, fg_color=THEME["sidebar_bg"])
        self.sidebar.pack(side="left", fill="y", padx=0, pady=0)
        self.sidebar.pack_propagate(False)

        # ── App Branding ──
        brand_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        brand_frame.pack(fill="x", padx=16, pady=(20, 24))

        ctk.CTkLabel(
            brand_frame,
            text="🌐  DocTranslator",
            font=ctk.CTkFont(size=17, weight="bold"),
            text_color=THEME["text_primary"],
            anchor="w",
        ).pack(anchor="w")

        ctk.CTkLabel(
            brand_frame,
            text="Offline Neural & Local LLM",
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"],
            anchor="w",
        ).pack(anchor="w", pady=(2, 0))

        # ── Navigation Buttons ──
        self.nav_buttons = {}
        nav_items = [
            ("docs", "📄  Documents", "Batch translate PPTX, XLSX, DOCX, PDF"),
            ("quick", "⚡  Quick Translate", "Instant side-by-side text lookup"),
            ("system", "⚙  System & AI", "Engines, models, and diagnostics"),
        ]

        nav_frame = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        nav_frame.pack(fill="x", padx=12, pady=0)

        for tab_id, label, _ in nav_items:
            btn = ctk.CTkButton(
                nav_frame,
                text=f"  {label}",
                height=42,
                anchor="w",
                font=ctk.CTkFont(size=13, weight="bold"),
                corner_radius=8,
                fg_color="transparent",
                text_color=THEME["text_secondary"],
                hover_color=THEME["btn_secondary"],
                command=lambda t=tab_id: self._switch_tab(t),
            )
            btn.pack(fill="x", pady=4)
            self.nav_buttons[tab_id] = btn

        # Spacer
        ctk.CTkFrame(self.sidebar, fg_color="transparent").pack(fill="both", expand=True)

        # ── Sidebar Status Card ──
        status_card = ctk.CTkFrame(self.sidebar, fg_color=THEME["card_bg"], corner_radius=8)
        status_card.pack(fill="x", padx=12, pady=(0, 12))

        ctk.CTkLabel(
            status_card,
            text="AI Engine Status",
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", padx=10, pady=(8, 4))

        self.side_argos_btn = ctk.CTkButton(
            status_card,
            text="● Argos: Checking...",
            height=24,
            anchor="w",
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=THEME["warning"],
            hover_color=THEME["btn_secondary"],
            command=lambda: self._switch_tab("system"),
        )
        self.side_argos_btn.pack(fill="x", padx=6, pady=2)

        self.side_ollama_btn = ctk.CTkButton(
            status_card,
            text="● Ollama: Checking...",
            height=24,
            anchor="w",
            font=ctk.CTkFont(size=11),
            fg_color="transparent",
            text_color=THEME["warning"],
            hover_color=THEME["btn_secondary"],
            command=lambda: self.system_view.refresh_ollama_status(),
        )
        self.side_ollama_btn.pack(fill="x", padx=6, pady=(0, 6))

        # ── Theme Toggle in Footer ──
        footer = ctk.CTkFrame(self.sidebar, fg_color="transparent")
        footer.pack(fill="x", padx=14, pady=(0, 16))

        self.theme_switch = ctk.CTkSwitch(
            footer, text="Dark Theme", font=ctk.CTkFont(size=11), command=self._toggle_theme
        )
        self.theme_switch.select()
        self.theme_switch.pack(side="left")

    def _switch_tab(self, tab_id: str):
        # Persist glossary before leaving active tab
        if self.active_tab == "docs" and hasattr(self.docs_view, "persist_glossary"):
            self.docs_view.persist_glossary()
        elif self.active_tab == "quick" and hasattr(self.quick_view, "persist_glossary"):
            self.quick_view.persist_glossary()

        self.active_tab = tab_id
        for tid, frame in self.tab_frames.items():
            if tid == tab_id:
                frame.pack(fill="both", expand=True)
            else:
                frame.pack_forget()

        for tid, btn in self.nav_buttons.items():
            if tid == tab_id:
                btn.configure(fg_color=THEME["primary"], text_color="#FFFFFF")
            else:
                btn.configure(fg_color="transparent", text_color=THEME["text_secondary"])

        # Sync glossary on entering tab
        if tab_id == "docs" and hasattr(self.docs_view, "sync_glossary"):
            self.docs_view.sync_glossary()
        elif tab_id == "quick" and hasattr(self.quick_view, "sync_glossary"):
            self.quick_view.sync_glossary()

    def _toggle_theme(self):
        if self.theme_switch.get():
            ctk.set_appearance_mode("dark")
            self.theme_switch.configure(text="Dark Theme")
        else:
            ctk.set_appearance_mode("light")
            self.theme_switch.configure(text="Light Theme")

    # ══════════════════════════════════════════════════════════════
    #  DIAGNOSTICS & STATUS SYNC
    # ══════════════════════════════════════════════════════════════

    def _on_ollama_status(self, alive: bool, models: list[str]):
        if alive:
            self.side_ollama_btn.configure(text=f"● Ollama: Online ({len(models)})", text_color=THEME["success"])
            if models:
                self.docs_view.update_model_choices(models)
                self.quick_view.update_models(models)
        else:
            self.side_ollama_btn.configure(text="● Ollama: Offline", text_color=THEME["error"])

    def _on_argos_status(self, ready: bool, summary: str):
        self.side_argos_btn.configure(text=summary, text_color=THEME["success"] if ready else THEME["warning"])

    # ══════════════════════════════════════════════════════════════
    #  EVENT DISPATCH & QUEUE UPDATES
    # ══════════════════════════════════════════════════════════════

    def _on_job_update(self, job: TranslationJob):
        self.root.after(0, self._update_job_widget, job)

    def _on_job_log(self, job_id: str, message: str):
        self.root.after(0, self._log, message)

    def _update_job_widget(self, job: TranslationJob):
        # Forward to DocumentsView if mounted
        docs_v = getattr(self, "docs_view", None)
        if isinstance(docs_v, DocumentsView):
            docs_v.update_job(job)
            return

        # Fallback / Mock direct-dictionary updates (for unittest compatibility)
        if not hasattr(self, "_job_widgets") or job.id not in self._job_widgets:
            return

        w = self._job_widgets[job.id]
        pct = job.progress / 100.0
        if "progress" in w:
            w["progress"].set(pct)

        if "status_label" in w:
            st_label = w["status_label"]
            if job.status == JobStatus.QUEUED:
                st_label.configure(text="⏳ Queued", text_color=THEME["text_secondary"], cursor="")
                st_label.unbind("<Button-1>")
            elif job.status == JobStatus.RUNNING:
                eta_str = getattr(job, "eta_str", "")
                if not eta_str and job.started_at and job.progress > 0 and job.progress < 100.0:
                    from engine.queue_manager import format_eta

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
                st_label.configure(text=status_text, text_color=THEME["primary"], cursor="")
                st_label.unbind("<Button-1>")
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

                st_label.configure(text=status_text, text_color=THEME["success"], cursor="")
                st_label.unbind("<Button-1>")
            elif job.status == JobStatus.FAILED:
                if job.error and hasattr(job.error, "title"):
                    err_display = job.error.title
                else:
                    err_display = job.error_message or "Failed"
                if len(err_display) > 22:
                    err_display = err_display[:20] + "..."
                st_label.configure(text=f"❌ {err_display}", text_color=THEME["error"], cursor="hand2")

                if job.error and hasattr(job.error, "format_user_dialog"):
                    err_obj = job.error
                    st_label.bind(
                        "<Button-1>",
                        lambda e, eo=err_obj: messagebox.showerror(
                            f"Translation Error [{eo.code.value}]",
                            eo.format_user_dialog(),
                        ),
                    )
                elif job.error_message:
                    err_msg = job.error_message
                    st_label.bind(
                        "<Button-1>",
                        lambda e, msg=err_msg: messagebox.showerror(
                            "Translation Error",
                            msg,
                        ),
                    )
            elif job.status == JobStatus.CANCELLED:
                st_label.configure(text="⛔ Cancelled", text_color=THEME["text_secondary"], cursor="")
                st_label.unbind("<Button-1>")

        if "cancel_btn" in w:
            if job.status in (JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED) or (
                job.status == JobStatus.RUNNING and getattr(job, "cancel_event", None) and job.cancel_event.is_set()
            ):
                w["cancel_btn"].configure(state="disabled")

    def _log(self, message: str):
        if hasattr(self, "docs_view") and hasattr(self.docs_view, "log"):
            self.docs_view.log(message)

    def cleanup(self):
        """Clean shutdown of translation queue worker thread and auto-persists state."""
        if hasattr(self, "docs_view") and hasattr(self.docs_view, "persist_glossary"):
            self.docs_view.persist_glossary()
        if hasattr(self, "quick_view") and hasattr(self.quick_view, "persist_glossary"):
            self.quick_view.persist_glossary()
        self.controller.shutdown()


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
