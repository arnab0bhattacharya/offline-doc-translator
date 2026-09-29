"""
gui/views/system_view.py
========================
System diagnostics and AI engine status view.
Enterprise 2-engine architecture:
  - ⚡ Machine Translation: Google MADLAD-400 3B (CTranslate2 INT8, Apache 2.0)
  - 🧠 AI Translation: Google Gemma 4 E2B IT QAT via Ollama (Apache 2.0)
"""

import json
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from tkinter import filedialog, messagebox

import customtkinter as ctk

from engine.backend_madlad import MADLADBackend
from engine.cache import CachePolicy
from engine.cache_locations import (
    cleanup_legacy_cache_remnants,
    clear_all_caches,
    get_cache_stats,
)
from engine.madlad_manager import (
    download_madlad_model,
    get_madlad_model_info,
    import_local_madlad_folder,
)
from engine.ollama_manager import get_ollama_manager
from engine.preflight import (
    check_disk_space,
    check_ollama_status,
    check_ram,
    list_installed_models,
)
from engine.system_specs import get_hardware_specs
from gui.theme import PINNED_OLLAMA_MODEL, THEME

SETTINGS_DIR = os.path.expanduser("~/.offline-translator")
SETTINGS_FILE = os.path.join(SETTINGS_DIR, "settings.json")

CACHE_POLICY_MAP = {
    "Encrypted (Default)": CachePolicy.ENCRYPTED_PERSISTENT,
    "In-Memory (Privacy)": CachePolicy.MEMORY_ONLY,
    "Plaintext (Compatibility)": CachePolicy.PLAINTEXT_PERSISTENT,
}
REVERSE_CACHE_POLICY_MAP = {v: k for k, v in CACHE_POLICY_MAP.items()}


def load_app_settings() -> dict:
    try:
        if os.path.isfile(SETTINGS_FILE):
            with open(SETTINGS_FILE, encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return {}


def save_app_settings(settings: dict) -> None:
    try:
        os.makedirs(SETTINGS_DIR, exist_ok=True)
        current = load_app_settings()
        current.update(settings)
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(current, f, indent=2)
    except Exception:
        pass


class SystemView(ctk.CTkFrame):
    """
    Renders the System & AI Diagnostics tab.
    """

    def __init__(
        self,
        master,
        madlad_backend: MADLADBackend | None = None,
        on_ollama_status: Callable[[bool, list[str]], None] | None = None,
        on_mt_status: Callable[[bool, str], None] | None = None,
        on_argos_status: Callable[[bool, str], None] | None = None,  # backward compatibility alias
        nmt_backend: object = None,  # backward compatibility ignore
        nllb_backend: object = None,  # backward compatibility ignore
        controller: object = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.madlad_backend = madlad_backend or MADLADBackend()
        self.on_ollama_status = on_ollama_status
        self.on_mt_status = on_mt_status or on_argos_status
        self.on_argos_status = self.on_mt_status
        self.controller = controller
        self._madlad_cancel_event = None
        self._ollama_cancel_event = None

        # Load persisted cache policy setting if present
        saved = load_app_settings()
        saved_policy_val = saved.get("cache_policy")
        if saved_policy_val and self.controller and hasattr(self.controller, "default_cache_policy"):
            try:
                self.controller.default_cache_policy = CachePolicy(saved_policy_val)
            except Exception:
                pass

        self._build_ui()

    def _build_ui(self):
        scroll = ctk.CTkScrollableFrame(self, fg_color="transparent")
        scroll.pack(fill="both", expand=True)

        # ── Page Header ──
        header = ctk.CTkFrame(scroll, fg_color="transparent")
        header.pack(fill="x", pady=(0, 16))

        ctk.CTkLabel(
            header,
            text="System & AI Diagnostics",
            font=ctk.CTkFont(size=22, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        ctk.CTkLabel(
            header,
            text="Verify installed enterprise AI engines, local Ollama models, and system hardware health.",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(2, 0))

        # ── Card 1: Machine Translation Engine (Google MADLAD-400 3B) ──
        card_madlad = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_madlad.pack(fill="x", pady=(0, 14))

        cm_inner = ctk.CTkFrame(card_madlad, fg_color="transparent")
        cm_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            cm_inner,
            text="Machine Translation Engine (Google MADLAD-400 3B, Apache 2.0)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_madlad_desc = ctk.CTkLabel(
            cm_inner,
            text="Probing MADLAD model files on disk...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_madlad_desc.pack(anchor="w", pady=(6, 8))

        self.sys_madlad_progress = ctk.CTkProgressBar(cm_inner)
        self.sys_madlad_progress.set(0.0)

        madlad_action_row = ctk.CTkFrame(cm_inner, fg_color="transparent")
        madlad_action_row.pack(fill="x", pady=(4, 0))

        self.sys_madlad_download_btn = ctk.CTkButton(
            madlad_action_row,
            text="⬇   Download Model (~3.0 GB)",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._download_madlad_gui,
        )
        self.sys_madlad_download_btn.pack(side="left", padx=(0, 8))

        self.sys_madlad_import_btn = ctk.CTkButton(
            madlad_action_row,
            text="📂  Import Local Folder...",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._import_madlad_gui,
        )
        self.sys_madlad_import_btn.pack(side="left", padx=(0, 8))

        self.sys_madlad_free_btn = ctk.CTkButton(
            madlad_action_row,
            text="🧹  Free Model Memory",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._free_madlad_memory_gui,
        )
        self.sys_madlad_free_btn.pack(side="left", padx=(0, 8))

        self.sys_madlad_cancel_btn = ctk.CTkButton(
            madlad_action_row,
            text="✕  Cancel",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._cancel_madlad_download_gui,
        )

        self.sys_madlad_msg = ctk.CTkLabel(madlad_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_madlad_msg.pack(side="left")

        # ── Card 2: AI Translation Engine (Google Gemma 4 via Ollama) ──
        card_ollama = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_ollama.pack(fill="x", pady=(0, 14))

        co_inner = ctk.CTkFrame(card_ollama, fg_color="transparent")
        co_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            co_inner,
            text="Local AI Translation Engine (Google Gemma 4 via Ollama, Apache 2.0)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_ollama_desc = ctk.CTkLabel(
            co_inner,
            text="Probing Ollama at http://localhost:11434...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_ollama_desc.pack(anchor="w", pady=(6, 10))

        self.sys_ollama_progress = ctk.CTkProgressBar(co_inner)
        self.sys_ollama_progress.set(0.0)

        self.ollama_action_row = ctk.CTkFrame(co_inner, fg_color="transparent")
        self.ollama_action_row.pack(fill="x")

        self.sys_ollama_download_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="⬇   Download Gemma 4 Model (~1.6 GB)",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._download_ollama_model_gui,
        )

        self.sys_ollama_cancel_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="✕  Cancel",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._cancel_ollama_download_gui,
        )

        self.sys_ollama_start_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="▶   Start Ollama in Background",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._start_ollama_gui,
        )

        self.sys_ollama_refresh_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="↻  Refresh Connection",
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self.refresh_ollama_status,
        )

        self.sys_ollama_free_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="🧹  Free Model Memory",
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._free_ollama_memory_gui,
        )

        self.sys_ollama_stop_btn = ctk.CTkButton(
            self.ollama_action_row,
            text="⏹  Stop Ollama",
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._stop_ollama_gui,
        )

        self.sys_ollama_msg = ctk.CTkLabel(self.ollama_action_row, text="", font=ctk.CTkFont(size=11))

        # ── Card 3: Hardware Diagnostics ──
        card_hw = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_hw.pack(fill="x", pady=(0, 14))

        ch_inner = ctk.CTkFrame(card_hw, fg_color="transparent")
        ch_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            ch_inner,
            text="System Hardware & Resource Guard",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_hw_desc = ctk.CTkLabel(
            ch_inner,
            text="Checking system RAM and disk...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_hw_desc.pack(anchor="w", pady=(6, 0))

        # ── Card 4: Translation Cache & Diagnostics ──
        card_cache = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_cache.pack(fill="x", pady=(0, 14))

        cc_inner = ctk.CTkFrame(card_cache, fg_color="transparent")
        cc_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            cc_inner,
            text="Translation Cache & Quality Diagnostics",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        # Cache Policy Row
        policy_frame = ctk.CTkFrame(cc_inner, fg_color="transparent")
        policy_frame.pack(fill="x", pady=(10, 4))

        ctk.CTkLabel(
            policy_frame,
            text="Cache Storage Policy:",
            font=ctk.CTkFont(size=12, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(side="left", padx=(0, 12))

        current_policy = (
            getattr(self.controller, "default_cache_policy", CachePolicy.ENCRYPTED_PERSISTENT)
            if self.controller
            else CachePolicy.ENCRYPTED_PERSISTENT
        )
        initial_display = REVERSE_CACHE_POLICY_MAP.get(current_policy, "Encrypted (Default)")
        self.cache_policy_var = ctk.StringVar(value=initial_display)

        self.cache_policy_menu = ctk.CTkOptionMenu(
            policy_frame,
            values=list(CACHE_POLICY_MAP.keys()),
            variable=self.cache_policy_var,
            command=self._on_cache_policy_changed,
            font=ctk.CTkFont(size=12),
            dropdown_font=ctk.CTkFont(size=12),
            height=30,
            width=230,
            fg_color=THEME["btn_secondary"],
            button_color=THEME["primary"],
            button_hover_color=THEME["primary_hover"],
            text_color=THEME["text_primary"],
        )
        self.cache_policy_menu.pack(side="left")

        self.sys_cache_policy_desc = ctk.CTkLabel(
            cc_inner,
            text=self._get_policy_explanation(current_policy),
            font=ctk.CTkFont(size=11),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_cache_policy_desc.pack(anchor="w", pady=(2, 10))

        self.sys_cache_desc = ctk.CTkLabel(
            cc_inner,
            text="Persistent cache stores previously translated text chunks to accelerate future runs.",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_cache_desc.pack(anchor="w", pady=(0, 8))

        cache_action_row = ctk.CTkFrame(cc_inner, fg_color="transparent")
        cache_action_row.pack(fill="x")

        self.sys_clear_cache_btn = ctk.CTkButton(
            cache_action_row,
            text="🗑   Clear Translation Cache",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._clear_cache_gui,
        )
        self.sys_clear_cache_btn.pack(side="left", padx=(0, 10))

        self.sys_review_btn = ctk.CTkButton(
            cache_action_row,
            text="⚠   Open Latest Review Log",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._open_review_log_gui,
        )
        self.sys_review_btn.pack(side="left", padx=(0, 10))

        self.sys_cache_msg = ctk.CTkLabel(cache_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_cache_msg.pack(side="left")
        self.refresh_cache_status()

    def refresh_status(self) -> None:
        """Refreshes all diagnostics: Ollama, MADLAD, Hardware, and Cache."""
        self.refresh_ollama_status()
        self.refresh_madlad_status()
        self.refresh_hw_status()
        self.refresh_cache_status()

    def refresh_cache_status(self) -> None:
        """Refreshes cache statistics and directory display."""
        try:
            stats = get_cache_stats()
            dir_path = stats["cache_dir"]
            count = stats["file_count"]
            size_kb = stats["total_kb"]
            size_mb = stats["total_mb"]
            size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_kb:.1f} KB"
            desc = (
                f"Application cache store: {dir_path}\n"
                f"Status: {count} cache file(s) ({size_str}) storing encrypted translation pairs."
            )
            self.sys_cache_desc.configure(text=desc)
        except Exception as e:
            self.sys_cache_desc.configure(text=f"Cache store: (Error probing cache: {e})")

    @staticmethod
    def _get_policy_explanation(policy: CachePolicy) -> str:
        if policy == CachePolicy.ENCRYPTED_PERSISTENT:
            return "Encrypted: Translated chunks are AES-GCM encrypted on disk using a machine-derived key."
        elif policy == CachePolicy.MEMORY_ONLY:
            return "In-Memory: Translations are kept in volatile RAM during the session and wiped on exit (Zero disk trace)."
        elif policy == CachePolicy.PLAINTEXT_PERSISTENT:
            return "Plaintext: Translations are stored in readable JSON on disk for interoperability."
        return ""

    def _on_cache_policy_changed(self, choice: str) -> None:
        policy = CACHE_POLICY_MAP.get(choice, CachePolicy.ENCRYPTED_PERSISTENT)
        if self.controller and hasattr(self.controller, "default_cache_policy"):
            self.controller.default_cache_policy = policy
        save_app_settings({"cache_policy": policy.value})
        self.sys_cache_policy_desc.configure(text=self._get_policy_explanation(policy))
        self.sys_cache_msg.configure(text=f"✓ Cache policy set to {choice}", text_color=THEME["success"])
        self.after(3000, lambda: self.sys_cache_msg.configure(text=""))

    def _open_review_log_gui(self) -> None:
        targets: list[str] = []
        if self.controller:
            completed_logs = getattr(self.controller, "completed_review_logs", [])
            targets = [p for p in completed_logs if os.path.exists(p) and os.path.getsize(p) > 0]
            last_rev = getattr(self.controller, "last_review_log", None)
            if not targets and last_rev and os.path.exists(last_rev):
                targets = [last_rev]

        if not targets:
            messagebox.showinfo(
                "Review Log",
                "No active review logs in current session.\nYou can browse and select an existing .needs_review.log file from disk.",
            )
            picked = filedialog.askopenfilename(
                title="Select Review Log",
                filetypes=[
                    ("Review Logs", "*.needs_review.log"),
                    ("Log / Text Files", "*.log;*.txt"),
                    ("All Files", "*.*"),
                ],
            )
            if picked:
                targets = [picked]

        if not targets:
            return

        for p in targets:
            try:
                if sys.platform == "win32":
                    os.startfile(p)
                else:
                    subprocess.Popen(["xdg-open", p])
            except Exception as e:
                messagebox.showerror("Cannot Open Review Log", f"Could not open {p}:\n{e}")

    def refresh_ollama_status(self, async_mode: bool = True) -> None:
        """Probes local Ollama instance and installed models."""
        if hasattr(self, "_mock_return_value") or type(self).__module__.startswith("unittest.mock"):
            async_mode = False

        if async_mode:
            threading.Thread(
                target=lambda: SystemView.refresh_ollama_status(self, async_mode=False),
                daemon=True,
            ).start()
            return

        alive = check_ollama_status()
        models: list[str] = []
        mgr = get_ollama_manager()

        if alive:
            models = list_installed_models()
            has_pinned = PINNED_OLLAMA_MODEL in models or any(m.startswith(PINNED_OLLAMA_MODEL) for m in models)
            loaded = mgr.get_loaded_models()
            spawned = mgr.spawned_by_app
        else:
            has_pinned = False
            loaded = []
            spawned = False

        def update_ui():
            try:
                for btn in (
                    self.sys_ollama_download_btn,
                    self.sys_ollama_start_btn,
                    self.sys_ollama_refresh_btn,
                    self.sys_ollama_free_btn,
                    self.sys_ollama_stop_btn,
                    self.sys_ollama_msg,
                ):
                    btn.pack_forget()

                if alive:
                    loaded_info = f"\nActive in Memory (VRAM/RAM): {', '.join(loaded)}" if loaded else ""

                    if has_pinned:
                        pinned_status = f"Pinned AI Model: ✅ Google Gemma 4 E2B ({PINNED_OLLAMA_MODEL}) is ready"
                        color = THEME["success"]
                        self.sys_ollama_download_btn.configure(text="↻  Re-download / Update Model")
                    else:
                        pinned_status = (
                            f"Pinned AI Model: ⚠ {PINNED_OLLAMA_MODEL} not installed.\n"
                            f"Click 'Download Gemma 4 Model' below to install (~1.6 GB)."
                        )
                        color = THEME["warning"]
                        self.sys_ollama_download_btn.configure(text="⬇   Download Gemma 4 Model (~1.6 GB)")

                    self.sys_ollama_desc.configure(
                        text=(
                            f"Service: Online at http://localhost:11434{loaded_info}\n"
                            f"{pinned_status}\n"
                            f"Installed Ollama Models: {', '.join(models) if models else 'None'}\n"
                            f"(Zero-Load Guarantee: Model weights load strictly on-demand)"
                        ),
                        text_color=color,
                    )
                    self.sys_ollama_download_btn.pack(side="left", padx=(0, 8))
                    self.sys_ollama_refresh_btn.pack(side="left", padx=(0, 8))
                    self.sys_ollama_free_btn.pack(side="left", padx=(0, 8))
                    if spawned:
                        self.sys_ollama_stop_btn.pack(side="left", padx=(0, 8))
                    self.sys_ollama_msg.pack(side="left")
                else:
                    self.sys_ollama_desc.configure(
                        text=(
                            "Service: Offline\n"
                            "Ollama is not running. Machine Translation mode operates 100% offline without it.\n"
                            "Start Ollama in the background to enable AI Translation mode or download Gemma 4."
                        ),
                        text_color=THEME["error"],
                    )
                    self.sys_ollama_start_btn.pack(side="left", padx=(0, 8))
                    self.sys_ollama_refresh_btn.pack(side="left", padx=(0, 8))
                    self.sys_ollama_msg.pack(side="left")

                if self.on_ollama_status:
                    try:
                        self.on_ollama_status(alive, models, loaded if alive else None)
                    except TypeError:
                        self.on_ollama_status(alive, models)
            except Exception:
                pass

        if (
            hasattr(self, "after")
            and callable(getattr(self, "after", None))
            and threading.current_thread() is not threading.main_thread()
        ):
            self.after(0, update_ui)
        else:
            update_ui()

    def _cancel_ollama_download_gui(self) -> None:
        if self._ollama_cancel_event:
            self._ollama_cancel_event.set()
            self.sys_ollama_msg.configure(text="Cancelling download...", text_color=THEME["warning"])

    def _download_ollama_model_gui(self) -> None:
        """Downloads pinned Gemma 4 model via Ollama's /api/pull with streaming progress."""
        self._ollama_cancel_event = threading.Event()
        self.sys_ollama_download_btn.configure(state="disabled")
        self.sys_ollama_cancel_btn.pack(side="left", padx=(0, 8))
        self.sys_ollama_progress.set(0.0)
        self.sys_ollama_progress.pack(fill="x", padx=16, pady=(0, 8))
        self.sys_ollama_msg.configure(
            text=f"Connecting to Ollama to pull {PINNED_OLLAMA_MODEL}...",
            text_color=THEME["primary"],
        )

        def prog_cb(pct: float, status_str: str):
            def update():
                self.sys_ollama_progress.set(pct)
                self.sys_ollama_msg.configure(text=status_str, text_color=THEME["primary"])

            self.after(0, update)

        def worker():
            mgr = get_ollama_manager()
            success, msg = mgr.pull_model(
                PINNED_OLLAMA_MODEL,
                progress_callback=prog_cb,
                cancel_event=self._ollama_cancel_event,
            )

            def done():
                self.sys_ollama_cancel_btn.pack_forget()
                self.sys_ollama_progress.pack_forget()
                self.sys_ollama_download_btn.configure(state="normal")
                if success:
                    self.sys_ollama_msg.configure(
                        text=f"✓ {PINNED_OLLAMA_MODEL} installed successfully!",
                        text_color=THEME["success"],
                    )
                else:
                    self.sys_ollama_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_ollama_status()
                self.after(8000, lambda: self.sys_ollama_msg.configure(text=""))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _start_ollama_gui(self) -> None:
        self.sys_ollama_start_btn.configure(state="disabled", text="Starting Ollama...")
        self.sys_ollama_msg.configure(text="Launching daemon...", text_color=THEME["primary"])

        def worker():
            mgr = get_ollama_manager()
            success, msg = mgr.start_service(timeout=15.0)

            def done():
                self.sys_ollama_start_btn.configure(state="normal", text="▶   Start Ollama in Background")
                if success:
                    self.sys_ollama_msg.configure(text="✓ Service started", text_color=THEME["success"])
                else:
                    self.sys_ollama_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_ollama_status()
                self.after(5000, lambda: self.sys_ollama_msg.configure(text=""))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _free_ollama_memory_gui(self) -> None:
        self.sys_ollama_free_btn.configure(state="disabled", text="Freeing Memory...")
        self.sys_ollama_msg.configure(text="Unloading models...", text_color=THEME["primary"])

        def worker():
            mgr = get_ollama_manager()
            success, msg = mgr.unload_models()

            def done():
                self.sys_ollama_free_btn.configure(state="normal", text="🧹  Free Model Memory")
                if success:
                    self.sys_ollama_msg.configure(text="✓ Memory cleared", text_color=THEME["success"])
                else:
                    self.sys_ollama_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_ollama_status()
                self.after(5000, lambda: self.sys_ollama_msg.configure(text=""))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_ollama_gui(self) -> None:
        self.sys_ollama_stop_btn.configure(state="disabled", text="Stopping...")
        self.sys_ollama_msg.configure(text="Terminating service...", text_color=THEME["primary"])

        def worker():
            mgr = get_ollama_manager()
            mgr.stop_service()

            def done():
                self.sys_ollama_stop_btn.configure(state="normal", text="⏹  Stop Ollama")
                self.sys_ollama_msg.configure(text="✓ Ollama stopped", text_color=THEME["success"])
                self.refresh_ollama_status()
                self.after(5000, lambda: self.sys_ollama_msg.configure(text=""))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    # ── Machine Translation (MADLAD-400 3B) Management ──

    def refresh_madlad_status(self) -> None:
        """Verifies presence, active device, and RAM status of MADLAD-400 3B model."""
        info = get_madlad_model_info()
        installed = info["installed"]
        is_loaded = self.madlad_backend.is_model_loaded()
        dev_desc = getattr(self.madlad_backend, "active_device_description", "CPU INT8")

        if installed:
            loaded_str = (
                f"Loaded in Memory (~3.5 GB, {dev_desc})"
                if is_loaded
                else f"Unloaded (0 MB in RAM, loads on-demand on {dev_desc})"
            )
            self.sys_madlad_desc.configure(
                text=(
                    f"Status: Model installed and verified ({info['size_mb']} MB on disk).\n"
                    f"Location: {info['path']}\n"
                    f"Compute Device: {dev_desc}\n"
                    f"Memory State: {loaded_str}"
                ),
                text_color=THEME["success"],
            )
            self.sys_madlad_download_btn.configure(text="↻  Re-verify Model Files")
            if is_loaded:
                self.sys_madlad_free_btn.configure(state="normal")
            else:
                self.sys_madlad_free_btn.configure(state="disabled")
        else:
            missing_str = ", ".join(info["missing_files"]) if info["missing_files"] else "weights missing"
            self.sys_madlad_desc.configure(
                text=(
                    f"Status: Model not installed ({missing_str}).\n"
                    f"Target Location: {info['path']}\n"
                    f"Target Compute Device: {dev_desc}\n"
                    "Download the INT8 model (~3.0 GB) or import an existing folder for air-gapped offline use."
                ),
                text_color=THEME["warning"],
            )
            self.sys_madlad_download_btn.configure(text="⬇   Download Model (~3.0 GB)")
            self.sys_madlad_free_btn.configure(state="disabled")

        if self.on_mt_status:
            if not installed:
                status_label = "● MT: Not Installed"
            elif is_loaded:
                status_label = "● MT: Active (RAM)"
            else:
                status_label = "● MT: Standby"
            self.on_mt_status(installed, status_label)

    def _free_madlad_memory_gui(self) -> None:
        """Explicitly unloads MADLAD-400 3B model from RAM."""
        self.madlad_backend.unload()
        self.sys_madlad_msg.configure(text="✓ MADLAD-400 model evicted from RAM", text_color=THEME["success"])
        self.refresh_madlad_status()
        self.after(5000, lambda: self.sys_madlad_msg.configure(text=""))

    def _import_madlad_gui(self) -> None:
        """Allows user to select a folder containing MADLAD CTranslate2 model files."""
        folder = filedialog.askdirectory(
            title="Select Folder Containing MADLAD-400 3B Model Files (model.bin, shared_vocabulary.json)"
        )
        if not folder:
            return

        def worker():
            success, msg = import_local_madlad_folder(folder)

            def done():
                if success:
                    messagebox.showinfo("Import Succeeded", msg)
                else:
                    messagebox.showerror("Import Failed", msg)
                self.refresh_madlad_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_madlad_download_gui(self) -> None:
        if self._madlad_cancel_event:
            self._madlad_cancel_event.set()
            self.sys_madlad_msg.configure(text="Cancelling download...", text_color=THEME["warning"])

    def _download_madlad_gui(self) -> None:
        """Downloads MADLAD-400 3B CTranslate2 model weights with visual progress and cancellation."""
        self._madlad_cancel_event = threading.Event()
        self.sys_madlad_download_btn.configure(state="disabled")
        self.sys_madlad_import_btn.configure(state="disabled")
        self.sys_madlad_cancel_btn.pack(side="left", padx=(0, 8))
        self.sys_madlad_progress.set(0.0)
        self.sys_madlad_progress.pack(fill="x", padx=16, pady=(0, 8))
        self.sys_madlad_msg.configure(text="Connecting to HuggingFace Hub...", text_color=THEME["primary"])

        def prog_cb(pct: float, status_str: str):
            def update():
                self.sys_madlad_progress.set(pct)
                self.sys_madlad_msg.configure(text=status_str, text_color=THEME["primary"])

            self.after(0, update)

        def worker():
            success, msg = download_madlad_model(
                progress_cb=prog_cb,
                cancel_event=self._madlad_cancel_event,
            )

            def done():
                self.sys_madlad_cancel_btn.pack_forget()
                self.sys_madlad_progress.pack_forget()
                self.sys_madlad_download_btn.configure(state="normal")
                self.sys_madlad_import_btn.configure(state="normal")
                if success:
                    self.sys_madlad_msg.configure(text="✓ Model installed successfully!", text_color=THEME["success"])
                else:
                    self.sys_madlad_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_madlad_status()
                self.after(8000, lambda: self.sys_madlad_msg.configure(text=""))

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_hw_status(self, async_mode: bool = True) -> None:
        """Queries and displays host machine specifications and translation engine resource allocation."""
        if hasattr(self, "_mock_return_value") or type(self).__module__.startswith("unittest.mock"):
            async_mode = False

        if async_mode:
            threading.Thread(
                target=lambda: SystemView.refresh_hw_status(self, async_mode=False),
                daemon=True,
            ).start()
            return

        try:
            ram_ok, _ = check_ram()
            disk_ok, _ = check_disk_space()
            specs = get_hardware_specs()

            gpu_str = ", ".join(specs.gpus) if specs.gpus else "Integrated Graphics"
            if specs.cuda_available and specs.cuda_device_name:
                accel_str = f"NVIDIA CUDA Enabled ({specs.cuda_device_name}, {specs.cuda_vram_gb} GB VRAM)"
            else:
                accel_str = f"{gpu_str} (CPU Execution — No NVIDIA CUDA GPU detected)"

            allocated_threads = getattr(self.madlad_backend, "intra_threads", min(4, specs.logical_threads))

            txt = (
                f"Processor: {specs.cpu_name} ({specs.physical_cores} Physical Cores, {specs.logical_threads} Logical Processors)\n"
                f"System Memory: {specs.ram_available_gb:.2f} GB available / {specs.ram_total_gb:.2f} GB total ({'Healthy' if ram_ok else 'Low'})\n"
                f"Disk Storage: {specs.disk_free_gb:.2f} GB free / {specs.disk_total_gb:.2f} GB total ({'Healthy' if disk_ok else 'Low'})\n"
                f"Graphics & Compute: {accel_str}\n"
                f"Engine Allocation: {allocated_threads} of {specs.logical_threads} threads allocated (Dynamic Headroom Pool: auto-tuned for {specs.ram_total_gb:.1f} GB system with {specs.ram_available_gb:.2f} GB available)"
            )
        except Exception as e:
            txt = f"Hardware diagnostics error: {e}"

        def update():
            try:
                self.sys_hw_desc.configure(text=txt)
            except Exception:
                pass

        if (
            hasattr(self, "after")
            and callable(getattr(self, "after", None))
            and threading.current_thread() is not threading.main_thread()
        ):
            self.after(0, update)
        else:
            update()

    def _clear_cache_gui(self) -> None:
        stats = get_cache_stats()
        count = stats["file_count"]
        size_kb = stats["total_kb"]
        size_mb = stats["total_mb"]
        size_str = f"{size_mb:.2f} MB" if size_mb >= 1.0 else f"{size_kb:.1f} KB"

        if count == 0:
            messagebox.showinfo("Translation Cache", "Translation cache is already empty.")
            return

        confirm = messagebox.askyesno(
            "Clear Translation Cache",
            f"Are you sure you want to clear the translation cache?\n\n"
            f"Location: {stats['cache_dir']}\n"
            f"Files to remove: {count} ({size_str})\n\n"
            "All cached sentence pairs across all translation modes will be permanently deleted.",
        )
        if not confirm:
            return

        # Clean legacy cache files and backup remnants in CWD if any remain
        cleanup_legacy_cache_remnants(os.path.abspath("."))

        result = clear_all_caches()
        deleted = result["deleted"]
        freed_kb = result["freed_bytes"] / 1024.0
        freed_mb = result["freed_bytes"] / (1024.0 * 1024.0)
        freed_str = f"{freed_mb:.2f} MB" if freed_mb >= 1.0 else f"{freed_kb:.1f} KB"

        if result["failed"] > 0:
            self.sys_cache_msg.configure(
                text=f"⚠ Cleared {deleted} file(s) ({freed_str}), {result['failed']} locked/failed",
                text_color=THEME["warning"],
            )
        else:
            self.sys_cache_msg.configure(
                text=f"✓ Cleared {deleted} cache file(s) ({freed_str} freed)",
                text_color=THEME["success"],
            )
        self.refresh_cache_status()
        self.after(5000, lambda: self.sys_cache_msg.configure(text=""))
