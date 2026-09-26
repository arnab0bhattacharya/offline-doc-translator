"""
gui/views/system_view.py
========================
System diagnostics and AI engine status view.
"""

import os
import threading
from collections.abc import Callable
from tkinter import filedialog, messagebox

import customtkinter as ctk

from engine.backend_madlad import MADLADBackend
from engine.backend_nllb import NLLBBackend
from engine.backend_nmt import NMTBackend
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
from engine.nllb_manager import (
    download_nllb_model,
    get_nllb_model_info,
    import_local_model_folder,
)
from engine.ollama_manager import get_ollama_manager
from engine.preflight import (
    check_disk_space,
    check_ollama_status,
    check_ram,
    list_installed_models,
)
from gui.theme import THEME


class SystemView(ctk.CTkFrame):
    """
    Renders the System & AI Diagnostics tab.
    """

    def __init__(
        self,
        master,
        nmt_backend: NMTBackend | None = None,
        nllb_backend: NLLBBackend | None = None,
        madlad_backend: MADLADBackend | None = None,
        on_ollama_status: Callable[[bool, list[str]], None] | None = None,
        on_argos_status: Callable[[bool, str], None] | None = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.nmt_backend = nmt_backend or NMTBackend()
        self.nllb_backend = nllb_backend or NLLBBackend()
        self.madlad_backend = madlad_backend or MADLADBackend()
        self.on_ollama_status = on_ollama_status
        self.on_argos_status = on_argos_status
        self._nllb_cancel_event = None
        self._madlad_cancel_event = None

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
            text="Verify installed AI engines, local Ollama models, and system hardware health.",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
        ).pack(anchor="w", pady=(2, 0))

        # ── Card: Argos Offline Neural Engine ──
        card_argos = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_argos.pack(fill="x", pady=(0, 14))

        ca_inner = ctk.CTkFrame(card_argos, fg_color="transparent")
        ca_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            ca_inner,
            text="Argos Offline Neural Engine",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_argos_desc = ctk.CTkLabel(
            ca_inner,
            text="Verifying installed language packages on disk...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_argos_desc.pack(anchor="w", pady=(6, 12))

        argos_action_row = ctk.CTkFrame(ca_inner, fg_color="transparent")
        argos_action_row.pack(fill="x")

        self.sys_argos_btn = ctk.CTkButton(
            argos_action_row,
            text="⬇   Download / Reinstall Language Packages (JA ↔ EN)",
            height=34,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._install_argos_packages_gui,
        )
        self.sys_argos_btn.pack(side="left", padx=(0, 10))

        self.sys_argos_msg = ctk.CTkLabel(argos_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_argos_msg.pack(side="left")

        # ── Card: NLLB-200 3.3B Neural Engine ──
        card_nllb = ctk.CTkFrame(
            scroll,
            fg_color=THEME["card_bg"],
            border_color=THEME["card_border"],
            border_width=1,
            corner_radius=12,
        )
        card_nllb.pack(fill="x", pady=(0, 14))

        cn_inner = ctk.CTkFrame(card_nllb, fg_color="transparent")
        cn_inner.pack(fill="x", padx=16, pady=16)

        ctk.CTkLabel(
            cn_inner,
            text="NLLB-200 3.3B Neural Engine (High Fidelity, Meta AI)",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_nllb_desc = ctk.CTkLabel(
            cn_inner,
            text="Probing NLLB model files on disk...",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_nllb_desc.pack(anchor="w", pady=(6, 8))

        self.sys_nllb_progress = ctk.CTkProgressBar(cn_inner)
        self.sys_nllb_progress.set(0.0)

        nllb_action_row = ctk.CTkFrame(cn_inner, fg_color="transparent")
        nllb_action_row.pack(fill="x", pady=(4, 0))

        self.sys_nllb_download_btn = ctk.CTkButton(
            nllb_action_row,
            text="⬇   Download Model (~3.4 GB)",
            height=32,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=THEME["primary"],
            hover_color=THEME["primary_hover"],
            command=self._download_nllb_gui,
        )
        self.sys_nllb_download_btn.pack(side="left", padx=(0, 8))

        self.sys_nllb_import_btn = ctk.CTkButton(
            nllb_action_row,
            text="📂  Import Local Folder...",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._import_nllb_gui,
        )
        self.sys_nllb_import_btn.pack(side="left", padx=(0, 8))

        self.sys_nllb_free_btn = ctk.CTkButton(
            nllb_action_row,
            text="🧹  Free Model Memory",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._free_nllb_memory_gui,
        )
        self.sys_nllb_free_btn.pack(side="left", padx=(0, 8))

        self.sys_nllb_cancel_btn = ctk.CTkButton(
            nllb_action_row,
            text="✕  Cancel",
            height=32,
            font=ctk.CTkFont(size=12),
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            text_color=THEME["btn_sec_text"],
            command=self._cancel_nllb_download_gui,
        )

        self.sys_nllb_msg = ctk.CTkLabel(nllb_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_nllb_msg.pack(side="left")

        # ── Card: MADLAD-400 3B Neural Engine ──
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
            text="MADLAD-400 3B Neural Engine (Document MT, Google Research)",
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

        # ── Card: Ollama Engine ──
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
            text="Local Ollama LLM Service",
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
        self.sys_ollama_desc.pack(anchor="w", pady=(6, 12))

        self.ollama_action_row = ctk.CTkFrame(co_inner, fg_color="transparent")
        self.ollama_action_row.pack(fill="x")

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

        # ── Card: Hardware ──
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
            text="System Hardware & Memory",
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

        # ── Card: Translation Cache ──
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
            text="Translation Cache",
            font=ctk.CTkFont(size=15, weight="bold"),
            text_color=THEME["text_primary"],
        ).pack(anchor="w")

        self.sys_cache_desc = ctk.CTkLabel(
            cc_inner,
            text="Persistent cache stores previously translated text chunks to accelerate future runs.",
            font=ctk.CTkFont(size=12),
            text_color=THEME["text_secondary"],
            justify="left",
        )
        self.sys_cache_desc.pack(anchor="w", pady=(6, 12))

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

        self.sys_cache_msg = ctk.CTkLabel(cache_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_cache_msg.pack(side="left")
        self.refresh_cache_status()

    def refresh_status(self) -> None:
        """Refreshes all diagnostics: Ollama, Argos, NLLB, MADLAD, Hardware, and Cache."""
        self.refresh_ollama_status()
        self.refresh_argos_status()
        self.refresh_nllb_status()
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

    def refresh_ollama_status(self) -> None:
        """Probes local Ollama instance and installed models."""
        alive = check_ollama_status()
        models: list[str] = []
        mgr = get_ollama_manager()

        for btn in (
            self.sys_ollama_start_btn,
            self.sys_ollama_refresh_btn,
            self.sys_ollama_free_btn,
            self.sys_ollama_stop_btn,
            self.sys_ollama_msg,
        ):
            btn.pack_forget()

        if alive:
            models = list_installed_models()
            loaded = mgr.get_loaded_models()
            loaded_info = f"\nActive in Memory (VRAM/RAM): {', '.join(loaded)}" if loaded else ""
            self.sys_ollama_desc.configure(
                text=(
                    f"Service: Online at http://localhost:11434{loaded_info}\n"
                    f"Installed Models: {', '.join(models) if models else 'None'}\n"
                    f"(Zero-Load Guarantee: Model weights load strictly on-demand)"
                ),
                text_color=THEME["success"],
            )
            self.sys_ollama_refresh_btn.pack(side="left", padx=(0, 8))
            self.sys_ollama_free_btn.pack(side="left", padx=(0, 8))
            if mgr.spawned_by_app:
                self.sys_ollama_stop_btn.pack(side="left", padx=(0, 8))
            self.sys_ollama_msg.pack(side="left")
        else:
            self.sys_ollama_desc.configure(
                text=(
                    "Service: Offline\n"
                    "Ollama is not running. Fast NMT mode will operate 100% offline without it.\n"
                    "Start Ollama in the background to enable Pure LLM mode."
                ),
                text_color=THEME["error"],
            )
            self.sys_ollama_start_btn.pack(side="left", padx=(0, 8))
            self.sys_ollama_refresh_btn.pack(side="left", padx=(0, 8))
            self.sys_ollama_msg.pack(side="left")

        if self.on_ollama_status:
            self.on_ollama_status(alive, models)

    def _start_ollama_gui(self) -> None:
        self.sys_ollama_start_btn.configure(state="disabled", text="⏳ Starting Ollama...")
        self.sys_ollama_msg.configure(text="Spawning background service...", text_color=THEME["primary"])

        def worker():
            mgr = get_ollama_manager()
            success, msg = mgr.start_service(timeout=15.0)

            def done():
                self.sys_ollama_start_btn.configure(state="normal", text="▶   Start Ollama in Background")
                self.sys_ollama_msg.configure(
                    text=msg,
                    text_color=THEME["success"] if success else THEME["error"],
                )
                self.refresh_ollama_status()
                if not success:
                    messagebox.showerror("Ollama Startup Failed", msg)

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _free_ollama_memory_gui(self) -> None:
        self.sys_ollama_free_btn.configure(state="disabled", text="Freeing...")

        def worker():
            mgr = get_ollama_manager()
            unloaded = mgr.unload_all_models()

            def done():
                self.sys_ollama_free_btn.configure(state="normal", text="🧹  Free Model Memory")
                if unloaded:
                    txt = f"✓ Evicted {len(unloaded)} model(s) from VRAM: {', '.join(unloaded)}"
                    self.sys_ollama_msg.configure(text=txt, text_color=THEME["success"])
                else:
                    self.sys_ollama_msg.configure(
                        text="ℹ No active models were held in VRAM.", text_color=THEME["text_secondary"]
                    )
                self.refresh_ollama_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _stop_ollama_gui(self) -> None:
        self.sys_ollama_stop_btn.configure(state="disabled", text="Stopping...")

        def worker():
            mgr = get_ollama_manager()
            success, msg = mgr.stop_service()

            def done():
                self.sys_ollama_stop_btn.configure(state="normal", text="⏹  Stop Ollama")
                self.sys_ollama_msg.configure(
                    text=msg,
                    text_color=THEME["success"] if success else THEME["warning"],
                )
                self.refresh_ollama_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_argos_status(self) -> None:
        """Verifies presence of offline Argos translation models."""
        ja_en = self.nmt_backend.is_ready("ja2en")
        en_ja = self.nmt_backend.is_ready("en2ja")

        ready = bool(ja_en and en_ja)
        if ready:
            summary = "● Argos: Ready (JA↔EN)"
            self.sys_argos_desc.configure(
                text=(
                    "Status: All language packages installed and verified.\n"
                    "- Japanese → English: Ready\n"
                    "- English → Japanese: Ready\n"
                    "Offline neural translation is ready."
                ),
                text_color=THEME["success"],
            )
        else:
            missing = []
            if not ja_en:
                missing.append("Japanese → English")
            if not en_ja:
                missing.append("English → Japanese")

            summary = "● Argos: Missing Packages"
            self.sys_argos_desc.configure(
                text=(
                    f"Status: Missing packages: {', '.join(missing)}\n"
                    f"Click the button below to download and install them directly (~100MB each)."
                ),
                text_color=THEME["warning"],
            )

        if self.on_argos_status:
            self.on_argos_status(ready, summary)

    def refresh_nllb_status(self) -> None:
        """Verifies presence and RAM status of NLLB-200 3.3B model."""
        info = get_nllb_model_info()
        installed = info["installed"]
        is_loaded = self.nllb_backend.is_model_loaded()

        if installed:
            loaded_str = (
                "Loaded in Memory (~3.8 GB RAM)" if is_loaded else "Unloaded (0 MB in RAM, loads on translation)"
            )
            self.sys_nllb_desc.configure(
                text=(
                    f"Status: Model installed and verified ({info['size_mb']} MB on disk).\n"
                    f"Location: {info['path']}\n"
                    f"Memory State: {loaded_str}"
                ),
                text_color=THEME["success"],
            )
            self.sys_nllb_download_btn.configure(text="↻  Re-verify Model Files")
            if is_loaded:
                self.sys_nllb_free_btn.configure(state="normal")
            else:
                self.sys_nllb_free_btn.configure(state="disabled")
        else:
            missing_str = ", ".join(info["missing_files"]) if info["missing_files"] else "weights missing"
            self.sys_nllb_desc.configure(
                text=(
                    f"Status: Model not installed ({missing_str}).\n"
                    f"Target Location: {info['path']}\n"
                    "Download the INT8 model (~3.4 GB) or import an existing folder for air-gapped offline use."
                ),
                text_color=THEME["warning"],
            )
            self.sys_nllb_download_btn.configure(text="⬇   Download Model (~3.4 GB)")
            self.sys_nllb_free_btn.configure(state="disabled")

    def _free_nllb_memory_gui(self) -> None:
        self.nllb_backend.unload()
        self.sys_nllb_msg.configure(text="✓ Model memory evicted (0 MB in RAM)", text_color=THEME["success"])
        self.refresh_nllb_status()
        self.after(4000, lambda: self.sys_nllb_msg.configure(text=""))

    def _import_nllb_gui(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder Containing NLLB-200 Model Files")
        if not folder:
            return

        self.sys_nllb_msg.configure(text="Importing model files...", text_color=THEME["primary"])

        def worker():
            ok, msg = import_local_model_folder(folder)

            def done():
                if ok:
                    self.sys_nllb_msg.configure(text="✓ Model imported successfully!", text_color=THEME["success"])
                else:
                    self.sys_nllb_msg.configure(text=f"⚠ Import failed: {msg}", text_color=THEME["error"])
                self.refresh_nllb_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_nllb_download_gui(self) -> None:
        if hasattr(self, "_nllb_cancel_event") and self._nllb_cancel_event:
            self._nllb_cancel_event.set()
            self.sys_nllb_msg.configure(text="Cancelling download...", text_color=THEME["warning"])

    def _download_nllb_gui(self) -> None:
        self.sys_nllb_download_btn.configure(state="disabled", text="Downloading...")
        self.sys_nllb_import_btn.configure(state="disabled")
        self.sys_nllb_free_btn.configure(state="disabled")
        self.sys_nllb_cancel_btn.pack(side="left", padx=(0, 8), after=self.sys_nllb_download_btn)
        self.sys_nllb_progress.pack(fill="x", pady=(0, 6))
        self.sys_nllb_progress.set(0.0)
        self.sys_nllb_msg.configure(text="Initializing download from Hugging Face...", text_color=THEME["primary"])

        self._nllb_cancel_event = threading.Event()

        def prog_cb(pct: float, status_str: str):
            def update():
                self.sys_nllb_progress.set(pct / 100.0)
                self.sys_nllb_msg.configure(text=status_str, text_color=THEME["primary"])

            self.after(0, update)

        def worker():
            success, msg = download_nllb_model(
                progress_cb=prog_cb,
                cancel_event=self._nllb_cancel_event,
            )

            def done():
                self.sys_nllb_cancel_btn.pack_forget()
                self.sys_nllb_download_btn.configure(state="normal", text="⬇   Download Model (~3.4 GB)")
                self.sys_nllb_import_btn.configure(state="normal")
                self.sys_nllb_progress.pack_forget()
                if success:
                    self.sys_nllb_msg.configure(text=f"✓ {msg}", text_color=THEME["success"])
                else:
                    self.sys_nllb_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_nllb_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_madlad_status(self) -> None:
        """Verifies presence and RAM status of MADLAD-400 3B model."""
        info = get_madlad_model_info()
        installed = info["installed"]
        is_loaded = self.madlad_backend.is_model_loaded()

        if installed:
            loaded_str = (
                "Loaded in Memory (~3.5 GB RAM)" if is_loaded else "Unloaded (0 MB in RAM, loads on translation)"
            )
            self.sys_madlad_desc.configure(
                text=(
                    f"Status: Model installed and verified ({info['size_mb']} MB on disk).\n"
                    f"Location: {info['path']}\n"
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
                    "Download the INT8 model (~3.0 GB) or import an existing folder for air-gapped offline use."
                ),
                text_color=THEME["warning"],
            )
            self.sys_madlad_download_btn.configure(text="⬇   Download Model (~3.0 GB)")
            self.sys_madlad_free_btn.configure(state="disabled")

    def _free_madlad_memory_gui(self) -> None:
        self.madlad_backend.unload()
        self.sys_madlad_msg.configure(text="✓ Model memory evicted (0 MB in RAM)", text_color=THEME["success"])
        self.refresh_madlad_status()
        self.after(4000, lambda: self.sys_madlad_msg.configure(text=""))

    def _import_madlad_gui(self) -> None:
        folder = filedialog.askdirectory(title="Select Folder Containing MADLAD-400 Model Files")
        if not folder:
            return

        self.sys_madlad_msg.configure(text="Importing model files...", text_color=THEME["primary"])

        def worker():
            ok, msg = import_local_madlad_folder(folder)

            def done():
                if ok:
                    self.sys_madlad_msg.configure(text="✓ Model imported successfully!", text_color=THEME["success"])
                else:
                    self.sys_madlad_msg.configure(text=f"⚠ Import failed: {msg}", text_color=THEME["error"])
                self.refresh_madlad_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def _cancel_madlad_download_gui(self) -> None:
        if hasattr(self, "_madlad_cancel_event") and self._madlad_cancel_event:
            self._madlad_cancel_event.set()
            self.sys_madlad_msg.configure(text="Cancelling download...", text_color=THEME["warning"])

    def _download_madlad_gui(self) -> None:
        self.sys_madlad_download_btn.configure(state="disabled", text="Downloading...")
        self.sys_madlad_import_btn.configure(state="disabled")
        self.sys_madlad_free_btn.configure(state="disabled")
        self.sys_madlad_cancel_btn.pack(side="left", padx=(0, 8), after=self.sys_madlad_download_btn)
        self.sys_madlad_progress.pack(fill="x", pady=(0, 6))
        self.sys_madlad_progress.set(0.0)
        self.sys_madlad_msg.configure(text="Initializing download from Hugging Face...", text_color=THEME["primary"])

        self._madlad_cancel_event = threading.Event()

        def prog_cb(pct: float, status_str: str):
            def update():
                self.sys_madlad_progress.set(pct / 100.0)
                self.sys_madlad_msg.configure(text=status_str, text_color=THEME["primary"])

            self.after(0, update)

        def worker():
            success, msg = download_madlad_model(
                progress_cb=prog_cb,
                cancel_event=self._madlad_cancel_event,
            )

            def done():
                self.sys_madlad_cancel_btn.pack_forget()
                self.sys_madlad_download_btn.configure(state="normal", text="⬇   Download Model (~3.0 GB)")
                self.sys_madlad_import_btn.configure(state="normal")
                self.sys_madlad_progress.pack_forget()
                if success:
                    self.sys_madlad_msg.configure(text=f"✓ {msg}", text_color=THEME["success"])
                else:
                    self.sys_madlad_msg.configure(text=f"⚠ {msg}", text_color=THEME["error"])
                self.refresh_madlad_status()

            self.after(0, done)

        threading.Thread(target=worker, daemon=True).start()

    def refresh_hw_status(self) -> None:
        """Checks available RAM and disk space."""
        ram_ok, ram_avail = check_ram()
        disk_ok, disk_free = check_disk_space()

        txt = (
            f"Available System RAM: {ram_avail:.2f} GB ({'Healthy' if ram_ok else 'Low'})\n"
            f"Available Disk Space: {disk_free:.2f} GB ({'Healthy' if disk_ok else 'Low'})\n"
            f"Active Memory Guard: Dynamic model flush active."
        )
        self.sys_hw_desc.configure(text=txt)

    def _install_argos_packages_gui(self) -> None:
        self.sys_argos_btn.configure(state="disabled", text="Installing Packages...")
        self.sys_argos_msg.configure(text="Updating package index...", text_color=THEME["primary"])

        def worker():
            try:

                def cb(msg):
                    self.after(0, self.sys_argos_msg.configure, {"text": msg, "text_color": THEME["primary"]})

                ok1 = self.nmt_backend.install_language_pair("ja", "en", log_cb=cb)
                ok2 = self.nmt_backend.install_language_pair("en", "ja", log_cb=cb)

                if ok1 and ok2:
                    self.after(0, self._on_argos_complete, True, "✓ Packages installed successfully!")
                else:
                    self.after(0, self._on_argos_complete, False, "⚠ Could not install all packages. Check internet.")
            except Exception as e:
                self.after(0, self._on_argos_complete, False, f"⚠ Error: {e}")

        threading.Thread(target=worker, daemon=True).start()

    def _on_argos_complete(self, success: bool, msg: str) -> None:
        self.sys_argos_btn.configure(state="normal", text="⬇   Download / Reinstall Language Packages (JA ↔ EN)")
        self.sys_argos_msg.configure(text=msg, text_color=THEME["success"] if success else THEME["error"])
        self.refresh_argos_status()

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
