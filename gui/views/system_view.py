"""
gui/views/system_view.py
========================
System diagnostics and AI engine status view.
"""

import os
import threading
from tkinter import messagebox
from typing import Callable, List, Optional
import customtkinter as ctk

from engine.backend_nmt import NMTBackend
from engine.core import TranslationEngine
from engine.preflight import (
    check_ollama_status,
    list_installed_models,
    check_ram,
    check_disk_space,
)
from gui.theme import THEME, GEMMA_PRESETS


class SystemView(ctk.CTkFrame):
    """
    Renders the System & AI Diagnostics tab.
    """

    def __init__(
        self,
        master,
        nmt_backend: Optional[NMTBackend] = None,
        on_ollama_status: Optional[Callable[[bool, List[str]], None]] = None,
        on_argos_status: Optional[Callable[[bool, str], None]] = None,
        **kwargs,
    ):
        super().__init__(master, fg_color="transparent", **kwargs)
        self.nmt_backend = nmt_backend or NMTBackend()
        self.on_ollama_status = on_ollama_status
        self.on_argos_status = on_argos_status

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
            text="📦  Argos Offline Neural Engine",
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
            text="🧠  Local Ollama LLM Service",
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

        ctk.CTkButton(
            co_inner,
            text="↻  Refresh Ollama Connection",
            width=180,
            height=32,
            fg_color=THEME["btn_secondary"],
            hover_color=THEME["btn_sec_hover"],
            command=self.refresh_ollama_status,
        ).pack(anchor="w")

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
            text="💻  System Hardware & Memory Diagnostics",
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
            text="💾  Translation Cache",
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
            command=self._clear_cache_gui,
        )
        self.sys_clear_cache_btn.pack(side="left", padx=(0, 10))

        self.sys_cache_msg = ctk.CTkLabel(cache_action_row, text="", font=ctk.CTkFont(size=11))
        self.sys_cache_msg.pack(side="left")

    def refresh_status(self) -> None:
        """Refreshes all diagnostics: Ollama, Argos, and Hardware."""
        self.refresh_ollama_status()
        self.refresh_argos_status()
        self.refresh_hw_status()

    def refresh_ollama_status(self) -> None:
        """Probes local Ollama instance and installed models."""
        alive = check_ollama_status()
        models: List[str] = []
        if alive:
            models = list_installed_models()
            self.sys_ollama_desc.configure(
                text=(
                    f"Service: Online at http://localhost:11434\n"
                    f"Installed Models: {', '.join(models) if models else 'None'}\n"
                    f"(Zero-Load Guarantee: Model weights load strictly on-demand)"
                ),
                text_color=THEME["success"],
            )
        else:
            self.sys_ollama_desc.configure(
                text=(
                    "Service: Offline\n"
                    "Ollama is not running. Fast NMT mode will operate 100% offline without it.\n"
                    "Start the Ollama app to enable Pure LLM mode."
                ),
                text_color=THEME["error"],
            )

        if self.on_ollama_status:
            self.on_ollama_status(alive, models)

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
        confirm = messagebox.askyesno(
            "Clear Translation Cache",
            "Are you sure you want to clear the local translation cache? All cached sentence pairs will be removed.",
        )
        if not confirm:
            return

        engine = TranslationEngine()
        engine.clear_cache()
        for cp in [
            os.path.abspath("translation_cache.json"),
            os.path.abspath(".translation_cache.json"),
        ]:
            if os.path.exists(cp):
                try:
                    os.remove(cp)
                except OSError:
                    pass

        self.sys_cache_msg.configure(text="✓ Cache cleared successfully", text_color=THEME["success"])
        self.after(3000, lambda: self.sys_cache_msg.configure(text=""))
