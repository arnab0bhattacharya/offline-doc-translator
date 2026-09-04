"""
installer/install_argos_packages.py
====================================
Self-contained helper script to download and install Argos Translate
offline neural language packages (JA <-> EN).
Runs independently with zero project-level imports.

Usage:
    python install_argos_packages.py [--direction ja2en|en2ja|both]
"""

import sys
import os

# Ensure safe console output on Windows cp1252
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "argos_install.log")


def log(msg: str):
    print(msg, flush=True)
    try:
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(f"{msg}\n")
    except Exception:
        pass


def install_pair(from_code: str, to_code: str) -> bool:
    try:
        import argostranslate.package
    except ImportError:
        log("[!] argostranslate is not installed in the current Python environment.")
        log("    Please run: pip install argostranslate")
        return False

    try:
        # 1. Check if already installed
        installed = argostranslate.package.get_installed_packages()
        for pkg in installed:
            if pkg.from_code == from_code and pkg.to_code == to_code:
                log(f"[OK] Package {from_code} to {to_code} is already installed.")
                return True

        # 2. Update index
        log(f"[*] Updating Argos package index for {from_code} -> {to_code}...")
        argostranslate.package.update_package_index()
        available = argostranslate.package.get_available_packages()

        target_pkg = None
        for pkg in available:
            if pkg.from_code == from_code and pkg.to_code == to_code:
                target_pkg = pkg
                break

        if not target_pkg:
            log(f"[!] Language pair {from_code} -> {to_code} not found in Argos package index.")
            return False

        # 3. Download and install
        log(f"[*] Downloading offline model: {target_pkg} (~100MB)...")
        download_path = target_pkg.download()
        log(f"[*] Installing model from: {download_path}...")
        argostranslate.package.install_from_path(download_path)
        log(f"[OK] Successfully installed {from_code} -> {to_code} translation package.")
        return True

    except Exception as e:
        log(f"[!] Error installing {from_code} -> {to_code}: {e}")
        return False


def main():
    direction = "both"
    if len(sys.argv) > 1 and sys.argv[1] == "--direction":
        direction = sys.argv[2] if len(sys.argv) > 2 else "both"

    log("=" * 60)
    log(" Argos Translate Language Package Installer")
    log("=" * 60)

    success = True

    if direction in ("both", "ja2en"):
        log("\n--- Installing Japanese -> English Model ---")
        if not install_pair("ja", "en"):
            success = False

    if direction in ("both", "en2ja"):
        log("\n--- Installing English -> Japanese Model ---")
        if not install_pair("en", "ja"):
            success = False

    log("\n" + "=" * 60)
    if success:
        log(" [OK] All requested Argos packages are ready.")
    else:
        log(" [!] Some packages could not be installed automatically.")
        log("     You can also install them anytime directly within the application.")
    log("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
