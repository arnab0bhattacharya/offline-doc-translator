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
import zipfile
import hashlib
from typing import Tuple

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


def check_privileges():
    """Warns if running with unnecessary administrative privileges."""
    if sys.platform == "win32":
        try:
            import ctypes
            if ctypes.windll.shell32.IsUserAnAdmin() != 0:
                log("[!] Notice: Running as Administrator.")
                log("    Argos packages install into your user profile; admin rights are not required.")
        except Exception:
            pass


def verify_package_archive(archive_path: str) -> Tuple[bool, str]:
    """
    Validates the downloaded .argosmodel package archive:
    1. Computes SHA-256 digest.
    2. Validates zip file integrity.
    3. Prevents path traversal vulnerabilities.
    """
    if not os.path.exists(archive_path):
        return False, ""

    sha256 = hashlib.sha256()
    try:
        with open(archive_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        digest = sha256.hexdigest()

        with zipfile.ZipFile(archive_path, "r") as zf:
            if zf.testzip() is not None:
                log(f"[!] Package zip file failed CRC validation: {archive_path}")
                return False, digest

            for member in zf.infolist():
                norm = os.path.normpath(member.filename)
                if norm.startswith("..") or os.path.isabs(norm) or norm.startswith("/") or norm.startswith("\\"):
                    log(f"[!] Unsafe path in package archive: {member.filename}")
                    return False, digest

        return True, digest
    except Exception as e:
        log(f"[!] Failed to verify package integrity: {e}")
        return False, ""


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

        # 3. Download and verify integrity
        log(f"[*] Downloading offline model: {target_pkg} (~100MB)...")
        download_path = target_pkg.download()

        is_safe, digest = verify_package_archive(download_path)
        if not is_safe:
            log(f"[!] Downloaded package failed integrity verification. Aborting installation.")
            if os.path.exists(download_path):
                try:
                    os.remove(download_path)
                except OSError:
                    pass
            return False

        log(f"[*] Package verified (SHA-256: {digest[:16]}...).")
        log(f"[*] Installing model from: {download_path}...")
        argostranslate.package.install_from_path(download_path)
        log(f"[OK] Successfully installed {from_code} -> {to_code} translation package.")
        return True

    except Exception as e:
        log(f"[!] Error installing {from_code} -> {to_code}: {e}")
        return False


def main():
    check_privileges()

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
        log("     You can also install them anytime directly within the application GUI (System tab).")
    log("=" * 60)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
