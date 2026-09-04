"""
build.py
========
Build script to create a standalone Windows executable using PyInstaller.

Usage:
    python build.py

Output:
    dist/OfflineTranslator/OfflineTranslator.exe
"""

import os
import sys
import subprocess
import shutil

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DIST_DIR = os.path.join(PROJECT_ROOT, "dist")
BUILD_DIR = os.path.join(PROJECT_ROOT, "build")
SPEC_FILE = os.path.join(PROJECT_ROOT, "build.spec")

APP_NAME = "OfflineTranslator"


def find_customtkinter_path():
    """Find the customtkinter package installation path."""
    try:
        import customtkinter
        return os.path.dirname(customtkinter.__file__)
    except ImportError:
        print("[!] customtkinter not installed. Run: pip install customtkinter")
        sys.exit(1)


def build():
    print("=" * 60)
    print(f"  Building {APP_NAME} (PyInstaller One-Dir Mode)")
    print("=" * 60)

    # Check PyInstaller
    try:
        import PyInstaller
        print(f"  [OK] PyInstaller {PyInstaller.__version__}")
    except ImportError:
        print("[!] PyInstaller not installed. Run: pip install pyinstaller")
        sys.exit(1)

    ctk_path = find_customtkinter_path()
    print(f"  [OK] CustomTkinter at: {ctk_path}")

    # Clean previous builds
    for d in [BUILD_DIR, os.path.join(DIST_DIR, APP_NAME)]:
        if os.path.exists(d):
            print(f"  [*] Cleaning {d}...")
            shutil.rmtree(d, ignore_errors=True)

    # Build command
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",
        "--clean",
        # One-dir mode (faster startup)
        "--onedir",
        # Windowed mode (no console window)
        "--windowed",
        # Add icon if exists
    ]

    icon_path = os.path.join(PROJECT_ROOT, "assets", "icon.ico")
    if os.path.exists(icon_path):
        cmd.extend(["--icon", icon_path])
        print(f"  [OK] Using icon: {icon_path}")

    # Hidden imports that PyInstaller might miss
    hidden_imports = [
        "customtkinter",
        "darkdetect",
        "psutil",
        "requests",
        "tqdm",
        "engine",
        "engine.core",
        "engine.errors",
        "engine.preflight",
        "engine.backend_nmt",
        "engine.backend_llm",
        "engine.queue_manager",
        "formats",
        "formats.base",
        "formats.pptx_handler",
        "formats.xlsx_handler",
        "formats.docx_handler",
        "formats.pdf_handler",
        "gui",
        "gui.app",
    ]

    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])

    # Add customtkinter data files (theme JSON, etc.)
    cmd.extend(["--add-data", f"{ctk_path};customtkinter/"])

    # Add project packages and assets as data
    for pkg in ["engine", "formats", "gui", "assets"]:
        pkg_path = os.path.join(PROJECT_ROOT, pkg)
        if os.path.isdir(pkg_path):
            cmd.extend(["--add-data", f"{pkg_path};{pkg}/"])

    # Entry point
    cmd.append(os.path.join(PROJECT_ROOT, "main.py"))

    print(f"\n  [*] Running PyInstaller...\n")
    result = subprocess.run(cmd, cwd=PROJECT_ROOT)

    if result.returncode != 0:
        print(f"\n[!] Build FAILED with exit code {result.returncode}")
        sys.exit(1)

    # Copy requirements.txt into dist for reference
    req_src = os.path.join(PROJECT_ROOT, "requirements.txt")
    req_dst = os.path.join(DIST_DIR, APP_NAME, "requirements.txt")
    if os.path.exists(req_src):
        shutil.copy2(req_src, req_dst)

    # Copy assets into dist root for easy access
    assets_src = os.path.join(PROJECT_ROOT, "assets")
    assets_dst = os.path.join(DIST_DIR, APP_NAME, "assets")
    if os.path.exists(assets_src):
        if os.path.exists(assets_dst):
            shutil.rmtree(assets_dst, ignore_errors=True)
        shutil.copytree(assets_src, assets_dst)

    # Clean up intermediate build directory so users don't accidentally run the incomplete exe in build/
    if os.path.exists(BUILD_DIR):
        print("  [*] Cleaning up intermediate build scratch files...")
        shutil.rmtree(BUILD_DIR, ignore_errors=True)

    exe_path = os.path.join(DIST_DIR, APP_NAME, f"{APP_NAME}.exe")
    print("\n" + "=" * 65)
    print(f"  BUILD SUCCESSFUL!")
    print("=" * 65)
    print(f"\n  [PROGRAM EXECUTABLE]")
    print(f"  Path: {exe_path}")
    print(f"  Or simply double-click: Launch_App.bat")
    print(f"\n  NOTE: This is the standalone program itself, NOT an installer.")
    print(f"\n  [CREATING THE INSTALLER WIZARD (.exe setup)]")
    print(f"  To package this into an Inno Setup installer wizard:")
    print(f"  1. Install Inno Setup (free) from https://jrsoftware.org/isdl.php")
    print(f"  2. Open 'installer/setup.iss' in Inno Setup Compiler")
    print(f"  3. Click Build -> Compile")
    print(f"  The installer wizard will be generated at:")
    print(f"  installer/Output/OfflineTranslatorSetup.exe")
    print("=" * 65)


if __name__ == "__main__":
    build()
