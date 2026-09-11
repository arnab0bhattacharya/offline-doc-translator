#!/usr/bin/env python
"""
Run all static analysis checks: Ruff (lint + format) and Bandit (security).

Usage:
    python lint.py          # Run all checks (non-fixing)
    python lint.py --fix    # Auto-fix what Ruff can fix safely
    python lint.py --ci     # Strict mode for CI: exit 1 on any finding
"""
import subprocess
import sys
import time

# ANSI colors (Windows Terminal supports these)
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
BOLD = "\033[1m"
RESET = "\033[0m"

FIX_MODE = "--fix" in sys.argv
CI_MODE = "--ci" in sys.argv

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

SOURCE_DIRS = ["engine/", "formats/", "gui/", "tests/", "main.py"]
overall_exit = 0


def section(title: str) -> None:
    print(f"\n{BOLD}{CYAN}{'=' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  {title}{RESET}")
    print(f"{BOLD}{CYAN}{'=' * 60}{RESET}\n")


def run_tool(name: str, cmd: list[str]) -> int:
    """Run a tool and return its exit code."""
    global overall_exit
    start = time.perf_counter()
    result = subprocess.run(cmd, capture_output=False)
    elapsed = time.perf_counter() - start

    if result.returncode == 0:
        print(f"\n  {GREEN}[PASS] {name} passed{RESET} ({elapsed:.1f}s)")
    else:
        print(f"\n  {RED}[FAIL] {name} found issues{RESET} ({elapsed:.1f}s)")
        overall_exit = 1

    return result.returncode


# ── 1. Ruff Lint ──
section("Ruff Lint — Code Quality & Style")
ruff_cmd = [sys.executable, "-m", "ruff", "check"]
if FIX_MODE:
    ruff_cmd.append("--fix")
ruff_cmd.extend(SOURCE_DIRS)
run_tool("Ruff lint", ruff_cmd)

# ── 2. Ruff Format Check ──
section("Ruff Format — Code Formatting")
fmt_cmd = [sys.executable, "-m", "ruff", "format", "--check"]
if FIX_MODE:
    fmt_cmd = [sys.executable, "-m", "ruff", "format"]
fmt_cmd.extend(SOURCE_DIRS)
run_tool("Ruff format", fmt_cmd)

# ── 3. Bandit Security Scan ──
section("Bandit — Security Analysis")
bandit_cmd = [
    sys.executable, "-m", "bandit",
    "-r", "engine/", "formats/", "gui/", "main.py",
    "-c", "pyproject.toml",
    "-q",  # Quiet — only show findings
]
if not CI_MODE:
    bandit_cmd.extend(["-ll"])  # Low severity and above
run_tool("Bandit", bandit_cmd)

# ── Summary ──
section("Summary")
if overall_exit == 0:
    print(f"  {GREEN}{BOLD}All checks passed! [PASS]{RESET}\n")
else:
    if FIX_MODE:
        print(f"  {YELLOW}Some issues remain after auto-fix. Review the output above.{RESET}\n")
    else:
        print(f"  {YELLOW}Issues found. Run '{BOLD}python lint.py --fix{RESET}{YELLOW}' to auto-fix what's safe.{RESET}\n")

sys.exit(overall_exit if CI_MODE else 0)
