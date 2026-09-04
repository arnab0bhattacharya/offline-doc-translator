"""
main.py
=======
Unified entry point for the Offline Document Translation Suite.
Supports Fast NMT Mode, and Pure LLM Mode.

Usage:
  1. Desktop GUI mode (default if run without arguments):
     python main.py

  2. Command Line Interface (CLI):
     python main.py --input presentation.pptx --direction ja2en --mode fast_nmt
     python main.py --input sheet.xlsx --direction en2ja --mode pure_llm
     python main.py --input document.docx --direction ja2en --glossary "リード:Sales Leads, 当四半期:Q3 Period"
"""

import sys
import os

# Ensure project root is first in sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
from tqdm import tqdm
from typing import Dict

from engine.errors import TranslatorError
from engine.preflight import run_preflight, run_nmt_preflight
from engine.core import TranslationEngine, TranslationMode, DIRECTIONS
from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler
from formats.docx_handler import DOCXHandler
from formats.pdf_handler import PDFHandler


def parse_cli_glossary(glossary_arg: str) -> Dict[str, str]:
    """Parses glossary from string or file path."""
    if not glossary_arg:
        return {}
    if os.path.exists(glossary_arg):
        with open(glossary_arg, "r", encoding="utf-8") as f:
            content = f.read()
    else:
        content = glossary_arg

    glossary = {}
    for entry in content.replace("\n", ",").split(","):
        entry = entry.strip()
        if not entry:
            continue
        if "->" in entry:
            parts = entry.split("->", 1)
        elif ":" in entry:
            parts = entry.split(":", 1)
        elif "=" in entry:
            parts = entry.split("=", 1)
        else:
            continue
        k = parts[0].strip()
        v = parts[1].strip()
        if k and v:
            glossary[k] = v
    return glossary


def run_cli(
    input_path: str,
    output_path: str,
    direction: str,
    mode_str: str,
    model_name: str,
    glossary: Dict[str, str]
) -> None:
    """Executes translation in terminal with tqdm progress bar and live telemetry."""
    print(f"\n=======================================================")
    print(f" Offline Document Translator (CLI Mode)")
    print(f"=======================================================")
    print(f" Input File : {input_path}")
    print(f" Output File: {output_path}")
    print(f" Mode       : {mode_str.upper()}")
    print(f" Direction  : {direction}")
    print(f" Model      : {model_name}")
    if glossary:
        print(f" Glossary   : {len(glossary)} active rule(s)")
    print(f"=======================================================\n")

    review_log_path = f"{output_path}.needs_review.log"
    if os.path.exists(review_log_path):
        try:
            os.remove(review_log_path)
        except Exception:
            pass

    # 1. Run Preflight. Local file/disk validation always applies; Ollama is
    # required only for Pure LLM mode, while Fast NMT requires Argos packages.
    print("[1/3] Running system diagnostics & preflight...")
    llm_available = mode_str == "pure_llm"
    try:
        run_preflight(
            model_name=model_name,
            input_path=input_path,
            output_path=output_path,
            check_model=(mode_str == "pure_llm"),
            require_ollama=(mode_str == "pure_llm")
        )
        if mode_str == "fast_nmt":
            run_nmt_preflight(direction)
        print("  -> Preflight checks passed successfully.")
    except TranslatorError as err:
        print(f"\n[!] PREFLIGHT FAILED: [{err.code.value}] {err.title}")
        print(f"    Message: {err.user_message}")
        print(f"    Action : {err.action}")
        sys.exit(1)

    # 2. Initialize Engine & Cache
    mode_enum = TranslationMode(mode_str)
    print(f"[2/3] Initializing {mode_str.upper()} engine & persistent cache...")
    engine = TranslationEngine(
        model_name=model_name,
        mode=mode_enum,
        glossary=glossary,
        cache_file=os.path.join(os.path.dirname(os.path.abspath(output_path)), ".translation_cache.json"),
        allow_llm=llm_available
    )
    engine.load_cache(direction)

    # 3. Dispatch Format Handler
    ext = os.path.splitext(input_path)[1].lower()
    if ext == ".pptx":
        handler = PPTXHandler(engine)
    elif ext == ".xlsx":
        handler = XLSXHandler(engine)
    elif ext == ".docx":
        handler = DOCXHandler(engine)
    elif ext == ".pdf":
        handler = PDFHandler(engine)
    else:
        print(f"[!] Error: Unsupported file extension '{ext}'.")
        sys.exit(1)

    print(f"[3/3] Translating {ext.upper()} document...")

    pbar = None

    def cli_progress(current, total, msg):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Translating", unit="chunk")
        pbar.total = total
        pbar.n = current
        pbar.set_description(f"{msg[:30]}")
        pbar.refresh()

    def cli_log(msg: str):
        if pbar:
            pbar.write(msg)
        else:
            print(msg)

    try:
        stats = handler.translate(
            input_path=input_path,
            output_path=output_path,
            direction=direction,
            review_log_path=review_log_path,
            progress_cb=cli_progress,
            log_cb=cli_log
        )
        if pbar:
            pbar.close()

        print("\n=======================================================")
        print(f" Translation Complete! [{mode_str.upper()} Mode]")
        print("=======================================================")
        print(f" Translated Chunks : {stats.get('translated', 0)}")
        print(f" Reverted / Review : {stats.get('reverted', 0)}")
        print(f" Skipped (Untrans) : {stats.get('skipped', 0)}")
        print(f" Total Evaluated   : {stats.get('total', 0)}")
        print(f" Output Saved to   : {output_path}")

        if os.path.exists(review_log_path) and os.path.getsize(review_log_path) > 0:
            print(f"\n[!] Notice [E06]: Some items failed validation and were kept in original language.")
            print(f"    See '{review_log_path}' for the full audit log.")
        print("=======================================================\n")

    except TranslatorError as terr:
        if pbar:
            pbar.close()
        print(f"\n[!] TRANSLATION FAILED: [{terr.code.value}] {terr.title}")
        print(f"    Message: {terr.user_message}")
        print(f"    Action : {terr.action}")
        sys.exit(1)
    except Exception as e:
        if pbar:
            pbar.close()
        print(f"\n[!] Unexpected Error: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Offline Document Translator (PPTX, XLSX, DOCX, PDF) powered by Local NMT + Local LLMs."
    )
    parser.add_argument("--input", help="Source document path (.pptx, .xlsx, .docx, .pdf)")
    parser.add_argument("--output", help="Output path. Defaults to '<input>_<direction>.<ext>'")
    parser.add_argument("--direction", choices=DIRECTIONS, help="Translation direction: ja2en or en2ja")
    parser.add_argument("--mode", default="fast_nmt", choices=["fast_nmt", "pure_llm"], help="Engine mode (default: fast_nmt)")
    parser.add_argument("--model", default="gemma4:e2b-it-qat", help="Ollama model name (default: gemma4:e2b-it-qat)")
    parser.add_argument("--glossary", help="Custom glossary string (e.g. 'Term:Translation') or text file path")
    parser.add_argument("--gui", action="store_true", help="Force launch Desktop GUI")

    args = parser.parse_args()

    if args.gui or len(sys.argv) == 1:
        from gui.app import launch_gui
        launch_gui()
        return

    if not args.input:
        parser.error("--input is required in CLI mode. Run without arguments to launch the GUI.")
    if not args.direction:
        parser.error("--direction is required (ja2en or en2ja).")

    input_file = args.input
    direction = args.direction
    mode_str = args.mode
    model_name = args.model
    glossary = parse_cli_glossary(args.glossary)

    if args.output:
        output_file = args.output
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_{direction}{ext}"

    run_cli(input_file, output_file, direction, mode_str, model_name, glossary)


if __name__ == "__main__":
    main()
