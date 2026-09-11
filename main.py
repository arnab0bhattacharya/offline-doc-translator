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
from typing import Dict, Union

from engine.errors import TranslatorError
from engine.core import TranslationMode, DIRECTIONS
from engine.cache import CachePolicy
from engine.run_job import execute_translation
from engine.logging import TranslationLogger, TranslationLogEvent


MAX_GLOSSARY_ENTRIES: int = 10_000
MAX_TERM_LENGTH: int = 200


def parse_cli_glossary(glossary_arg: str) -> Dict[str, str]:
    """Parses glossary from string or file path with size and term limits."""
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
        if not entry or entry.startswith("#"):
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
        if not k or not v:
            continue
        if len(k) > MAX_TERM_LENGTH or len(v) > MAX_TERM_LENGTH:
            continue
        if len(glossary) >= MAX_GLOSSARY_ENTRIES:
            break
        glossary[k] = v
    return glossary


def run_cli(
    input_path: str,
    output_path: str,
    direction: str,
    mode_str: str,
    model_name: str,
    glossary: Dict[str, str],
    include_source_text: bool = False,
    cache_policy: Union[CachePolicy, str] = CachePolicy.ENCRYPTED_PERSISTENT,
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
    print(f" Cache      : {str(cache_policy).upper()}")
    if glossary:
        print(f" Glossary   : {len(glossary)} active rule(s)")
    print(f"=======================================================")

    mode_enum = mode_str if isinstance(mode_str, TranslationMode) else TranslationMode(mode_str)
    review_log_path = f"{output_path}.needs_review.log"

    pbar = None

    def cli_progress(current, total, msg):
        nonlocal pbar
        if pbar is None:
            pbar = tqdm(total=total, desc="Translating", unit="chunk")
        pbar.total = total
        pbar.n = current
        pbar.set_description(f"{msg[:30]}")
        pbar.refresh()

    cli_logger = TranslationLogger(name="cli")

    def cli_event_handler(event: TranslationLogEvent):
        msg = event.to_cli_string()
        if pbar:
            pbar.write(msg)
        else:
            print(f"  -> {msg}")

    cli_logger.subscribe(cli_event_handler)

    try:
        stats = execute_translation(
            input_path=input_path,
            output_path=output_path,
            direction=direction,
            mode=mode_enum,
            model_name=model_name,
            glossary=glossary,
            progress_cb=cli_progress,
            log_cb=cli_logger.as_log_cb(),
            include_source_text=include_source_text,
            logger=cli_logger,
            cache_policy=cache_policy,
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
    parser.add_argument("--mode", default=TranslationMode.FAST_NMT.value, choices=[m.value for m in TranslationMode], help="Engine mode (default: fast_nmt)")
    parser.add_argument("--model", default="gemma4:e2b-it-qat", help="Ollama model name (default: gemma4:e2b-it-qat)")
    parser.add_argument("--glossary", help="Custom glossary string (e.g. 'Term:Translation') or text file path")
    parser.add_argument("--include-source-text", action="store_true", help="Include original text in review log for debugging (default: false, for privacy)")
    parser.add_argument(
        "--cache-policy",
        default=CachePolicy.ENCRYPTED_PERSISTENT.value,
        choices=[p.value for p in CachePolicy],
        help="Cache storage policy: encrypted_persistent (default), memory_only, or plaintext_persistent",
    )
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

    run_cli(
        input_file,
        output_file,
        direction,
        mode_str,
        model_name,
        glossary,
        include_source_text=args.include_source_text,
        cache_policy=args.cache_policy,
    )



if __name__ == "__main__":
    main()
