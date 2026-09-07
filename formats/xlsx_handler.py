"""
formats/xlsx_handler.py
=======================
Excel (.xlsx) format handler.
Processes worksheets row-by-row with granular cell-level live progress and logging.
Injects translated text using native ECMA-376 inline strings (<c t="inlineStr">),
preserving cell styling attributes (s="...") while completely safeguarding formulas (<f>)
and numerical values (<v>) by construction.
"""

import os
import re
import shutil
import tempfile
from typing import Callable, Optional, Dict, Any, List, Tuple

from formats.base import BaseFormatHandler
from engine.core import escape_xml, unescape_xml, hash_text, should_translate


class XLSXHandler(BaseFormatHandler):
    """Handles translation of Excel workbooks (.xlsx)."""

    def _parse_shared_strings(self, shared_strings_path: str) -> List[str]:
        """Loads all strings from xl/sharedStrings.xml into a 0-indexed list."""
        if not os.path.exists(shared_strings_path):
            return []

        with open(shared_strings_path, "r", encoding="utf-8") as f:
            xml_data = f.read()

        si_pattern = re.compile(r"<si>(.*?)</si>", re.DOTALL)
        t_pattern = re.compile(r"<t(?:\s[^>]*)?>(.*?)</t>", re.DOTALL)

        strings = []
        for si_match in si_pattern.finditer(xml_data):
            si_content = si_match.group(1)
            t_pieces = t_pattern.findall(si_content)
            strings.append(unescape_xml("".join(t_pieces)))

        return strings

    def _count_translatable_cells(
        self,
        sheets_dir: str,
        shared_strings: List[str],
        direction: str
    ) -> int:
        """Fast pre-scan to calculate the exact total number of translatable text cells."""
        total_count = 0
        cell_pattern = re.compile(r"<c\s+([^>]*?)(?:>(.*?)</c>|/>)", re.DOTALL)

        for filename in os.listdir(sheets_dir):
            if not (filename.startswith("sheet") and filename.endswith(".xml")):
                continue
            sheet_path = os.path.join(sheets_dir, filename)
            with open(sheet_path, "r", encoding="utf-8") as f:
                xml_data = f.read()

            for c_match in cell_pattern.finditer(xml_data):
                attrs = c_match.group(1)
                body = c_match.group(2) or ""
                if "<f" in body:
                    continue

                t_match = re.search(r't="([a-zA-Z0-9]+)"', attrs)
                cell_type = t_match.group(1) if t_match else ""

                cell_text = ""
                if cell_type == "s":
                    v_match = re.search(r"<v>(\d+)</v>", body)
                    if v_match:
                        s_idx = int(v_match.group(1))
                        if 0 <= s_idx < len(shared_strings):
                            cell_text = shared_strings[s_idx]
                elif cell_type == "inlineStr":
                    t_matches = re.findall(r"<t(?:\s[^>]*)?>(.*?)</t>", body, re.DOTALL)
                    cell_text = unescape_xml("".join(t_matches))

                if cell_text.strip() and should_translate(cell_text, direction):
                    total_count += 1

        return total_count

    def _process_sheet_xml(
        self,
        sheet_xml: str,
        sheet_name: str,
        shared_strings: List[str],
        direction: str,
        review_log_path: Optional[str],
        stats: Dict[str, int],
        progress_state: Dict[str, Any],
        progress_cb: Optional[Callable[[int, int, str], None]],
        log_cb: Optional[Callable[[str], None]]
    ) -> str:
        row_pattern = re.compile(r"(<row(?:\s[^>]*)?>)(.*?)(</row>)", re.DOTALL)
        cell_pattern = re.compile(r"<c\s+([^>]*?)(?:>(.*?)</c>|/>)", re.DOTALL)

        def row_repl(row_match):
            row_start = row_match.group(1)
            row_content = row_match.group(2)
            row_end = row_match.group(3)

            # First pass over the row: gather text for row context
            row_texts = []
            cells_data = []

            for c_match in cell_pattern.finditer(row_content):
                attrs = c_match.group(1)
                body = c_match.group(2) or ""

                r_match = re.search(r'r="([A-Z0-9]+)"', attrs)
                cell_ref = r_match.group(1) if r_match else ""

                t_match = re.search(r't="([a-zA-Z0-9]+)"', attrs)
                cell_type = t_match.group(1) if t_match else ""

                s_match = re.search(r's="(\d+)"', attrs)
                style_attr = f' s="{s_match.group(1)}"' if s_match else ""

                has_formula = "<f" in body
                cell_text = ""
                is_text_cell = False

                if not has_formula:
                    if cell_type == "s":
                        v_match = re.search(r"<v>(\d+)</v>", body)
                        if v_match:
                            s_idx = int(v_match.group(1))
                            if 0 <= s_idx < len(shared_strings):
                                cell_text = shared_strings[s_idx]
                                is_text_cell = True
                    elif cell_type == "inlineStr":
                        t_matches = re.findall(r"<t(?:\s[^>]*)?>(.*?)</t>", body, re.DOTALL)
                        cell_text = unescape_xml("".join(t_matches))
                        is_text_cell = True

                if is_text_cell and cell_text.strip():
                    row_texts.append(f"{cell_ref}: {cell_text.strip()}")

                cells_data.append({
                    "full_match": c_match.group(0),
                    "start": c_match.start(),
                    "end": c_match.end(),
                    "ref": cell_ref,
                    "type": cell_type,
                    "style_attr": style_attr,
                    "text": cell_text,
                    "is_text": is_text_cell
                })

            row_context_str = " | ".join(row_texts[:8]) if row_texts else None

            # Second pass: translate and replace text cells
            new_row_content = ""
            last_idx = 0

            for cell in cells_data:
                new_row_content += row_content[last_idx:cell["start"]]

                if cell["is_text"] and cell["text"].strip():
                    raw_text = cell["text"]
                    stats["total"] += 1

                    if should_translate(raw_text, direction):
                        progress_state["current"] += 1
                        cur_idx = progress_state["current"]
                        total_items = progress_state["total"]

                        if progress_cb:
                            progress_cb(
                                cur_idx,
                                total_items,
                                f"[{cur_idx}/{total_items}] {sheet_name}!{cell['ref']}: \"{raw_text[:20]}..\""
                            )

                        context = f"Sheet: {sheet_name} | Row context: {row_context_str}" if row_context_str else f"Sheet: {sheet_name}"
                        chunk_id = f"{cell['ref']}_{hash_text(raw_text)[:6]}"

                        translated_text, was_translated, was_reverted = self.engine.translate_chunk(
                            text=raw_text,
                            direction=direction,
                            context=context,
                            location_id=f"{sheet_name}!{cell['ref']}",
                            chunk_id=chunk_id,
                            review_log_path=review_log_path,
                            log_cb=log_cb
                        )

                        if was_translated:
                            stats["translated"] += 1
                            escaped_text = escape_xml(translated_text)
                            new_cell_xml = (
                                f'<c r="{cell["ref"]}" t="inlineStr"{cell["style_attr"]}>'
                                f'<is><t xml:space="preserve">{escaped_text}</t></is>'
                                f'</c>'
                            )
                            new_row_content += new_cell_xml
                            last_idx = cell["end"]
                            continue
                        elif was_reverted:
                            stats["reverted"] += 1
                    else:
                        stats["skipped"] += 1

                new_row_content += cell["full_match"]
                last_idx = cell["end"]

            new_row_content += row_content[last_idx:]
            return row_start + new_row_content + row_end

        return row_pattern.sub(row_repl, sheet_xml)

    def translate(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        review_log_path: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        work_dir = tempfile.mkdtemp(prefix="trans_xlsx_")

        try:
            if log_cb:
                log_cb(f"[*] Extracting Excel archive: {os.path.basename(input_path)}...")
            self.extract_zip(input_path, work_dir)

            shared_strings_file = os.path.join(work_dir, "xl", "sharedStrings.xml")
            shared_strings = self._parse_shared_strings(shared_strings_file)
            if log_cb:
                log_cb(f"[*] Loaded {len(shared_strings)} shared string entries.")

            sheets_dir = os.path.join(work_dir, "xl", "worksheets")
            if not os.path.exists(sheets_dir):
                self.pack_zip(work_dir, output_path)
                return stats

            sheet_files = sorted([
                f for f in os.listdir(sheets_dir)
                if f.startswith("sheet") and f.endswith(".xml")
            ])

            if log_cb:
                log_cb(f"[*] Scanning {len(sheet_files)} worksheet(s) for translatable cells...")

            # 1. Pre-scan for exact translatable count
            total_translatable = self._count_translatable_cells(sheets_dir, shared_strings, direction)
            if log_cb:
                log_cb(f"[*] Found {total_translatable} cell(s) matching translation direction '{direction}'.")

            progress_state = {
                "current": 0,
                "total": max(1, total_translatable)
            }

            # 2. Process worksheets with live cell updates
            for filename in sheet_files:
                sheet_path = os.path.join(sheets_dir, filename)
                sheet_label = filename.replace(".xml", "")

                with open(sheet_path, "r", encoding="utf-8") as f:
                    xml_data = f.read()

                processed_xml = self._process_sheet_xml(
                    sheet_xml=xml_data,
                    sheet_name=sheet_label,
                    shared_strings=shared_strings,
                    direction=direction,
                    review_log_path=review_log_path,
                    stats=stats,
                    progress_state=progress_state,
                    progress_cb=progress_cb,
                    log_cb=log_cb
                )

                with open(sheet_path, "w", encoding="utf-8") as f:
                    f.write(processed_xml)

                self.engine.save_cache_atomically(log_cb=log_cb)

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Packing translated Excel file...")
            if log_cb:
                log_cb(f"[*] Packing translated Excel workbook into {os.path.basename(output_path)}...")

            self.pack_zip(work_dir, output_path)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self.engine.flush_model()

        return stats
