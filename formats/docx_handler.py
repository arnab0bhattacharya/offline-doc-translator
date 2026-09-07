"""
formats/docx_handler.py
=======================
Word (.docx) format handler.
Processes body text (word/document.xml) along with headers (header*.xml),
footers (footer*.xml), footnotes (footnotes.xml), and endnotes (endnotes.xml).
Emits paragraph-level live progress and telemetry.
"""

import os
import re
import shutil
import tempfile
from typing import Callable, Optional, Dict, Any, List, Tuple

try:
    from .base import BaseFormatHandler
    from ..engine.core import escape_xml, unescape_xml, hash_text, should_translate
except (ImportError, ValueError):
    from formats.base import BaseFormatHandler
    from engine.core import escape_xml, unescape_xml, hash_text, should_translate


class DOCXHandler(BaseFormatHandler):
    """Handles translation of Microsoft Word documents (.docx)."""

    def _count_translatable_paragraphs(self, target_files: List[Tuple[str, str]], direction: str) -> int:
        """Counts total translatable paragraphs across all Word XML parts."""
        total = 0
        p_pattern = re.compile(r"<w:p(?: [^>]+)?>(.*?)</w:p>", re.DOTALL)
        t_pattern = re.compile(r"<w:t(?:\s[^>]*)?>(.*?)</w:t>", re.DOTALL)

        for _, file_path in target_files:
            with open(file_path, "r", encoding="utf-8") as f:
                xml_str = f.read()
            for p_match in p_pattern.finditer(xml_str):
                t_matches = t_pattern.findall(p_match.group(1))
                full_text = unescape_xml("".join(t_matches)).strip()
                if full_text and should_translate(full_text, direction):
                    total += 1
        return total

    def _process_xml_content(
        self,
        xml_str: str,
        part_name: str,
        direction: str,
        review_log_path: Optional[str],
        stats: Dict[str, int],
        progress_state: Dict[str, Any],
        progress_cb: Optional[Callable[[int, int, str], None]],
        log_cb: Optional[Callable[[str], None]]
    ) -> str:
        p_pattern = re.compile(r"(<w:p(?: [^>]+)?>)(.*?)(</w:p>)", re.DOTALL)
        t_pattern = re.compile(r"(<w:t(?:\s[^>]*)?>)(.*?)(</w:t>)", re.DOTALL)

        recent_paragraphs: List[str] = []

        def p_repl(match):
            p_start = match.group(1)
            p_content = match.group(2)
            p_end = match.group(3)

            t_matches = list(t_pattern.finditer(p_content))
            if not t_matches:
                return match.group(0)

            full_text = unescape_xml("".join(m.group(2) for m in t_matches))
            if not full_text.strip():
                return match.group(0)

            stats["total"] += 1

            if not should_translate(full_text, direction):
                stats["skipped"] += 1
                recent_paragraphs.append(full_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)
                return match.group(0)

            progress_state["current"] += 1
            cur_idx = progress_state["current"]
            total_items = progress_state["total"]

            if progress_cb:
                progress_cb(
                    cur_idx,
                    total_items,
                    f"[{cur_idx}/{total_items}] {part_name}: \"{full_text[:20]}..\""
                )

            context_str = " | ".join(recent_paragraphs[-2:]) if recent_paragraphs else None
            chunk_id = hash_text(full_text)[:8]

            translated_text, was_translated, was_reverted = self.engine.translate_chunk(
                text=full_text,
                direction=direction,
                context=context_str,
                location_id=part_name,
                chunk_id=chunk_id,
                review_log_path=review_log_path,
                log_cb=log_cb
            )

            if was_reverted:
                stats["reverted"] += 1
                recent_paragraphs.append(full_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)
                return match.group(0)

            if was_translated:
                stats["translated"] += 1
                escaped_translation = escape_xml(translated_text)

                recent_paragraphs.append(translated_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)

                new_p_content = ""
                last_idx = 0
                for i, m in enumerate(t_matches):
                    new_p_content += p_content[last_idx:m.start()]
                    if i == 0:
                        new_p_content += f'<w:t xml:space="preserve">{escaped_translation}</w:t>'
                    else:
                        new_p_content += '<w:t></w:t>'
                    last_idx = m.end()

                new_p_content += p_content[last_idx:]
                return p_start + new_p_content + p_end

            return match.group(0)

        return p_pattern.sub(p_repl, xml_str)

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
        work_dir = tempfile.mkdtemp(prefix="trans_docx_")

        try:
            if log_cb:
                log_cb(f"[*] Extracting Word document: {os.path.basename(input_path)}...")
            self.extract_zip(input_path, work_dir)

            word_dir = os.path.join(work_dir, "word")
            if not os.path.exists(word_dir):
                self.pack_zip(work_dir, output_path)
                return stats

            target_files = []
            doc_xml = os.path.join(word_dir, "document.xml")
            if os.path.exists(doc_xml):
                target_files.append(("document.xml", doc_xml))

            for f in os.listdir(word_dir):
                if f.endswith(".xml") and (
                    f.startswith("header") or
                    f.startswith("footer") or
                    f.startswith("footnotes") or
                    f.startswith("endnotes")
                ):
                    target_files.append((f, os.path.join(word_dir, f)))

            total_translatable = self._count_translatable_paragraphs(target_files, direction)
            if log_cb:
                log_cb(f"[*] Found {total_translatable} translatable paragraph(s) across {len(target_files)} document part(s).")

            progress_state = {
                "current": 0,
                "total": max(1, total_translatable)
            }

            for label, file_path in target_files:
                with open(file_path, "r", encoding="utf-8") as f:
                    xml_data = f.read()

                processed_xml = self._process_xml_content(
                    xml_str=xml_data,
                    part_name=label,
                    direction=direction,
                    review_log_path=review_log_path,
                    stats=stats,
                    progress_state=progress_state,
                    progress_cb=progress_cb,
                    log_cb=log_cb
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(processed_xml)

                self.engine.save_cache_atomically(log_cb=log_cb)

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Packing translated Word document...")
            if log_cb:
                log_cb(f"[*] Packing translated Word document into {os.path.basename(output_path)}...")

            self.pack_zip(work_dir, output_path)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self.engine.flush_model()

        return stats
