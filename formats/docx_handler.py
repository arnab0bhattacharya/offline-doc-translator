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
import threading
from typing import Callable, Optional, Dict, Any, List, Tuple

from formats.base import BaseFormatHandler, XML_TAG_ATTRS
from formats.xml_utils import parse_xml_safely
from engine.core import escape_xml, unescape_xml, hash_text, should_translate
from engine.errors import ErrorCode, TranslatorError


class DOCXHandler(BaseFormatHandler):
    """Handles translation of Microsoft Word documents (.docx)."""

    def _count_translatable_paragraphs(self, target_files: List[Tuple[str, str]], direction: str) -> int:
        """Counts total translatable paragraphs across all Word XML parts."""
        total = 0
        for _, file_path in target_files:
            self.validate_xml_part_size(file_path)
            with open(file_path, "r", encoding="utf-8") as f:
                xml_str = f.read()
            total += self.count_translatable_paragraphs_in_xml(xml_str, tag_prefix="w", direction=direction)
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
        log_cb: Optional[Callable[[str], None]],
        cancel_event: Optional[threading.Event] = None,
    ) -> str:
        p_pattern = re.compile(rf"(<w:p\b{XML_TAG_ATTRS}>)(.*?)(</w:p>)", re.DOTALL)
        recent_paragraphs: List[str] = []

        def p_repl(match):
            if cancel_event and cancel_event.is_set():
                raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

            p_start = match.group(1)
            p_content = match.group(2)
            p_end = match.group(3)

            full_text, t_matches = self.extract_paragraph_text_nodes(p_content, tag_prefix="w")
            if not full_text.strip():
                return match.group(0)

            context_str = " | ".join(recent_paragraphs[-2:]) if recent_paragraphs else None
            new_p_content = self._translate_and_replace_text_nodes(
                full_text=full_text,
                t_matches=t_matches,
                p_content=p_content,
                direction=direction,
                context=context_str,
                part_name=part_name,
                review_log_path=review_log_path,
                stats=stats,
                progress_state=progress_state,
                progress_cb=progress_cb,
                log_cb=log_cb,
                tag_prefix="w",
                recent_paragraphs=recent_paragraphs,
                cancel_event=cancel_event,
            )

            if new_p_content is not None:
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
        log_cb: Optional[Callable[[str], None]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Dict[str, Any]:
        self.validate_input_file(input_path)
        if cancel_event and cancel_event.is_set():
            raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        work_dir = tempfile.mkdtemp(prefix="trans_docx_")

        try:
            if log_cb:
                log_cb(f"[*] Extracting Word document: {os.path.basename(input_path)}...")
            self.extract_zip(input_path, work_dir, policy=self.policy)

            if cancel_event and cancel_event.is_set():
                raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

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
                if cancel_event and cancel_event.is_set():
                    raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")
                self.validate_xml_part_size(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    xml_data = f.read()

                # Security: check for XXE / prohibited entity or DTD declarations
                try:
                    parse_xml_safely(xml_data)
                except TranslatorError:
                    raise
                except Exception:
                    pass

                processed_xml = self._process_xml_content(
                    xml_str=xml_data,
                    part_name=label,
                    direction=direction,
                    review_log_path=review_log_path,
                    stats=stats,
                    progress_state=progress_state,
                    progress_cb=progress_cb,
                    log_cb=log_cb,
                    cancel_event=cancel_event,
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(processed_xml)

                self.engine.save_cache_atomically(log_cb=log_cb)

            if cancel_event and cancel_event.is_set():
                raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Packing translated Word document...")
            if log_cb:
                log_cb(f"[*] Packing translated Word document into {os.path.basename(output_path)}...")

            self.pack_zip(work_dir, output_path)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self.engine.flush_model()

        return stats
