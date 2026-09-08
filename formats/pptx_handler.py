"""
formats/pptx_handler.py
=======================
PowerPoint (.pptx) format handler.
Performs surgical XML manipulation on ppt/slides/slide*.xml.
Extracts slide title/header as contextual reference to eliminate pronoun/subject ambiguity.
Emits paragraph-level live progress and telemetry.
"""

import os
import re
import shutil
import tempfile
from typing import Callable, Optional, Dict, Any, List

from formats.base import BaseFormatHandler
from engine.core import escape_xml, unescape_xml, hash_text, should_translate


class PPTXHandler(BaseFormatHandler):
    """Handles translation of PowerPoint presentations (.pptx)."""

    def _extract_slide_title(self, xml_str: str) -> Optional[str]:
        """Heuristically extracts the first non-empty text paragraph (typically the title/heading)."""
        p_pattern = re.compile(r"<a:p(?: [^>]+)?>(.*?)</a:p>", re.DOTALL)
        for p_match in p_pattern.finditer(xml_str):
            full_text, _ = self.extract_paragraph_text_nodes(p_match.group(1), tag_prefix="a")
            full_text = full_text.strip()
            if full_text and len(full_text) > 2:
                return full_text
        return None

    def _count_translatable_paragraphs(self, slides_path: str, direction: str) -> int:
        """Counts total translatable paragraphs across all slides."""
        total = 0
        if not os.path.exists(slides_path):
            return 0

        for f in os.listdir(slides_path):
            if not (f.startswith("slide") and f.endswith(".xml")):
                continue
            slide_file = os.path.join(slides_path, f)
            self.validate_xml_part_size(slide_file)
            with open(slide_file, "r", encoding="utf-8") as file:
                xml_str = file.read()
            total += self.count_translatable_paragraphs_in_xml(xml_str, tag_prefix="a", direction=direction)
        return total

    def _process_slide_xml(
        self,
        xml_str: str,
        slide_name: str,
        direction: str,
        review_log_path: Optional[str],
        stats: Dict[str, int],
        slide_title: Optional[str],
        progress_state: Dict[str, Any],
        progress_cb: Optional[Callable[[int, int, str], None]],
        log_cb: Optional[Callable[[str], None]]
    ) -> str:
        # 1. Protect <a:fld> blocks
        fld_pattern = re.compile(r"<a:fld.*?</a:fld>", re.DOTALL)
        flds: Dict[str, str] = {}

        def fld_repl(m):
            key = f"__FLD_{len(flds)}__"
            flds[key] = m.group(0)
            return key

        xml_str = fld_pattern.sub(fld_repl, xml_str)

        # 2. Extract and translate paragraphs
        p_pattern = re.compile(r"(<a:p(?: [^>]+)?>)(.*?)(</a:p>)", re.DOTALL)

        def p_repl(match):
            p_start = match.group(1)
            p_content = match.group(2)
            p_end = match.group(3)

            full_text, t_matches = self.extract_paragraph_text_nodes(p_content, tag_prefix="a")
            if not full_text.strip():
                return match.group(0)

            # Use slide title as context if it differs from current paragraph
            context = f"Slide Header: {slide_title}" if slide_title and slide_title != full_text else None
            new_p_content = self._translate_and_replace_text_nodes(
                full_text=full_text,
                t_matches=t_matches,
                p_content=p_content,
                direction=direction,
                context=context,
                part_name=slide_name,
                review_log_path=review_log_path,
                stats=stats,
                progress_state=progress_state,
                progress_cb=progress_cb,
                log_cb=log_cb,
                tag_prefix="a",
            )

            if new_p_content is not None:
                return p_start + new_p_content + p_end

            return match.group(0)

        xml_str = p_pattern.sub(p_repl, xml_str)

        # 3. Restore protected <a:fld> blocks
        for key, val in flds.items():
            xml_str = xml_str.replace(key, val)

        return xml_str

    def translate(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        review_log_path: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None,
        log_cb: Optional[Callable[[str], None]] = None
    ) -> Dict[str, Any]:
        self.validate_input_file(input_path)
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        work_dir = tempfile.mkdtemp(prefix="trans_pptx_")

        try:
            if log_cb:
                log_cb(f"[*] Extracting PowerPoint presentation: {os.path.basename(input_path)}...")
            self.extract_zip(input_path, work_dir, policy=self.policy)
            slides_path = os.path.join(work_dir, "ppt", "slides")

            if not os.path.exists(slides_path):
                self.pack_zip(work_dir, output_path)
                return stats

            slide_files = sorted(
                [f for f in os.listdir(slides_path) if f.startswith("slide") and f.endswith(".xml")],
                key=lambda x: int(re.search(r"\d+", x).group(0)) if re.search(r"\d+", x) else 0
            )

            total_translatable = self._count_translatable_paragraphs(slides_path, direction)
            if log_cb:
                log_cb(f"[*] Found {total_translatable} translatable paragraph(s) across {len(slide_files)} slide(s).")

            progress_state = {
                "current": 0,
                "total": max(1, total_translatable)
            }

            for idx, filename in enumerate(slide_files):
                file_path = os.path.join(slides_path, filename)
                self.validate_xml_part_size(file_path)
                with open(file_path, "r", encoding="utf-8") as f:
                    xml_data = f.read()

                slide_title = self._extract_slide_title(xml_data)
                processed_xml = self._process_slide_xml(
                    xml_str=xml_data,
                    slide_name=filename.replace(".xml", ""),
                    direction=direction,
                    review_log_path=review_log_path,
                    stats=stats,
                    slide_title=slide_title,
                    progress_state=progress_state,
                    progress_cb=progress_cb,
                    log_cb=log_cb
                )

                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(processed_xml)

                self.engine.save_cache_atomically(log_cb=log_cb)

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Packing translated PowerPoint file...")
            if log_cb:
                log_cb(f"[*] Packing translated PowerPoint into {os.path.basename(output_path)}...")

            self.pack_zip(work_dir, output_path)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self.engine.flush_model()

        return stats
