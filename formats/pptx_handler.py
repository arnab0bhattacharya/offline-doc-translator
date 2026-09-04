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

try:
    from .base import BaseFormatHandler
    from ..engine.core import escape_xml, unescape_xml, hash_text, should_translate
except (ImportError, ValueError):
    from formats.base import BaseFormatHandler
    from engine.core import escape_xml, unescape_xml, hash_text, should_translate


class PPTXHandler(BaseFormatHandler):
    """Handles translation of PowerPoint presentations (.pptx)."""

    def _extract_slide_title(self, xml_str: str) -> Optional[str]:
        """Heuristically extracts the first non-empty text paragraph (typically the title/heading)."""
        p_pattern = re.compile(r"<a:p(?: [^>]+)?>(.*?)</a:p>", re.DOTALL)
        t_pattern = re.compile(r"<a:t(?:\s[^>]*)?>(.*?)</a:t>", re.DOTALL)
        
        for p_match in p_pattern.finditer(xml_str):
            p_content = p_match.group(1)
            t_matches = t_pattern.findall(p_content)
            full_text = unescape_xml("".join(t_matches)).strip()
            if full_text and len(full_text) > 2:
                return full_text
        return None

    def _count_translatable_paragraphs(self, slides_path: str, direction: str) -> int:
        """Counts total translatable paragraphs across all slides."""
        total = 0
        p_pattern = re.compile(r"<a:p(?: [^>]+)?>(.*?)</a:p>", re.DOTALL)
        t_pattern = re.compile(r"<a:t(?:\s[^>]*)?>(.*?)</a:t>", re.DOTALL)

        if not os.path.exists(slides_path):
            return 0

        for f in os.listdir(slides_path):
            if not (f.startswith("slide") and f.endswith(".xml")):
                continue
            with open(os.path.join(slides_path, f), "r", encoding="utf-8") as file:
                xml_str = file.read()
            for p_match in p_pattern.finditer(xml_str):
                t_matches = t_pattern.findall(p_match.group(1))
                full_text = unescape_xml("".join(t_matches)).strip()
                if full_text and should_translate(full_text, direction):
                    total += 1
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

            t_pattern = re.compile(r"(<a:t(?:\s[^>]*)?>)(.*?)(</a:t>)", re.DOTALL)
            t_matches = list(t_pattern.finditer(p_content))

            if not t_matches:
                return match.group(0)

            full_text = unescape_xml("".join(m.group(2) for m in t_matches))
            if not full_text.strip():
                return match.group(0)

            stats["total"] += 1

            if not should_translate(full_text, direction):
                stats["skipped"] += 1
                return match.group(0)

            progress_state["current"] += 1
            cur_idx = progress_state["current"]
            total_items = progress_state["total"]

            if progress_cb:
                progress_cb(
                    cur_idx,
                    total_items,
                    f"[{cur_idx}/{total_items}] {slide_name}: \"{full_text[:20]}..\""
                )

            # Use slide title as context if it differs from current paragraph
            context = f"Slide Header: {slide_title}" if slide_title and slide_title != full_text else None
            chunk_id = hash_text(full_text)[:8]

            translated_text, was_translated, was_reverted = self.engine.translate_chunk(
                text=full_text,
                direction=direction,
                context=context,
                location_id=f"{slide_name}",
                chunk_id=chunk_id,
                review_log_path=review_log_path,
                log_cb=log_cb
            )

            if was_reverted:
                stats["reverted"] += 1
                return match.group(0)

            if was_translated:
                stats["translated"] += 1
                escaped_translation = escape_xml(translated_text)

                new_p_content = ""
                last_idx = 0
                for i, m in enumerate(t_matches):
                    new_p_content += p_content[last_idx:m.start()]
                    if i == 0:
                        new_p_content += f'<a:t xml:space="preserve">{escaped_translation}</a:t>'
                    else:
                        new_p_content += '<a:t></a:t>'
                    last_idx = m.end()

                new_p_content += p_content[last_idx:]
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
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        work_dir = tempfile.mkdtemp(prefix="trans_pptx_")

        try:
            if log_cb:
                log_cb(f"[*] Extracting PowerPoint presentation: {os.path.basename(input_path)}...")
            self.extract_zip(input_path, work_dir)
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

                self.engine.save_cache_atomically()

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Packing translated PowerPoint file...")
            if log_cb:
                log_cb(f"[*] Packing translated PowerPoint into {os.path.basename(output_path)}...")

            self.pack_zip(work_dir, output_path)

        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            self.engine.flush_model()

        return stats
