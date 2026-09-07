"""
formats/pdf_handler.py
======================
Digital PDF format handler using PyMuPDF (fitz).
Strictly scoped for digitally-born PDFs (vector/native text).
Executes a 3-phase pipeline with live block-by-block progress and telemetry:
  Phase 1: Spatial text-block & coordinate extraction with page-level context.
  Phase 2: Core translation with placeholder masking & bidirectional gates.
  Phase 3: Coordinate redaction with background color sampling and CJK font injection.
"""

import os
import shutil
import tempfile
from typing import Callable, Optional, Dict, Any, List, Tuple

try:
    from .base import BaseFormatHandler
    from ..engine.core import hash_text, should_translate
    from ..engine.errors import ErrorCode, TranslatorError
except (ImportError, ValueError):
    from formats.base import BaseFormatHandler
    from engine.core import hash_text, should_translate
    from engine.errors import ErrorCode, TranslatorError


class PDFHandler(BaseFormatHandler):
    """Handles translation of digitally-born PDF documents."""

    def _sample_background_color(self, page, rect) -> Tuple[float, float, float]:
        """
        Samples the corner pixel near the bounding box to match background color
        (e.g., white, light gray, shaded table row). Defaults to white on failure.
        """
        try:
            sample_rect = page.rect & rect
            if sample_rect.is_empty:
                return (1.0, 1.0, 1.0)
            pix = page.get_pixmap(clip=sample_rect, dpi=72)
            if pix.width > 0 and pix.height > 0:
                pixel = pix.pixel(0, 0)
                return (pixel[0] / 255.0, pixel[1] / 255.0, pixel[2] / 255.0)
        except Exception:
            pass
        return (1.0, 1.0, 1.0)

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

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise TranslatorError(
                ErrorCode.E04,
                detail="PyMuPDF library is not installed. Run 'pip install PyMuPDF'.",
                original_exc=e
            )

        try:
            if log_cb:
                log_cb(f"[*] Opening PDF document: {os.path.basename(input_path)}...")
            doc = fitz.open(input_path)
        except Exception as e:
            raise TranslatorError(
                ErrorCode.E04,
                detail=f"Could not open PDF file '{input_path}'. It may be encrypted or corrupted.",
                original_exc=e
            )

        total_pages = len(doc)
        if total_pages == 0:
            doc.close()
            return stats

        cjk_font = "japan" if direction == "en2ja" else "helv"

        try:
            # 1. Count total translatable blocks
            total_translatable = 0
            all_page_blocks = []

            for page_idx in range(total_pages):
                page = doc[page_idx]
                blocks = page.get_text("blocks")
                text_blocks = [b for b in blocks if len(b) >= 7 and b[6] == 0]
                text_blocks.sort(key=lambda b: (round(b[1] / 10) * 10, b[0]))
                all_page_blocks.append(text_blocks)

                for b in text_blocks:
                    clean_text = b[4].strip()
                    if clean_text and should_translate(clean_text, direction):
                        total_translatable += 1

            if not any(all_page_blocks):
                raise TranslatorError(
                    ErrorCode.E04,
                    detail="The PDF has no selectable text. Scanned PDFs require OCR before translation."
                )

            if log_cb:
                log_cb(f"[*] Found {total_translatable} translatable block(s) across {total_pages} page(s).")

            progress_state = {
                "current": 0,
                "total": max(1, total_translatable)
            }

            # 2. Process page by page
            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_num = page_idx + 1

                page_text_preview = page.get_text("text").strip()
                page_context = page_text_preview[:300] if page_text_preview else None

                text_blocks = all_page_blocks[page_idx]
                translated_blocks = []

                for b_idx, block in enumerate(text_blocks):
                    x0, y0, x1, y1, raw_text, block_no, _ = block
                    clean_text = raw_text.strip()

                    if not clean_text:
                        continue

                    stats["total"] += 1

                    if not should_translate(clean_text, direction):
                        stats["skipped"] += 1
                        continue

                    progress_state["current"] += 1
                    cur_idx = progress_state["current"]
                    total_items = progress_state["total"]

                    if progress_cb:
                        progress_cb(
                            cur_idx,
                            total_items,
                            f"[{cur_idx}/{total_items}] Page {page_num} Block {block_no}: \"{clean_text[:20]}..\""
                        )

                    chunk_id = f"p{page_num}_b{block_no}_{hash_text(clean_text)[:6]}"

                    trans_text, was_translated, was_reverted = self.engine.translate_chunk(
                        text=clean_text,
                        direction=direction,
                        context=page_context,
                        location_id=f"Page {page_num} Block {block_no}",
                        chunk_id=chunk_id,
                        review_log_path=review_log_path,
                        log_cb=log_cb
                    )

                    if was_reverted:
                        stats["reverted"] += 1
                        continue

                    if was_translated:
                        stats["translated"] += 1
                        translated_blocks.append({
                            "rect": fitz.Rect(x0, y0, x1, y1),
                            "text": trans_text,
                            "original": clean_text,
                            "block_no": block_no
                        })

                # Phase 3: Redact & Re-insert
                for item in translated_blocks:
                    rect = item["rect"]
                    trans_text = item["text"]

                    estimated_fontsize = max(8.0, min(14.0, (rect.height / max(1, len(trans_text.splitlines()))) * 0.85))
                    font_floor = max(6.0, estimated_fontsize * 0.70)
                    cur_size = estimated_fontsize
                    fitted = False

                    # Verify fit on a disposable page before removing the source text.
                    # insert_textbox returns a negative value without drawing when it cannot fit.
                    scratch_doc = fitz.open()
                    try:
                        scratch_page = scratch_doc.new_page(width=rect.width, height=rect.height)
                        scratch_rect = fitz.Rect(0, 0, rect.width, rect.height)
                        while cur_size >= font_floor:
                            rc = scratch_page.insert_textbox(
                                scratch_rect,
                                trans_text,
                                fontsize=cur_size,
                                fontname=cjk_font,
                                color=(0, 0, 0),
                                align=fitz.TEXT_ALIGN_LEFT
                            )
                            if rc >= 0:
                                fitted = True
                                break
                            cur_size -= 0.5
                    finally:
                        scratch_doc.close()

                    if not fitted:
                        # Preserve the original block if translation cannot be rendered safely.
                        stats["translated"] -= 1
                        stats["reverted"] += 1
                        if review_log_path:
                            self.engine.log_needs_review(
                                review_log_path,
                                f"PDF Page {page_num}",
                                f"Overflow_b{item['block_no']}",
                                item["original"],
                                hash_text(item["original"])
                            )
                        continue

                    bg_color = self._sample_background_color(page, rect)
                    page.add_redact_annot(rect, fill=bg_color)
                    page.apply_redactions()
                    page.insert_textbox(
                        rect,
                        trans_text,
                        fontsize=cur_size,
                        fontname=cjk_font,
                        color=(0, 0, 0),
                        align=fitz.TEXT_ALIGN_LEFT
                    )

                self.engine.save_cache_atomically(log_cb=log_cb)

            if progress_cb:
                progress_cb(progress_state["total"], progress_state["total"], "Saving translated PDF document...")
            if log_cb:
                log_cb(f"[*] Saving translated PDF into {os.path.basename(output_path)}...")

            out_dir = os.path.dirname(os.path.abspath(output_path))
            os.makedirs(out_dir, exist_ok=True)
            fd, temp_output = tempfile.mkstemp(prefix=".translated_", suffix=".pdf", dir=out_dir)
            os.close(fd)
            os.remove(temp_output)
            try:
                doc.save(temp_output, garbage=4, deflate=True)
                validation_doc = fitz.open(temp_output)
                validation_doc.close()
                os.replace(temp_output, output_path)
            finally:
                if os.path.exists(temp_output):
                    os.remove(temp_output)

        finally:
            doc.close()
            self.engine.flush_model()

        return stats
