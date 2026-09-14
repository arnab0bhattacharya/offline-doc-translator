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
import re
import tempfile
import threading
from collections.abc import Callable
from typing import Any

from engine.core import TranslationResult, hash_text, should_translate
from engine.errors import ErrorCode, TranslatorError
from formats.base import BaseFormatHandler

HAS_CJK = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FAF]")


class PDFHandler(BaseFormatHandler):
    """Handles translation of digitally-born PDF documents."""

    def _sample_background_color(self, page, rect) -> tuple[float, float, float]:
        """
        Samples background color using the median of 4 corner pixels and the center pixel
        from a margin-expanded clip. This rejects outlier pixels (border lines, ink strokes)
        to accurately match shaded table rows, tinted callouts, or white pages.
        Defaults to white (1.0, 1.0, 1.0) on failure.
        """
        try:
            import fitz

            margin = 2.0  # Sample slightly outside the text area
            rx0 = getattr(rect, "x0", rect[0])
            ry0 = getattr(rect, "y0", rect[1])
            rx1 = getattr(rect, "x1", rect[2])
            ry1 = getattr(rect, "y1", rect[3])

            px0 = page.rect.x0
            py0 = page.rect.y0
            px1 = page.rect.x1
            py1 = page.rect.y1

            sample_rect = (
                fitz.Rect(
                    max(px0, rx0 - margin),
                    max(py0, ry0 - margin),
                    min(px1, rx1 + margin),
                    min(py1, ry1 + margin),
                )
                & page.rect
            )

            if sample_rect.is_empty:
                return (1.0, 1.0, 1.0)

            pix = page.get_pixmap(clip=sample_rect, dpi=72)
            if pix.width < 2 or pix.height < 2:
                return (1.0, 1.0, 1.0)

            # Sample 4 corners + center
            points = [
                (0, 0),
                (pix.width - 1, 0),
                (0, pix.height - 1),
                (pix.width - 1, pix.height - 1),
                (pix.width // 2, pix.height // 2),
            ]

            samples = []
            for x, y in points:
                p = pix.pixel(x, y)
                if len(p) == 1:
                    samples.append((p[0], p[0], p[0]))
                else:
                    samples.append((p[0], p[1], p[2]))

            # Use median per channel to reject outliers (border lines, ink)
            mid = len(samples) // 2
            r = sorted(s[0] for s in samples)[mid]
            g = sorted(s[1] for s in samples)[mid]
            b = sorted(s[2] for s in samples)[mid]
            return (r / 255.0, g / 255.0, b / 255.0)
        except Exception:
            return (1.0, 1.0, 1.0)

    @staticmethod
    def _color_to_rgb(color_val: Any) -> tuple[float, float, float]:
        """Convert PyMuPDF integer color (0xRRGGBB) to (r, g, b) float tuple in [0.0, 1.0]."""
        try:
            c = int(color_val)
            if c <= 0:
                return (0.0, 0.0, 0.0)
            r = ((c >> 16) & 0xFF) / 255.0
            g = ((c >> 8) & 0xFF) / 255.0
            b = (c & 0xFF) / 255.0
            return (r, g, b)
        except Exception:
            return (0.0, 0.0, 0.0)

    def translate(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        review_log_path: str | None = None,
        progress_cb: Callable[[int, int, str], None] | None = None,
        log_cb: Callable[[str], None] | None = None,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        self.validate_input_file(input_path)
        if cancel_event and cancel_event.is_set():
            raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")
        stats = {
            "total": 0,
            "translated": 0,
            "reverted": 0,
            "skipped": 0,
            "pages_with_text": 0,
            "total_pages": 0,
        }

        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise TranslatorError(
                ErrorCode.E04,
                detail="PyMuPDF library is not installed. Run 'pip install PyMuPDF'.",
                original_exc=e,
            ) from e

        try:
            if log_cb:
                log_cb(f"[*] Opening PDF document: {os.path.basename(input_path)}...")
            doc = fitz.open(input_path)
        except Exception as e:
            raise TranslatorError(
                ErrorCode.E04,
                detail=f"Could not open PDF file '{input_path}'. It may be encrypted or corrupted.",
                original_exc=e,
            ) from e

        total_pages = len(doc)
        stats["total_pages"] = total_pages
        if total_pages == 0:
            doc.close()
            return stats

        if total_pages > self.policy.max_pdf_pages:
            doc.close()
            raise TranslatorError(
                ErrorCode.E04,
                detail=(
                    f"PDF '{os.path.basename(input_path)}' ({total_pages} pages) "
                    f"exceeds maximum page limit ({self.policy.max_pdf_pages} pages)."
                ),
            )

        try:
            # 1. Count total translatable blocks via get_text("dict")
            total_translatable = 0
            all_page_blocks = []

            for page_idx in range(total_pages):
                page = doc[page_idx]
                page_dict = page.get_text("dict")
                text_blocks = []

                for block_idx, block in enumerate(page_dict.get("blocks", [])):
                    if block.get("type") != 0:
                        continue

                    bbox = block.get("bbox")
                    if not bbox or len(bbox) < 4:
                        continue

                    lines = block.get("lines", [])
                    spans = []
                    lines_text = []
                    for line in lines:
                        line_spans = line.get("spans", [])
                        spans.extend(line_spans)
                        line_str = "".join(s.get("text", "") for s in line_spans)
                        lines_text.append(line_str)

                    raw_text = "\n".join(lines_text)
                    clean_text = raw_text.strip()

                    # Dominant font size, font family, and text color
                    non_empty_spans = [s for s in spans if s.get("text", "").strip()]
                    if non_empty_spans:
                        size_counts: dict[float, int] = {}
                        font_counts: dict[str, int] = {}
                        color_counts: dict[int, int] = {}
                        for s in non_empty_spans:
                            ch_len = len(s.get("text", "").strip())
                            sz = round(float(s.get("size", 12.0)), 1)
                            fn = s.get("font", "helv")
                            cl = s.get("color", 0)

                            size_counts[sz] = size_counts.get(sz, 0) + ch_len
                            font_counts[fn] = font_counts.get(fn, 0) + ch_len
                            color_counts[cl] = color_counts.get(cl, 0) + ch_len

                        dominant_size = max(size_counts.keys(), key=lambda k: size_counts[k])
                        dominant_font = max(font_counts.keys(), key=lambda k: font_counts[k])
                        dominant_color = max(color_counts.keys(), key=lambda k: color_counts[k])
                    else:
                        dominant_size = 12.0
                        dominant_font = "helv"
                        dominant_color = 0

                    block_no = block.get("number", block_idx)
                    text_blocks.append(
                        {
                            "bbox": bbox,
                            "rect": fitz.Rect(bbox),
                            "text": clean_text,
                            "raw_text": raw_text,
                            "block_no": block_no,
                            "original_fontsize": dominant_size,
                            "dominant_font": dominant_font,
                            "dominant_color": dominant_color,
                            "spans": spans,
                        }
                    )

                text_blocks.sort(key=lambda b: (round(b["bbox"][1] / 10) * 10, b["bbox"][0]))
                all_page_blocks.append(text_blocks)

                for b in text_blocks:
                    if b["text"] and should_translate(b["text"], direction):
                        total_translatable += 1

            pages_with_text = sum(1 for page_blocks in all_page_blocks if any(b["text"] for b in page_blocks))
            text_coverage_ratio = pages_with_text / total_pages if total_pages > 0 else 0.0

            stats["pages_with_text"] = pages_with_text
            stats["total_pages"] = total_pages

            if text_coverage_ratio == 0.0:
                raise TranslatorError(
                    ErrorCode.E04,
                    detail=(
                        "The PDF has no selectable text — it appears to be a scanned document. "
                        "To translate scanned PDFs, first run OCR using Adobe Acrobat's 'Recognize Text' feature, "
                        "or free tools like NAPS2 (naps2.com) or ocrmypdf (pip install ocrmypdf), "
                        "then re-open the OCR'd PDF here."
                    ),
                )

            if text_coverage_ratio < 0.5 and log_cb:
                log_cb(
                    f"⚠ Warning: Only {pages_with_text}/{total_pages} pages "
                    f"({text_coverage_ratio:.0%}) have selectable text. "
                    f"Remaining pages may be scanned images and will be skipped."
                )

            if log_cb:
                log_cb(
                    f"[*] Found {total_translatable} translatable block(s) across "
                    f"{pages_with_text}/{total_pages} page(s)."
                )

            progress_state = {"current": 0, "total": max(1, total_translatable)}

            # 2. Process page by page
            for page_idx in range(total_pages):
                if cancel_event and cancel_event.is_set():
                    raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                page = doc[page_idx]
                page_num = page_idx + 1

                text_blocks = all_page_blocks[page_idx]
                if not text_blocks or not any(b["text"] for b in text_blocks):
                    if log_cb:
                        log_cb(f"  [!] Page {page_num}: No selectable text (possibly scanned). Skipping.")
                    continue

                page_text_preview = page.get_text("text").strip()
                page_context = page_text_preview[:300] if page_text_preview else None

                translated_blocks = []

                for b_idx, block_info in enumerate(text_blocks):
                    if cancel_event and cancel_event.is_set():
                        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                    clean_text = block_info["text"]
                    block_no = block_info["block_no"]
                    rect = block_info["rect"]
                    original_fontsize = block_info["original_fontsize"]
                    dominant_font = block_info["dominant_font"]
                    dominant_color = block_info["dominant_color"]

                    if not clean_text:
                        continue

                    stats["total"] += 1

                    if not should_translate(clean_text, direction):
                        stats["skipped"] += 1
                        continue

                    if cancel_event and cancel_event.is_set():
                        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                    progress_state["current"] += 1
                    cur_idx = progress_state["current"]
                    total_items = progress_state["total"]

                    if progress_cb:
                        progress_cb(
                            cur_idx,
                            total_items,
                            f'[{cur_idx}/{total_items}] Page {page_num} Block {block_no}: "{clean_text[:20]}.."',
                        )

                    chunk_id = f"p{page_num}_b{block_no}_{hash_text(clean_text)[:6]}"

                    result = self.engine.translate_chunk(
                        text=clean_text,
                        direction=direction,
                        context=page_context,
                        location_id=f"Page {page_num} Block {block_no}",
                        chunk_id=chunk_id,
                        review_log_path=review_log_path,
                        log_cb=log_cb,
                    )
                    if isinstance(result, tuple):
                        result = TranslationResult(*result)

                    if cancel_event and cancel_event.is_set():
                        raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                    if result.was_reverted:
                        stats["reverted"] += 1
                        continue

                    if result.was_translated:
                        stats["translated"] += 1
                        translated_blocks.append(
                            {
                                "rect": rect,
                                "text": result.text,
                                "original": clean_text,
                                "block_no": block_no,
                                "original_fontsize": original_fontsize,
                                "dominant_font": dominant_font,
                                "dominant_color": dominant_color,
                            }
                        )

                if cancel_event and cancel_event.is_set():
                    raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                # Phase 3: Batch Redact & Re-insert
                fitted_blocks = []
                for item in translated_blocks:
                    rect = item["rect"]
                    trans_text = item["text"]

                    if rect.width < 1.0 or rect.height < 1.0:
                        stats["translated"] -= 1
                        stats["reverted"] += 1
                        continue

                    effective_font = "japan" if direction == "en2ja" or HAS_CJK.search(trans_text) else "helv"

                    # Start fitting from the original font size extracted via span metrics
                    start_fontsize = max(6.0, min(72.0, float(item.get("original_fontsize", 12.0))))
                    font_floor = max(4.5, start_fontsize * 0.60)
                    cur_size = start_fontsize
                    fitted = False

                    text_color = self._color_to_rgb(item.get("dominant_color", 0))

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
                                fontname=effective_font,
                                color=text_color,
                                align=fitz.TEXT_ALIGN_LEFT,
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
                                hash_text(item["original"]),
                            )
                        continue

                    item["fitted_size"] = cur_size
                    item["fontname"] = effective_font
                    item["text_color"] = text_color
                    fitted_blocks.append(item)

                if cancel_event and cancel_event.is_set():
                    raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

                if fitted_blocks:
                    # 1. Add all redaction annotations for the page
                    for item in fitted_blocks:
                        bg_color = self._sample_background_color(page, item["rect"])
                        page.add_redact_annot(item["rect"], fill=bg_color)
                        item["bg_color"] = bg_color

                    # 2. Single apply_redactions call for entire page
                    page.apply_redactions()

                    # 3. Insert all translated text boxes
                    for item in fitted_blocks:
                        page.insert_textbox(
                            item["rect"],
                            item["text"],
                            fontsize=item["fitted_size"],
                            fontname=item["fontname"],
                            color=item.get("text_color", (0, 0, 0)),
                            align=fitz.TEXT_ALIGN_LEFT,
                        )

                self.engine.save_cache_atomically(log_cb=log_cb)

            if cancel_event and cancel_event.is_set():
                raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

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
