"""
formats/base.py
===============
Abstract base class defining the contract for all document format handlers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any, List, Tuple, Union, TYPE_CHECKING
import os
import re
import shutil
import zipfile
import tempfile
import threading

from engine.core import escape_xml, unescape_xml, hash_text, should_translate
from engine.errors import ErrorCode, TranslatorError
from engine.security_policy import DocumentSecurityPolicy, DEFAULT_POLICY

if TYPE_CHECKING:
    from engine.core import TranslationEngine

# Module-level constants for zip bomb protection (derived from DEFAULT_POLICY)
MAX_EXTRACTED_BYTES = DEFAULT_POLICY.max_extracted_bytes
MAX_ARCHIVE_ENTRIES = DEFAULT_POLICY.max_archive_entries
MAX_SINGLE_ENTRY_BYTES = DEFAULT_POLICY.max_single_entry_bytes


class BaseFormatHandler(ABC):
    """Base class for document format translation handlers."""

    def __init__(
        self,
        engine: "TranslationEngine",
        policy: Optional[DocumentSecurityPolicy] = None,
    ):
        self.engine = engine
        self.policy = policy or DEFAULT_POLICY

    @abstractmethod
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
        """
        Translates the document from input_path to output_path.
        
        Args:
            input_path: Absolute or relative path to source document.
            output_path: Path where translated document should be saved.
            direction: 'ja2en' or 'en2ja'.
            review_log_path: Path for audit log of skipped/failed items.
            progress_cb: Optional callback func(current_step, total_steps, message).
            log_cb: Optional callback func(log_message).
            cancel_event: Optional threading.Event to signal cooperative cancellation.
            
        Returns:
            Dict with statistics: {"total": int, "translated": int, "reverted": int, "skipped": int}
        """
        pass

    def validate_input_file(self, input_path: str) -> None:
        """Validates that input file exists and does not exceed policy maximum input size."""
        if not os.path.exists(input_path):
            raise TranslatorError(
                ErrorCode.E04,
                detail=f"Input file not found: '{input_path}'."
            )
        file_size = os.path.getsize(input_path)
        if file_size > self.policy.max_input_bytes:
            raise TranslatorError(
                ErrorCode.E04,
                detail=(
                    f"Input file '{os.path.basename(input_path)}' ({file_size / 1024 / 1024:.1f} MB) "
                    f"exceeds maximum allowed size ({self.policy.max_input_bytes / 1024 / 1024:.0f} MB)."
                )
            )

    def validate_xml_part_size(self, file_path: str) -> None:
        """Validates that an uncompressed XML part does not exceed policy limit."""
        if os.path.exists(file_path):
            part_size = os.path.getsize(file_path)
            if part_size > self.policy.max_xml_part_bytes:
                raise TranslatorError(
                    ErrorCode.E04,
                    detail=(
                        f"XML part '{os.path.basename(file_path)}' ({part_size / 1024 / 1024:.1f} MB) "
                        f"exceeds maximum allowed size ({self.policy.max_xml_part_bytes / 1024 / 1024:.0f} MB)."
                    )
                )

    @staticmethod
    def extract_zip(
        archive_path: str,
        target_dir: str,
        policy: Optional[DocumentSecurityPolicy] = None,
    ) -> None:
        """Extracts an OOXML archive with decompression-bomb protection."""
        pol = policy or DEFAULT_POLICY
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as z:
            target_root = os.path.abspath(target_dir)

            # Security: check entry count
            members = z.infolist()
            if len(members) > pol.max_archive_entries:
                raise zipfile.BadZipFile(
                    f"Archive has {len(members)} entries (max {pol.max_archive_entries}). "
                    f"Refusing to extract — possible zip bomb."
                )

            # Security: check total and per-entry uncompressed sizes
            total_uncompressed = 0
            for member in members:
                if member.file_size > pol.max_single_entry_bytes:
                    raise zipfile.BadZipFile(
                        f"Archive member '{member.filename}' is {member.file_size / 1024 / 1024:.0f} MB "
                        f"(max {pol.max_single_entry_bytes / 1024 / 1024:.0f} MB)."
                    )
                total_uncompressed += member.file_size

            if total_uncompressed > pol.max_extracted_bytes:
                raise zipfile.BadZipFile(
                    f"Archive total uncompressed size is {total_uncompressed / 1024 / 1024:.0f} MB "
                    f"(max {pol.max_extracted_bytes / 1024 / 1024:.0f} MB). "
                    f"Refusing to extract — possible zip bomb."
                )

            # Security: path traversal check
            for member in members:
                member_path = os.path.abspath(os.path.join(target_root, member.filename))
                if os.path.commonpath([target_root, member_path]) != target_root:
                    raise zipfile.BadZipFile(f"Unsafe archive member: {member.filename}")

            z.extractall(target_dir)


    @staticmethod
    def pack_zip(source_dir: str, output_archive: str) -> None:
        """Writes, integrity-checks, and atomically replaces an OOXML archive."""
        out_dir = os.path.dirname(os.path.abspath(output_archive))
        os.makedirs(out_dir, exist_ok=True)
        fd, temp_archive = tempfile.mkstemp(prefix=".translated_", suffix=".zip", dir=out_dir)
        os.close(fd)
        try:
            with zipfile.ZipFile(temp_archive, "w", zipfile.ZIP_DEFLATED) as z:
                for root, _, files in os.walk(source_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        arcname = os.path.relpath(file_path, source_dir)
                        z.write(file_path, arcname)

            with zipfile.ZipFile(temp_archive, "r") as z:
                if z.testzip() is not None:
                    raise zipfile.BadZipFile("Translated archive failed integrity validation.")
            os.replace(temp_archive, output_archive)
        finally:
            if os.path.exists(temp_archive):
                os.remove(temp_archive)

    # ── Shared OOXML Text-Unit Processing ─────────────────────────

    @staticmethod
    def count_translatable_paragraphs_in_xml(
        xml_str: str,
        tag_prefix: str,
        direction: str,
    ) -> int:
        """
        Counts total translatable paragraphs in an OOXML XML string.
        tag_prefix is 'w' for Word (.docx) or 'a' for PowerPoint (.pptx).
        """
        total = 0
        p_pattern = re.compile(rf"<{tag_prefix}:p(?: [^>]+)?>(.*?)</{tag_prefix}:p>", re.DOTALL)
        t_pattern = re.compile(rf"<{tag_prefix}:t(?:\s[^>]*)?>(.*?)</{tag_prefix}:t>", re.DOTALL)

        for p_match in p_pattern.finditer(xml_str):
            t_matches = t_pattern.findall(p_match.group(1))
            full_text = unescape_xml("".join(t_matches)).strip()
            if full_text and should_translate(full_text, direction):
                total += 1
        return total

    @staticmethod
    def extract_paragraph_text_nodes(
        p_content: str,
        tag_prefix: str,
    ) -> Tuple[str, List[Any]]:
        """
        Extracts unescaped aggregated text and regex match objects for <prefix:t> nodes.
        tag_prefix is 'w' for Word (.docx) or 'a' for PowerPoint (.pptx).
        """
        t_pattern = re.compile(rf"(<{tag_prefix}:t(?:\s[^>]*)?>)(.*?)(</{tag_prefix}:t>)", re.DOTALL)
        t_matches = list(t_pattern.finditer(p_content))
        if not t_matches:
            return "", []
        full_text = unescape_xml("".join(m.group(2) for m in t_matches))
        return full_text, t_matches

    def _translate_and_replace_text_nodes(
        self,
        full_text: str,
        t_matches: List[Any],
        p_content: str,
        direction: str,
        context: Optional[Union[str, Callable[[str], Optional[str]]]],
        part_name: str,
        review_log_path: Optional[str],
        stats: Dict[str, int],
        progress_state: Dict[str, Any],
        progress_cb: Optional[Callable[[int, int, str], None]],
        log_cb: Optional[Callable[[str], None]],
        tag_prefix: str,
        recent_paragraphs: Optional[List[str]] = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Optional[str]:
        """
        Shared logic for translating and replacing text units within an OOXML paragraph:
        1. Validates text and checks should_translate.
        2. Dispatches progress callback.
        3. Calls self.engine.translate_chunk with context and error handling.
        4. On translation: replaces first <tag_prefix:t> node with translated text
           (marked xml:space="preserve") and empties subsequent nodes.
        5. Updates stats and optional sliding context list (recent_paragraphs).

        Returns:
            Modified p_content string if translated, or None if skipped/reverted/unmodified.
        """
        if cancel_event and cancel_event.is_set():
            raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

        if not t_matches or not full_text or not full_text.strip():
            return None

        if len(full_text) > self.policy.max_text_chunk_chars:
            raise TranslatorError(
                ErrorCode.E04,
                detail=(
                    f"Text chunk length ({len(full_text)} chars) exceeds maximum allowed limit "
                    f"({self.policy.max_text_chunk_chars} chars)."
                )
            )

        stats["total"] += 1

        if not should_translate(full_text, direction):
            stats["skipped"] += 1
            if recent_paragraphs is not None:
                recent_paragraphs.append(full_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)
            return None

        if cancel_event and cancel_event.is_set():
            raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

        progress_state["current"] += 1
        cur_idx = progress_state["current"]
        total_items = progress_state["total"]

        if progress_cb:
            progress_cb(
                cur_idx,
                total_items,
                f"[{cur_idx}/{total_items}] {part_name}: \"{full_text[:20]}..\""
            )

        context_str = context(full_text) if callable(context) else context
        chunk_id = hash_text(full_text)[:8]

        translated_text, was_translated, was_reverted = self.engine.translate_chunk(
            text=full_text,
            direction=direction,
            context=context_str,
            location_id=part_name,
            chunk_id=chunk_id,
            review_log_path=review_log_path,
            log_cb=log_cb,
        )

        if cancel_event and cancel_event.is_set():
            raise TranslatorError(ErrorCode.E09, detail="Translation cancelled by user.")

        if was_reverted:
            stats["reverted"] += 1
            if recent_paragraphs is not None:
                recent_paragraphs.append(full_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)
            return None

        if was_translated:
            stats["translated"] += 1
            escaped_translation = escape_xml(translated_text)

            if recent_paragraphs is not None:
                recent_paragraphs.append(translated_text.strip())
                if len(recent_paragraphs) > 2:
                    recent_paragraphs.pop(0)

            new_p_content = ""
            last_idx = 0
            for i, m in enumerate(t_matches):
                new_p_content += p_content[last_idx:m.start()]
                if i == 0:
                    new_p_content += f'<{tag_prefix}:t xml:space="preserve">{escaped_translation}</{tag_prefix}:t>'
                else:
                    new_p_content += f'<{tag_prefix}:t></{tag_prefix}:t>'
                last_idx = m.end()

            new_p_content += p_content[last_idx:]
            return new_p_content

        return None

