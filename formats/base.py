"""
formats/base.py
===============
Abstract base class defining the contract for all document format handlers.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict, Any
import os
import shutil
import zipfile
import tempfile

try:
    from ..engine.core import TranslationEngine
except (ImportError, ValueError):
    from engine.core import TranslationEngine

# Module-level constants for zip bomb protection
MAX_EXTRACTED_BYTES = 1_000 * 1024 * 1024   # 1 GB
MAX_ARCHIVE_ENTRIES = 20_000
MAX_SINGLE_ENTRY_BYTES = 100 * 1024 * 1024  # 100 MB


class BaseFormatHandler(ABC):
    """Base class for document format translation handlers."""

    def __init__(self, engine: TranslationEngine):
        self.engine = engine

    @abstractmethod
    def translate(
        self,
        input_path: str,
        output_path: str,
        direction: str,
        review_log_path: Optional[str] = None,
        progress_cb: Optional[Callable[[int, int, str], None]] = None
    ) -> Dict[str, Any]:
        """
        Translates the document from input_path to output_path.
        
        Args:
            input_path: Absolute or relative path to source document.
            output_path: Path where translated document should be saved.
            direction: 'ja2en' or 'en2ja'.
            review_log_path: Path for audit log of skipped/failed items.
            progress_cb: Optional callback func(current_step, total_steps, message).
            
        Returns:
            Dict with statistics: {"total": int, "translated": int, "reverted": int, "skipped": int}
        """
        pass

    @staticmethod
    def extract_zip(archive_path: str, target_dir: str) -> None:
        """Extracts an OOXML archive with decompression-bomb protection."""
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)

        with zipfile.ZipFile(archive_path, "r") as z:
            target_root = os.path.abspath(target_dir)

            # Security: check entry count
            members = z.infolist()
            if len(members) > MAX_ARCHIVE_ENTRIES:
                raise zipfile.BadZipFile(
                    f"Archive has {len(members)} entries (max {MAX_ARCHIVE_ENTRIES}). "
                    f"Refusing to extract — possible zip bomb."
                )

            # Security: check total and per-entry uncompressed sizes
            total_uncompressed = 0
            for member in members:
                if member.file_size > MAX_SINGLE_ENTRY_BYTES:
                    raise zipfile.BadZipFile(
                        f"Archive member '{member.filename}' is {member.file_size / 1024 / 1024:.0f} MB "
                        f"(max {MAX_SINGLE_ENTRY_BYTES / 1024 / 1024:.0f} MB)."
                    )
                total_uncompressed += member.file_size

            if total_uncompressed > MAX_EXTRACTED_BYTES:
                raise zipfile.BadZipFile(
                    f"Archive total uncompressed size is {total_uncompressed / 1024 / 1024:.0f} MB "
                    f"(max {MAX_EXTRACTED_BYTES / 1024 / 1024:.0f} MB). "
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
