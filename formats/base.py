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
        """Extracts an OOXML archive (PPTX/XLSX/DOCX) cleanly into a temp workspace."""
        if os.path.exists(target_dir):
            shutil.rmtree(target_dir, ignore_errors=True)
        os.makedirs(target_dir, exist_ok=True)
        with zipfile.ZipFile(archive_path, "r") as z:
            target_root = os.path.abspath(target_dir)
            for member in z.infolist():
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
