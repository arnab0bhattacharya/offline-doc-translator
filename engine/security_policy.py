"""
engine/security_policy.py
=========================
Centralized document security policy and resource limit configurations.
Guards against decompression bombs, runaway XML parts, oversized documents,
and memory exhaustion.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DocumentSecurityPolicy:
    """Centralized resource limits consumed by format handlers and engine."""
    max_input_bytes: int = 250 * 1024 * 1024        # 250 MB
    max_archive_entries: int = 20_000               # 20,000 entries
    max_extracted_bytes: int = 1_000 * 1024 * 1024   # 1 GB total uncompressed
    max_xml_part_bytes: int = 50 * 1024 * 1024       # 50 MB per XML part
    max_text_chunk_chars: int = 32_000              # 32,000 chars per unit
    max_pdf_pages: int = 5_000                      # 5,000 pages
    max_single_entry_bytes: int = 100 * 1024 * 1024  # 100 MB per archive entry


DEFAULT_POLICY = DocumentSecurityPolicy()
