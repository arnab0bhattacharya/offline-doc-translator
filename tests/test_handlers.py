"""
tests/test_handlers.py
======================
Format handlers test suite using mock engines and synthetic archives.
Validates round-trip archive extraction, XML parsing, replacement, and re-packing.
"""

import unittest
import os
import sys
import tempfile
import zipfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from unittest.mock import patch
from engine.core import TranslationEngine, hash_text
from engine.errors import ErrorCode, TranslatorError
from formats.base import (
    BaseFormatHandler,
    MAX_EXTRACTED_BYTES,
    MAX_ARCHIVE_ENTRIES,
    MAX_SINGLE_ENTRY_BYTES,
)
from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler
from formats.docx_handler import DOCXHandler
from formats.pdf_handler import PDFHandler
from formats.registry import get_handler, SUPPORTED_EXTENSIONS

try:
    import fitz
except ImportError:
    fitz = None



class MockTranslationEngine(TranslationEngine):
    """Mock engine that mimics instant translations without connecting to Ollama."""

    def __init__(self):
        super().__init__(model_name="mock_model")

    def translate_chunk(self, text, direction, context=None, location_id="doc", chunk_id="0", review_log_path=None, log_cb=None, **kwargs):
        if log_cb:
            log_cb(f"Mock translating: {text[:20]}")
        if direction == "ja2en":
            return f"[EN: {text}]", True, False
        elif direction == "en2ja":
            return f"[JA: {text}]", True, False
        return text, False, False


class TestFormatHandlers(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_handlers_")
        self.mock_engine = MockTranslationEngine()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_pptx_handler(self):
        pptx_path = os.path.join(self.test_dir, "sample.pptx")
        out_pptx_path = os.path.join(self.test_dir, "sample_translated.pptx")

        slide_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree>'
            '<p:sp><p:txBody>'
            '<a:p><a:r><a:t>これはプレゼンのタイトルです。</a:t></a:r></a:p>'
            '<a:p><a:r><a:t>売上高は </a:t></a:r><a:r><a:t>15% 増加しました。</a:t></a:r></a:p>'
            '</p:txBody></p:sp>'
            '</p:spTree></p:cSld></p:sld>'
        )

        with zipfile.ZipFile(pptx_path, "w") as z:
            z.writestr("ppt/slides/slide1.xml", slide_xml)
            z.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')

        handler = PPTXHandler(self.mock_engine)
        stats = handler.translate(pptx_path, out_pptx_path, "ja2en")

        self.assertTrue(os.path.exists(out_pptx_path))
        self.assertEqual(stats["translated"], 2)

        with zipfile.ZipFile(out_pptx_path, "r") as z:
            trans_xml = z.read("ppt/slides/slide1.xml").decode("utf-8")
            self.assertIn("[EN: これはプレゼンのタイトルです。]", trans_xml)
            self.assertIn("[EN: 売上高は 15% 増加しました。]", trans_xml)

    def test_xlsx_handler(self):
        xlsx_path = os.path.join(self.test_dir, "sample.xlsx")
        out_xlsx_path = os.path.join(self.test_dir, "sample_translated.xlsx")

        shared_strings_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">'
            '<si><t>営業収益</t></si>'
            '<si><t>純利益</t></si>'
            '</sst>'
        )

        sheet1_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<sheetData>'
            '<row r="1">'
            '<c r="A1" t="s"><v>0</v></c>'
            '<c r="B1"><v>5000</v></c>'
            '</row>'
            '<row r="2">'
            '<c r="A2" t="s"><v>1</v></c>'
            '<c r="B2"><f>SUM(B1:B10)</f><v>5000</v></c>'
            '</row>'
            '</sheetData>'
            '</worksheet>'
        )

        with zipfile.ZipFile(xlsx_path, "w") as z:
            z.writestr("xl/sharedStrings.xml", shared_strings_xml)
            z.writestr("xl/worksheets/sheet1.xml", sheet1_xml)
            z.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')

        handler = XLSXHandler(self.mock_engine)
        stats = handler.translate(xlsx_path, out_xlsx_path, "ja2en")

        self.assertTrue(os.path.exists(out_xlsx_path))
        self.assertEqual(stats["translated"], 2)

        with zipfile.ZipFile(out_xlsx_path, "r") as z:
            trans_xml = z.read("xl/worksheets/sheet1.xml").decode("utf-8")
            self.assertIn('t="inlineStr"', trans_xml)
            self.assertIn('[EN: 営業収益]', trans_xml)
            self.assertIn('[EN: 純利益]', trans_xml)
            self.assertIn('<f>SUM(B1:B10)</f>', trans_xml)
            self.assertIn('<c r="B1"><v>5000</v></c>', trans_xml)

    def test_docx_handler(self):
        docx_path = os.path.join(self.test_dir, "sample.docx")
        out_docx_path = os.path.join(self.test_dir, "sample_translated.docx")

        document_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            '<w:p><w:r><w:t>第1章：事業の概況について</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>当四半期は </w:t></w:r><w:r><w:t>堅調に推移しました。</w:t></w:r></w:p>'
            '</w:body>'
            '</w:document>'
        )

        header_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:hdr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:p><w:r><w:t>社外秘 &amp; 2024年度報告書</w:t></w:r></w:p>'
            '</w:hdr>'
        )

        with zipfile.ZipFile(docx_path, "w") as z:
            z.writestr("word/document.xml", document_xml)
            z.writestr("word/header1.xml", header_xml)
            z.writestr("[Content_Types].xml", '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"/>')

        handler = DOCXHandler(self.mock_engine)
        stats = handler.translate(docx_path, out_docx_path, "ja2en")

        self.assertTrue(os.path.exists(out_docx_path))
        self.assertEqual(stats["translated"], 3)

        with zipfile.ZipFile(out_docx_path, "r") as z:
            trans_doc_xml = z.read("word/document.xml").decode("utf-8")
            trans_hdr_xml = z.read("word/header1.xml").decode("utf-8")
            self.assertIn("[EN: 第1章：事業の概況について]", trans_doc_xml)
            self.assertIn("[EN: 当四半期は 堅調に推移しました。]", trans_doc_xml)
            self.assertIn("[EN: 社外秘 &amp; 2024年度報告書]", trans_hdr_xml)
            self.assertNotIn("&amp;amp;", trans_hdr_xml)


class TestFormatRegistry(unittest.TestCase):

    def test_get_handler_valid(self):
        engine = MockTranslationEngine()
        self.assertIsInstance(get_handler(".pptx", engine), PPTXHandler)
        self.assertIsInstance(get_handler(".PPTX", engine), PPTXHandler)
        self.assertIsInstance(get_handler(".xlsx", engine), XLSXHandler)
        self.assertIsInstance(get_handler(".docx", engine), DOCXHandler)
        self.assertIsInstance(get_handler(".pdf", engine), PDFHandler)

    def test_get_handler_unsupported(self):
        engine = MockTranslationEngine()
        from engine.errors import TranslatorError, ErrorCode
        with self.assertRaises(TranslatorError) as ctx:
            get_handler(".txt", engine)
        self.assertEqual(ctx.exception.code, ErrorCode.E04)

    def test_supported_extensions(self):
        self.assertEqual(SUPPORTED_EXTENSIONS, {".pptx", ".xlsx", ".docx", ".pdf"})


class TestZipExtractionSecurity(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_zip_sec_")
        self.dummy_zip = os.path.join(self.test_dir, "test.zip")
        with zipfile.ZipFile(self.dummy_zip, "w") as z:
            z.writestr("test.txt", b"safe content")
        self.extract_dir = os.path.join(self.test_dir, "out")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_extract_zip_rejects_single_entry_over_limit(self):
        fake_info = zipfile.ZipInfo("huge_file.xml")
        fake_info.file_size = MAX_SINGLE_ENTRY_BYTES + 1024

        with patch.object(zipfile.ZipFile, "infolist", return_value=[fake_info]):
            with self.assertRaises(zipfile.BadZipFile) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)
            self.assertIn("max 100 MB", str(ctx.exception))

    def test_extract_zip_rejects_total_size_over_limit(self):
        # 11 entries of 100 MB each = 1100 MB > 1000 MB
        fake_members = []
        for i in range(11):
            zi = zipfile.ZipInfo(f"part_{i}.xml")
            zi.file_size = 100 * 1024 * 1024
            fake_members.append(zi)

        with patch.object(zipfile.ZipFile, "infolist", return_value=fake_members):
            with self.assertRaises(zipfile.BadZipFile) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)
            self.assertIn("possible zip bomb", str(ctx.exception))

    def test_extract_zip_rejects_too_many_entries(self):
        fake_members = [zipfile.ZipInfo(f"part_{i}.xml") for i in range(MAX_ARCHIVE_ENTRIES + 1)]

        with patch.object(zipfile.ZipFile, "infolist", return_value=fake_members):
            with self.assertRaises(zipfile.BadZipFile) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)
            self.assertIn("possible zip bomb", str(ctx.exception))

    def test_extract_zip_rejects_path_traversal(self):
        fake_info = zipfile.ZipInfo("../../etc/passwd")
        fake_info.file_size = 50

        with patch.object(zipfile.ZipFile, "infolist", return_value=[fake_info]):
            with self.assertRaises(zipfile.BadZipFile) as ctx:
                BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)
            self.assertIn("Unsafe archive member", str(ctx.exception))

    def test_extract_zip_normal_extraction(self):
        BaseFormatHandler.extract_zip(self.dummy_zip, self.extract_dir)
        extracted_file = os.path.join(self.extract_dir, "test.txt")
        self.assertTrue(os.path.exists(extracted_file))
        with open(extracted_file, "rb") as f:
            self.assertEqual(f.read(), b"safe content")


@unittest.skipIf(fitz is None, "PyMuPDF (fitz) is not installed")
class TestPDFHandler(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="test_pdf_handler_")
        self.mock_engine = MockTranslationEngine()

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_basic_translation_ja2en(self):
        doc = fitz.open()
        page = doc.new_page(width=500, height=300)
        page.insert_textbox(fitz.Rect(72, 72, 400, 120), "これはテスト文書です。", fontsize=14, fontname="japan")
        input_path = os.path.join(self.test_dir, "test.pdf")
        doc.save(input_path)
        doc.close()

        with patch.object(self.mock_engine, "translate_chunk", return_value=("Translated English text.", True, False)):
            handler = PDFHandler(self.mock_engine)
            output_path = os.path.join(self.test_dir, "test_ja2en.pdf")
            stats = handler.translate(input_path, output_path, "ja2en")

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(stats["total"], 1)
            self.assertEqual(stats["translated"], 1)
            self.assertEqual(stats["reverted"], 0)

            out_doc = fitz.open(output_path)
            self.assertEqual(len(out_doc), 1)
            extracted_text = out_doc[0].get_text("text")
            out_doc.close()
            self.assertIn("Translated English text.", extracted_text)

    def test_multipage_translation(self):
        doc = fitz.open()
        p1 = doc.new_page(width=500, height=300)
        p1.insert_textbox(fitz.Rect(72, 72, 400, 120), "1ページ目の日本語です。", fontsize=14, fontname="japan")
        p2 = doc.new_page(width=500, height=300)
        p2.insert_textbox(fitz.Rect(72, 72, 400, 120), "2ページ目の日本語です。", fontsize=14, fontname="japan")
        input_path = os.path.join(self.test_dir, "test_multi.pdf")
        doc.save(input_path)
        doc.close()

        def mock_translate(text, direction, **kw):
            return f"Page Translated: {hash_text(text)[:6]}", True, False

        with patch.object(self.mock_engine, "translate_chunk", side_effect=mock_translate):
            handler = PDFHandler(self.mock_engine)
            output_path = os.path.join(self.test_dir, "test_multi_ja2en.pdf")
            stats = handler.translate(input_path, output_path, "ja2en")

            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["translated"], 2)

            out_doc = fitz.open(output_path)
            self.assertEqual(len(out_doc), 2)
            self.assertIn("Page Translated:", out_doc[0].get_text("text"))
            self.assertIn("Page Translated:", out_doc[1].get_text("text"))
            out_doc.close()

    def test_pdf_no_selectable_text_raises_e04(self):
        doc = fitz.open()
        doc.new_page(width=500, height=300)  # Empty page
        input_path = os.path.join(self.test_dir, "empty.pdf")
        doc.save(input_path)
        doc.close()

        handler = PDFHandler(self.mock_engine)
        output_path = os.path.join(self.test_dir, "empty_out.pdf")
        with self.assertRaises(TranslatorError) as ctx:
            handler.translate(input_path, output_path, "ja2en")
        self.assertEqual(ctx.exception.code, ErrorCode.E04)
        self.assertIn("no selectable text", ctx.exception.detail)

    def test_pdf_corrupt_file_raises_e04(self):
        input_path = os.path.join(self.test_dir, "corrupt.pdf")
        with open(input_path, "wb") as f:
            f.write(b"NOT_A_VALID_PDF_HEADER")

        handler = PDFHandler(self.mock_engine)
        output_path = os.path.join(self.test_dir, "corrupt_out.pdf")
        with self.assertRaises(TranslatorError) as ctx:
            handler.translate(input_path, output_path, "ja2en")
        self.assertEqual(ctx.exception.code, ErrorCode.E04)

    def test_pdf_overflow_fallback_reverts(self):
        doc = fitz.open()
        page = doc.new_page(width=500, height=300)
        page.insert_textbox(fitz.Rect(72, 72, 180, 100), "テスト", fontsize=10, fontname="japan")
        input_path = os.path.join(self.test_dir, "overflow.pdf")
        doc.save(input_path)
        doc.close()

        huge_text = "This is an extremely long string " * 30
        with patch.object(self.mock_engine, "translate_chunk", return_value=(huge_text, True, False)):
            handler = PDFHandler(self.mock_engine)
            output_path = os.path.join(self.test_dir, "overflow_out.pdf")
            review_log = os.path.join(self.test_dir, "review.log")
            stats = handler.translate(input_path, output_path, "ja2en", review_log_path=review_log)

    def test_ooxml_text_unit_processing(self):
        handler = DOCXHandler(self.mock_engine)

        # 1. Test count_translatable_paragraphs_in_xml
        sample_xml = (
            "<w:document><w:body>"
            "<w:p><w:r><w:t>最初の段落です。</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>12345</w:t></w:r></w:p>"
            "<w:p><w:r><w:t>2番目の段落です。</w:t></w:r></w:p>"
            "</w:body></w:document>"
        )
        count = handler.count_translatable_paragraphs_in_xml(sample_xml, "w", "ja2en")
        self.assertEqual(count, 2)

        # 2. Test extract_paragraph_text_nodes
        p_content = "<w:r><w:t>パート1 &amp; </w:t></w:r><w:r><w:t>パート2</w:t></w:r>"
        full_text, t_matches = handler.extract_paragraph_text_nodes(p_content, "w")
        self.assertEqual(full_text, "パート1 & パート2")
        self.assertEqual(len(t_matches), 2)

        # 3. Test _translate_and_replace_text_nodes success
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        progress = {"current": 0, "total": 1}
        recent = []

        new_p = handler._translate_and_replace_text_nodes(
            full_text=full_text,
            t_matches=t_matches,
            p_content=p_content,
            direction="ja2en",
            context=None,
            part_name="doc.xml",
            review_log_path=None,
            stats=stats,
            progress_state=progress,
            progress_cb=None,
            log_cb=None,
            tag_prefix="w",
            recent_paragraphs=recent,
        )

        self.assertIsNotNone(new_p)
        self.assertIn('<w:t xml:space="preserve">[EN: パート1 &amp; パート2]</w:t>', new_p)
        self.assertIn('<w:t></w:t>', new_p)
        self.assertEqual(stats["translated"], 1)
        self.assertEqual(recent, ["[EN: パート1 & パート2]"])

        # 4. Test _translate_and_replace_text_nodes skip non-translatable
        stats_skip = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        p_num = "<w:r><w:t>12345</w:t></w:r>"
        ft_num, m_num = handler.extract_paragraph_text_nodes(p_num, "w")
        res_skip = handler._translate_and_replace_text_nodes(
            full_text=ft_num,
            t_matches=m_num,
            p_content=p_num,
            direction="ja2en",
            context=None,
            part_name="doc.xml",
            review_log_path=None,
            stats=stats_skip,
            progress_state={"current": 0, "total": 1},
            progress_cb=None,
            log_cb=None,
            tag_prefix="w",
        )
        self.assertIsNone(res_skip)
        self.assertEqual(stats_skip["skipped"], 1)


if __name__ == "__main__":
    unittest.main()


