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

from engine.core import TranslationEngine
from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler
from formats.docx_handler import DOCXHandler
from formats.pdf_handler import PDFHandler
from formats.registry import get_handler, SUPPORTED_EXTENSIONS


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


if __name__ == "__main__":
    unittest.main()
