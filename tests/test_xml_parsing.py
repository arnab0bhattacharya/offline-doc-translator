"""
tests/test_xml_parsing.py
=========================
Comprehensive fixture-based XML regex edge-case test suite.
Validates OOXML regex parsing, attribute handling, tag preservation,
performance, and entity handling across Word (.docx), PowerPoint (.pptx),
and Excel (.xlsx) handlers.
"""

import unittest
import os
import sys
import time
import threading

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core import TranslationEngine, escape_xml, unescape_xml
from engine.errors import ErrorCode, TranslatorError
from formats.base import BaseFormatHandler, XML_TAG_ATTRS
from formats.docx_handler import DOCXHandler
from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler


class MockXMLTranslationEngine(TranslationEngine):
    """Mock engine that logs translated strings and mimics translation."""

    def __init__(self, translation_prefix="[TRANS: ", should_revert=False):
        super().__init__(model_name="mock_xml_engine")
        self.translation_prefix = translation_prefix
        self.should_revert = should_revert
        self.received_texts = []
        self.received_contexts = []

    def translate_chunk(self, text, direction, context=None, location_id="doc", chunk_id="0", review_log_path=None, log_cb=None, **kwargs):
        self.received_texts.append(text)
        self.received_contexts.append(context)
        if self.should_revert:
            return text, False, True
        return f"{self.translation_prefix}{text}]", True, False


class TestXMLParsingEdgeCases(unittest.TestCase):
    """Fixture-based tests for all 9 XML regex edge cases."""

    def setUp(self):
        self.engine = MockXMLTranslationEngine()
        self.docx_handler = DOCXHandler(self.engine)
        self.pptx_handler = PPTXHandler(self.engine)
        self.xlsx_handler = XLSXHandler(self.engine)

    def _init_stats_prog(self, total=1):
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        prog = {"current": 0, "total": total}
        return stats, prog

    # ─────────────────────────────────────────────────────────────
    # Case 1: Paragraph with no <w:t> nodes
    # ─────────────────────────────────────────────────────────────
    def test_case_1_paragraph_with_no_text_nodes(self):
        """Paragraph with no <w:t> nodes passes through unchanged."""
        xml_input = (
            '<w:body>'
            '<w:p><w:pPr><w:pStyle w:val="Heading1"/><w:jc w:val="center"/></w:pPr></w:p>'
            '<w:p><w:r><w:br/></w:r></w:p>'
            '</w:body>'
        )
        stats, prog = self._init_stats_prog(2)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(result, xml_input)
        self.assertEqual(stats["total"], 0)
        self.assertEqual(stats["translated"], 0)
        self.assertEqual(len(self.engine.received_texts), 0)

    # ─────────────────────────────────────────────────────────────
    # Case 2: Multiple <w:t> nodes in one paragraph
    # ─────────────────────────────────────────────────────────────
    def test_case_2_multiple_text_nodes_in_paragraph(self):
        """Multiple <w:t> nodes: translation goes into first, others emptied."""
        xml_input = (
            '<w:p>'
            '<w:r><w:t>これは</w:t></w:r>'
            '<w:r><w:t>複数ノードの</w:t></w:r>'
            '<w:r><w:t>テストです。</w:t></w:r>'
            '</w:p>'
        )
        stats, prog = self._init_stats_prog(1)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(self.engine.received_texts, ["これは複数ノードのテストです。"])
        self.assertEqual(stats["translated"], 1)

        # First <w:t> has translation, subsequent <w:t> are empty
        expected_first = '<w:t xml:space="preserve">[TRANS: これは複数ノードのテストです。]</w:t>'
        expected_subsequent = '<w:t></w:t>'
        self.assertIn(expected_first, result)
        self.assertEqual(result.count(expected_subsequent), 2)
        # Verify structure: runs are preserved
        self.assertTrue(result.startswith('<w:p><w:r>'))
        self.assertTrue(result.endswith('</w:t></w:r></w:p>'))

    # ─────────────────────────────────────────────────────────────
    # Case 3: Paragraph with XML entities
    # ─────────────────────────────────────────────────────────────
    def test_case_3_xml_entities_roundtrip(self):
        """XML entities (&amp;, &lt;, &gt;, &quot;, &apos;) are unescaped for translation and re-escaped on output."""
        xml_input = (
            '<w:p>'
            '<w:r><w:t>研究 &amp; 開発: &lt;重要度&gt; &quot;高&quot; &apos;必須&apos;</w:t></w:r>'
            '</w:p>'
        )
        stats, prog = self._init_stats_prog(1)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        # Verify the engine received properly unescaped text
        self.assertEqual(len(self.engine.received_texts), 1)
        received = self.engine.received_texts[0]
        self.assertEqual(received, '研究 & 開発: <重要度> "高" \'必須\'')

        # Verify output XML is safely escaped
        self.assertIn('&amp;', result)
        self.assertIn('&lt;重要度&gt;', result)
        self.assertIn('&quot;高&quot;', result)
        self.assertIn('&apos;必須&apos;', result)
        # Unescaped angle brackets or ampersands should not appear in text content
        self.assertNotIn('<重要度>', result)

    # ─────────────────────────────────────────────────────────────
    # Case 4: Empty paragraphs
    # ─────────────────────────────────────────────────────────────
    def test_case_4_empty_paragraphs_flavors(self):
        """Empty paragraphs (self-closing, empty content, whitespace) pass through unchanged."""
        xml_inputs = [
            '<w:p/>',
            '<w:p></w:p>',
            '<w:p><w:pPr/></w:p>',
            '<w:p><w:r><w:t></w:t></w:r></w:p>',
            '<w:p><w:r><w:t>    </w:t></w:r></w:p>',
            '<w:p><w:r><w:t>\t\n\r</w:t></w:r></w:p>',
        ]
        for snippet in xml_inputs:
            stats, prog = self._init_stats_prog(1)
            result = self.docx_handler._process_xml_content(
                xml_str=snippet,
                part_name="document.xml",
                direction="ja2en",
                review_log_path=None,
                stats=stats,
                progress_state=prog,
                progress_cb=None,
                log_cb=None,
            )
            self.assertEqual(result, snippet, f"Snippet '{snippet}' should pass through unmodified.")
            self.assertEqual(stats["translated"], 0)

    # ─────────────────────────────────────────────────────────────
    # Case 5: Nested tags / formatting runs around <w:t>
    # ─────────────────────────────────────────────────────────────
    def test_case_5_nested_formatting_tags_preserved(self):
        """Paragraph-level and run-level formatting tags (<w:pPr>, <w:rPr>, <w:b/>, etc.) are strictly preserved."""
        xml_input = (
            '<w:p w14:paraId="1234ABCD">'
            '<w:pPr><w:jc w:val="both"/><w:rPr><w:lang w:val="ja-JP"/></w:rPr></w:pPr>'
            '<w:r>'
            '<w:rPr><w:b/><w:i/><w:color w:val="FF0000"/><w:sz w:val="28"/></w:rPr>'
            '<w:t>太字で赤色のテキスト</w:t>'
            '</w:r>'
            '<w:r><w:rPr><w:highlight w:val="yellow"/></w:rPr><w:t>ハイライトテキスト</w:t></w:r>'
            '</w:p>'
        )
        stats, prog = self._init_stats_prog(1)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(self.engine.received_texts, ["太字で赤色のテキストハイライトテキスト"])
        self.assertIn('<w:jc w:val="both"/>', result)
        self.assertIn('<w:b/>', result)
        self.assertIn('<w:i/>', result)
        self.assertIn('<w:color w:val="FF0000"/>', result)
        self.assertIn('<w:sz w:val="28"/>', result)
        self.assertIn('<w:highlight w:val="yellow"/>', result)
        self.assertIn('<w:t xml:space="preserve">[TRANS: 太字で赤色のテキストハイライトテキスト]</w:t>', result)
        self.assertIn('<w:t></w:t>', result)

    def test_case_5_inner_tag_within_text_node(self):
        """Unexpected non-standard tag inside <w:t> does not break regex parsing."""
        xml_input = '<w:p><w:r><w:t>前半<w:customSubTag val="1"/>後半</w:t></w:r></w:p>'
        stats, prog = self._init_stats_prog(1)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )
        self.assertIn('[TRANS:', result)
        self.assertTrue(result.startswith('<w:p>'))
        self.assertTrue(result.endswith('</w:p>'))

    # ─────────────────────────────────────────────────────────────
    # Case 6: Attribute values containing '>'
    # ─────────────────────────────────────────────────────────────
    def test_case_6_attribute_values_containing_greater_than_docx(self):
        """DOCX paragraph and text node attributes containing '>' do not cleave tags."""
        xml_input = (
            '<w:p w14:paraId="ABCD" customAttr="a > b" other=\'threshold > 10\'>'
            '<w:r>'
            '<w:t note="score > 90" xml:space="preserve">合格テキスト</w:t>'
            '</w:r>'
            '</w:p>'
        )
        stats, prog = self._init_stats_prog(1)
        result = self.docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(self.engine.received_texts, ["合格テキスト"])
        self.assertIn('<w:p w14:paraId="ABCD" customAttr="a > b" other=\'threshold > 10\'>', result)
        self.assertIn('[TRANS: 合格テキスト]', result)
        self.assertNotIn('b">', result.split('<w:p')[1].split('>')[0])  # ensure tag boundary wasn't corrupted

    def test_case_6_attribute_values_containing_greater_than_pptx(self):
        """PPTX slide paragraph and field attributes containing '>' do not cleave tags."""
        xml_input = (
            '<a:p custom="count > 5" style=\'width > 100\'>'
            '<a:r><a:t note="val > 0">スライド本文</a:t></a:r>'
            '</a:p>'
        )
        stats, prog = self._init_stats_prog(1)
        result = self.pptx_handler._process_slide_xml(
            xml_str=xml_input,
            slide_name="slide1",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            slide_title=None,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(self.engine.received_texts, ["スライド本文"])
        self.assertIn('<a:p custom="count > 5" style=\'width > 100\'>', result)
        self.assertIn('[TRANS: スライド本文]', result)

    def test_case_6_attribute_values_containing_greater_than_xlsx(self):
        """XLSX row and cell attributes containing '>' do not break cell scanning or replacement."""
        xml_sheet = (
            '<worksheet>'
            '<sheetData>'
            '<row r="1" custom="foo > bar">'
            '<c r="A1" custom="x > y" t="inlineStr"><is><t>セル一</t></is></c>'
            '<c r="B1" custom="1 > 0"/>'
            '<c r="C1" t="s" custom="val > 9"><v>0</v></c>'
            '</row>'
            '</sheetData>'
            '</worksheet>'
        )
        stats, prog = self._init_stats_prog(2)
        result = self.xlsx_handler._process_sheet_xml(
            sheet_xml=xml_sheet,
            sheet_name="Sheet1",
            shared_strings=["セル二"],
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(stats["translated"], 2)
        self.assertIn('<row r="1" custom="foo > bar">', result)
        self.assertIn('<c r="A1" t="inlineStr">', result)
        self.assertIn('[TRANS: セル一]', result)
        self.assertIn('<c r="B1" custom="1 > 0"/>', result)
        self.assertIn('<c r="C1" t="inlineStr">', result)
        self.assertIn('[TRANS: セル二]', result)

    # ─────────────────────────────────────────────────────────────
    # Case 7: Very long single-line XML (performance & catastrophic backtracking check)
    # ─────────────────────────────────────────────────────────────
    def test_case_7_very_long_single_line_xml_performance(self):
        """Very long single-line XML processes rapidly with linear scaling and no catastrophic backtracking."""
        num_paragraphs = 500
        paragraphs = []
        for i in range(num_paragraphs):
            paragraphs.append(
                f'<w:p w14:paraId="{i:08X}" custom="attr_{i} > 0">'
                f'<w:r><w:rPr><w:b/></w:rPr><w:t>これはテスト段落の番号{i}です。</w:t></w:r>'
                f'</w:p>'
            )
        single_line_xml = f'<w:document><w:body>{"".join(paragraphs)}</w:body></w:document>'
        self.assertGreater(len(single_line_xml), 50_000)

        stats, prog = self._init_stats_prog(num_paragraphs)

        start_time = time.perf_counter()
        result = self.docx_handler._process_xml_content(
            xml_str=single_line_xml,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )
        elapsed = time.perf_counter() - start_time

        self.assertEqual(stats["translated"], num_paragraphs)
        self.assertLess(elapsed, 2.0, f"Processing {num_paragraphs} paragraphs took {elapsed:.2f}s (expected < 2.0s)")
        self.assertIn('[TRANS: これはテスト段落の番号0です。]', result)
        self.assertIn(f'[TRANS: これはテスト段落の番号{num_paragraphs - 1}です。]', result)

    def test_case_7_pathological_unclosed_tags_no_freeze(self):
        """Unclosed tags and degenerate strings do not cause regex timeout or hanging."""
        degenerate_inputs = [
            '<w:p attr="' + 'x' * 10000,
            '<w:p><w:r><w:t>' + 'あ' * 10000,
            '<a:p ' + 'x="1" ' * 1000 + '>',
            '<row ' + 'custom="a > b" ' * 500 + '><c ' + 'r="A1" ' * 500,
        ]
        for degenerate in degenerate_inputs:
            start_time = time.perf_counter()
            # Test DOCX pattern
            self.docx_handler.count_translatable_paragraphs_in_xml(degenerate, "w", "ja2en")
            # Test PPTX title extraction
            self.pptx_handler._extract_slide_title(degenerate)
            elapsed = time.perf_counter() - start_time
            self.assertLess(elapsed, 0.5, f"Degenerate input took {elapsed:.2f}s (expected < 0.5s)")

    # ─────────────────────────────────────────────────────────────
    # Case 8: XLSX shared string with rich text formatting (<r> runs inside <si>)
    # ─────────────────────────────────────────────────────────────
    def test_case_8_xlsx_shared_string_rich_text_runs(self):
        """XLSX shared strings containing rich text runs (<r><t>...</t></r>) are merged correctly."""
        shared_strings_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="2" uniqueCount="2">'
            '<si>'
            '<r><rPr><b/><color rgb="FF0000"/></rPr><t>東京</t></r>'
            '<r><rPr><i/></rPr><t>特許</t></r>'
            '<r><t>許可局</t></r>'
            '</si>'
            '<si>'
            '<t>プレーン文字列</t>'
            '</si>'
            '</sst>'
        )

        import tempfile
        tmp_sst = os.path.join(tempfile.gettempdir(), f"test_sst_{os.getpid()}.xml")
        try:
            with open(tmp_sst, "w", encoding="utf-8") as f:
                f.write(shared_strings_xml)

            strings = self.xlsx_handler._parse_shared_strings(tmp_sst)
            self.assertEqual(len(strings), 2)
            self.assertEqual(strings[0], "東京特許許可局")
            self.assertEqual(strings[1], "プレーン文字列")

            # Now translate a sheet referencing this rich-text shared string
            sheet_xml = (
                '<worksheet><sheetData>'
                '<row r="1"><c r="A1" t="s"><v>0</v></c></row>'
                '</sheetData></worksheet>'
            )
            stats, prog = self._init_stats_prog(1)
            result = self.xlsx_handler._process_sheet_xml(
                sheet_xml=sheet_xml,
                sheet_name="Sheet1",
                shared_strings=strings,
                direction="ja2en",
                review_log_path=None,
                stats=stats,
                progress_state=prog,
                progress_cb=None,
                log_cb=None,
            )

            self.assertEqual(stats["translated"], 1)
            self.assertIn('<c r="A1" t="inlineStr">', result)
            self.assertIn('[TRANS: 東京特許許可局]', result)
        finally:
            if os.path.exists(tmp_sst):
                os.remove(tmp_sst)

    # ─────────────────────────────────────────────────────────────
    # Case 9: PPTX <a:fld> blocks protected and restored
    # ─────────────────────────────────────────────────────────────
    def test_case_9_pptx_fld_blocks_protected_and_restored(self):
        """PPTX <a:fld> dynamic field blocks are protected from mutation and restored intact."""
        xml_input = (
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:p>'
            '<a:r><a:t>スライド番号: </a:t></a:r>'
            '<a:fld id="{89B59275-538D-4131-A48F-E644E4DC5C12}" type="slidenum">'
            '<a:rPr lang="ja-JP" smtClean="0"/>'
            '<a:pPr/>'
            '<a:t>42</a:t>'
            '</a:fld>'
            '<a:r><a:t> ページ中</a:t></a:r>'
            '</a:p>'
            '</p:sld>'
        )

        stats, prog = self._init_stats_prog(1)
        result = self.pptx_handler._process_slide_xml(
            xml_str=xml_input,
            slide_name="slide1",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            slide_title=None,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        # The field body must remain intact and identical to the original
        expected_fld = (
            '<a:fld id="{89B59275-538D-4131-A48F-E644E4DC5C12}" type="slidenum">'
            '<a:rPr lang="ja-JP" smtClean="0"/>'
            '<a:pPr/>'
            '<a:t>42</a:t>'
            '</a:fld>'
        )
        self.assertIn(expected_fld, result)

        # The text translated should exclude the dynamic field content '42'
        self.assertEqual(self.engine.received_texts, ["スライド番号:  ページ中"])
        self.assertIn('[TRANS: スライド番号:  ページ中]', result)
        self.assertEqual(stats["translated"], 1)

    # ─────────────────────────────────────────────────────────────
    # Additional edge cases: Cancellation and Reversion
    # ─────────────────────────────────────────────────────────────
    def test_cancellation_during_docx_processing(self):
        """Cooperative cancellation event raises TranslatorError(ErrorCode.E09)."""
        cancel_event = threading.Event()
        cancel_event.set()

        xml_input = '<w:p><w:r><w:t>キャンセルテスト</w:t></w:r></w:p>'
        stats, prog = self._init_stats_prog(1)

        with self.assertRaises(TranslatorError) as ctx:
            self.docx_handler._process_xml_content(
                xml_str=xml_input,
                part_name="document.xml",
                direction="ja2en",
                review_log_path=None,
                stats=stats,
                progress_state=prog,
                progress_cb=None,
                log_cb=None,
                cancel_event=cancel_event,
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E09)

    def test_reverted_translation_leaves_paragraph_unmodified(self):
        """When translation is reverted, paragraph content is preserved unmodified and stats['reverted'] increments."""
        reverting_engine = MockXMLTranslationEngine(should_revert=True)
        docx_handler = DOCXHandler(reverting_engine)

        xml_input = '<w:p><w:r><w:t>リバートテスト</w:t></w:r></w:p>'
        stats, prog = self._init_stats_prog(1)
        result = docx_handler._process_xml_content(
            xml_str=xml_input,
            part_name="document.xml",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        self.assertEqual(result, xml_input)
        self.assertEqual(stats["reverted"], 1)
        self.assertEqual(stats["translated"], 0)


if __name__ == "__main__":
    unittest.main()
