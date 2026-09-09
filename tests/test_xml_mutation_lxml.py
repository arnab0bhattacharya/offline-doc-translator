"""
tests/test_xml_mutation_lxml.py
===============================
Test suite for lxml and defusedxml DOM-based OOXML mutation.
Validates:
- DOM mutation of Word (.docx) runs, preserving formatting properties.
- DOM mutation of PowerPoint (.pptx) text units with dynamic field (<a:fld>) protection.
- DOM parsing and mutation of Excel (.xlsx) shared strings and worksheet cells.
- XXE and malicious DTD / external entity injection blocking.
- XML-illegal character stripping and XML entity preservation under DOM manipulation.
"""

import unittest
import os
import sys
import tempfile
import zipfile
import shutil

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from engine.core import TranslationEngine, TranslationResult
from engine.errors import ErrorCode, TranslatorError
from formats.xml_utils import (
    parse_xml_safely,
    serialize_xml_safely,
    mutate_paragraph_text_nodes_lxml,
    mutate_complete_xml_part_dom,
    create_inline_str_cell_dom,
    check_xml_safety,
    get_secure_xml_parser,
)
from formats.docx_handler import DOCXHandler
from formats.pptx_handler import PPTXHandler
from formats.xlsx_handler import XLSXHandler


class MockDOMTranslationEngine(TranslationEngine):
    """Mock engine that records translation requests and returns formatted strings."""

    def __init__(self):
        super().__init__(model_name="mock_dom_engine")
        self.received_chunks = []

    def translate_chunk(self, text, direction, context=None, location_id="doc", chunk_id="0", **kwargs):
        self.received_chunks.append((text, context, location_id))
        return TranslationResult(f"[TRANS: {text}]", True, False, elapsed=0.01, source_backend="mock")


class TestXMLMutationLXML(unittest.TestCase):

    def setUp(self):
        self.engine = MockDOMTranslationEngine()
        self.test_dir = tempfile.mkdtemp(prefix="test_dom_")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # ─────────────────────────────────────────────────────────────
    # 1. Word (.docx) DOM Mutation & Formatting Preservation
    # ─────────────────────────────────────────────────────────────
    def test_docx_dom_mutation_preserves_runs_and_styles(self):
        """DOM mutation correctly updates the first <w:t> node and preserves all run formatting."""
        p_content = (
            '<w:r><w:rPr><w:b/><w:color w:val="FF0000"/><w:sz w:val="24"/></w:rPr>'
            '<w:t>第1四半期の</w:t></w:r>'
            '<w:r><w:rPr><w:i/></w:rPr><w:t>業績報告書</w:t></w:r>'
        )
        translated = "[TRANS: 第1四半期の業績報告書]"
        mutated = mutate_paragraph_text_nodes_lxml(
            p_content=p_content,
            tag_prefix="w",
            translated_text=translated,
        )

        self.assertIsNotNone(mutated)
        # Verify first <w:t> contains translated text and xml:space="preserve"
        self.assertIn(f'<w:t xml:space="preserve">{translated}</w:t>', mutated)
        # Verify second <w:t> is emptied
        self.assertIn('<w:t></w:t>', mutated)
        # Verify run formatting properties are preserved
        self.assertIn('<w:b/>', mutated)
        self.assertIn('<w:color w:val="FF0000"/>', mutated)
        self.assertIn('<w:sz w:val="24"/>', mutated)
        self.assertIn('<w:i/>', mutated)

    # ─────────────────────────────────────────────────────────────
    # 2. PowerPoint (.pptx) Dynamic Field Protection
    # ─────────────────────────────────────────────────────────────
    def test_pptx_dom_mutation_fld_protection(self):
        """Dynamic fields (<a:fld>) in slides are completely shielded from text mutation."""
        slide_xml = (
            '<p:sld xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
            'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main">'
            '<p:cSld><p:spTree><p:sp><p:txBody>'
            '<a:p>'
            '<a:r><a:t>ページ: </a:t></a:r>'
            '<a:fld id="{11111111-2222-3333-4444-555555555555}" type="slidenum">'
            '<a:rPr lang="ja-JP"/>'
            '<a:t>7</a:t>'
            '</a:fld>'
            '<a:r><a:t> / 全10ページ</a:t></a:r>'
            '</a:p>'
            '</p:txBody></p:sp></p:spTree></p:cSld></p:sld>'
        )

        handler = PPTXHandler(self.engine)
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        prog = {"current": 0, "total": 1}

        processed = handler._process_slide_xml(
            xml_str=slide_xml,
            slide_name="slide1",
            direction="ja2en",
            review_log_path=None,
            stats=stats,
            slide_title=None,
            progress_state=prog,
            progress_cb=None,
            log_cb=None,
        )

        # Field must remain verbatim
        expected_fld = (
            '<a:fld id="{11111111-2222-3333-4444-555555555555}" type="slidenum">'
            '<a:rPr lang="ja-JP"/>'
            '<a:t>7</a:t>'
            '</a:fld>'
        )
        self.assertIn(expected_fld, processed)
        # Translated text must exclude the dynamic field content '7'
        self.assertEqual(self.engine.received_chunks[0][0], "ページ:  / 全10ページ")
        self.assertIn('[TRANS: ページ:  / 全10ページ]', processed)
        self.assertEqual(stats["translated"], 1)

    # ─────────────────────────────────────────────────────────────
    # 3. Excel (.xlsx) Shared Strings & Inline Cell DOM Construction
    # ─────────────────────────────────────────────────────────────
    def test_xlsx_shared_strings_and_inline_mutation_dom(self):
        """Excel shared strings with rich text runs are parsed via DOM, and cells mutate to inlineStr."""
        sst_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<sst xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" count="1" uniqueCount="1">'
            '<si>'
            '<r><rPr><b/></rPr><t>海外</t></r>'
            '<r><t>事業本部</t></r>'
            '</si>'
            '</sst>'
        )
        sst_path = os.path.join(self.test_dir, "sharedStrings.xml")
        with open(sst_path, "w", encoding="utf-8") as f:
            f.write(sst_xml)

        handler = XLSXHandler(self.engine)
        strings = handler._parse_shared_strings(sst_path)
        self.assertEqual(strings, ["海外事業本部"])

        # Construct inlineStr cell via DOM
        cell_xml = create_inline_str_cell_dom(
            ref="B2",
            style_attr=' s="15"',
            translated_text="[TRANS: 海外事業本部]",
        )
        self.assertIn('r="B2"', cell_xml)
        self.assertIn('t="inlineStr"', cell_xml)
        self.assertIn('s="15"', cell_xml)
        self.assertIn('<is><t xml:space="preserve">[TRANS: 海外事業本部]</t></is>', cell_xml)

    # ─────────────────────────────────────────────────────────────
    # 4. XXE & Malicious DTD Injection Neutralization
    # ─────────────────────────────────────────────────────────────
    def test_xxe_and_entity_injection_blocked(self):
        """Documents attempting external entity expansion (XXE) or malicious DTDs are blocked."""
        malicious_xml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE root ['
            '<!ENTITY xxe SYSTEM "file:///c:/windows/win.ini">'
            ']>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body><w:p><w:r><w:t>&xxe;</w:t></w:r></w:p></w:body>'
            '</w:document>'
        )

        with self.assertRaises(TranslatorError) as ctx:
            parse_xml_safely(malicious_xml)
        self.assertEqual(ctx.exception.code, ErrorCode.E04)
        self.assertIn("Security violation in XML", ctx.exception.detail)

    def test_entity_bomb_billion_laughs_neutralized(self):
        """Billion laughs / entity recursion is rejected without memory exhaustion."""
        bomb_xml = (
            '<?xml version="1.0"?>'
            '<!DOCTYPE lolz ['
            '<!ENTITY lol "lol">'
            '<!ENTITY lol2 "&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;&lol;">'
            ']>'
            '<root>&lol2;</root>'
        )

        with self.assertRaises(TranslatorError) as ctx:
            parse_xml_safely(bomb_xml)
        self.assertEqual(ctx.exception.code, ErrorCode.E04)

    # ─────────────────────────────────────────────────────────────
    # 5. Entity Escaping and Safe Serialization
    # ─────────────────────────────────────────────────────────────
    def test_entities_and_quotes_dom_roundtrip(self):
        """Special XML entities (&, <, >, \", ') are safely formatted in DOM mutation."""
        p_content = '<w:r><w:t>テスト</w:t></w:r>'
        special_text = 'A & B < C > "D" \'E\''
        mutated = mutate_paragraph_text_nodes_lxml(
            p_content=p_content,
            tag_prefix="w",
            translated_text=special_text,
        )

        self.assertIn('&amp;', mutated)
        self.assertIn('&lt;', mutated)
        self.assertIn('&gt;', mutated)
        self.assertIn('&quot;', mutated)
        self.assertIn('&apos;', mutated)
        self.assertNotIn('< C >', mutated)

    # ─────────────────────────────────────────────────────────────
    # 6. Fragment Parsing and Safe Wrapper
    # ─────────────────────────────────────────────────────────────
    def test_unnamespaced_fragment_safe_parsing(self):
        """Unnamespaced XML fragments are parsed without crash and serialized without root wrapper."""
        fragment = '<w:p><w:r><w:t>フラグメント</w:t></w:r></w:p>'
        root, was_wrapped = parse_xml_safely(fragment)
        self.assertTrue(was_wrapped)
        serialized = serialize_xml_safely(root, was_wrapped=was_wrapped)
        self.assertIn('フラグメント', serialized)
        self.assertNotIn('<_wrap', serialized)

    # ─────────────────────────────────────────────────────────────
    # 7. Rich Namespaces (w14, w15, mc, a14) & Attribute Preservation
    # ─────────────────────────────────────────────────────────────
    def test_docx_dom_mutation_rich_namespaces_w14_w15_mc(self):
        """Fragments containing w14, w15, mc namespaces parse and mutate properly without fallback."""
        p_content = (
            '<w:pPr>'
            '<w14:paraId w14:val="12345678"/>'
            '<w14:textId w14:val="87654321"/>'
            '</w:pPr>'
            '<mc:AlternateContent>'
            '<mc:Choice Requires="w14">'
            '<w:r><w:t>Choice text</w:t></w:r>'
            '</mc:Choice>'
            '<mc:Fallback>'
            '<w:r><w:t>Fallback text</w:t></w:r>'
            '</mc:Fallback>'
            '</mc:AlternateContent>'
        )
        translated = "[TRANS: Choice text Fallback text]"
        mutated = mutate_paragraph_text_nodes_lxml(
            p_content=p_content,
            tag_prefix="w",
            translated_text=translated,
        )
        self.assertIsNotNone(mutated)
        self.assertIn('w14:paraId', mutated)
        self.assertIn('w14:val="12345678"', mutated)
        self.assertIn(translated, mutated)

    def test_docx_dom_mutation_preserves_t_node_attributes(self):
        """Attributes on <w:t> like custom annotations or xml:space are preserved during DOM mutation."""
        p_content = '<w:r><w:t note="score &gt; 90" xml:space="preserve">オリジナル</w:t></w:r>'
        translated = "Translated"
        mutated = mutate_paragraph_text_nodes_lxml(
            p_content=p_content,
            tag_prefix="w",
            translated_text=translated,
        )
        self.assertIsNotNone(mutated)
        self.assertIn('note="score &gt; 90"', mutated)
        self.assertIn('xml:space="preserve"', mutated)
        self.assertIn(translated, mutated)

    def test_pptx_dom_mutation_drawing14_and_mc(self):
        """PowerPoint fragments containing a14 and mc markup mutate cleanly on the DOM path."""
        p_content = (
            '<a:pPr>'
            '<mc:AlternateContent>'
            '<mc:Choice Requires="a14"/>'
            '</mc:AlternateContent>'
            '</a:pPr>'
            '<a:r><a:t>スライドプレゼンテーション</a:t></a:r>'
        )
        translated = "[TRANS: Slide Presentation]"
        mutated = mutate_paragraph_text_nodes_lxml(
            p_content=p_content,
            tag_prefix="a",
            translated_text=translated,
        )
        self.assertIsNotNone(mutated)
        self.assertIn('mc:AlternateContent', mutated)
        self.assertIn(translated, mutated)

    # ─────────────────────────────────────────────────────────────
    # 8. Malformed XML Handling & Strict Rejection (No Regex Write)
    # ─────────────────────────────────────────────────────────────
    def test_malformed_xml_in_paragraph_raises_e04(self):
        """Malformed XML in paragraph fragment strictly raises TranslatorError(ErrorCode.E04)."""
        malformed_content = '<w:r><w:t>Unclosed tag</w:r>'
        with self.assertRaises(TranslatorError) as ctx:
            mutate_paragraph_text_nodes_lxml(
                p_content=malformed_content,
                tag_prefix="w",
                translated_text="Should fail",
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E04)
        self.assertIn("Malformed XML in paragraph", ctx.exception.detail)

    def test_base_handler_no_regex_write_fallback_on_malformed_xml(self):
        """_translate_and_replace_text_nodes raises TranslatorError(ErrorCode.E04) on malformed XML."""
        handler = DOCXHandler(self.engine)
        malformed_p = '<w:r><w:t>テスト</w:r>'
        stats = {"total": 0, "translated": 0, "reverted": 0, "skipped": 0}
        prog = {"current": 0, "total": 1}

        with self.assertRaises(TranslatorError) as ctx:
            handler._translate_and_replace_text_nodes(
                full_text="テスト",
                t_matches=["テスト"],
                p_content=malformed_p,
                direction="ja2en",
                context=None,
                part_name="document.xml",
                review_log_path=None,
                stats=stats,
                progress_state=prog,
                progress_cb=None,
                log_cb=None,
                tag_prefix="w",
            )
        self.assertEqual(ctx.exception.code, ErrorCode.E04)

    # ─────────────────────────────────────────────────────────────
    # 9. Full Document DOM Part Mutation
    # ─────────────────────────────────────────────────────────────
    def test_mutate_complete_xml_part_dom(self):
        """mutate_complete_xml_part_dom performs full document DOM mutation with XPath."""
        doc_xml = (
            '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
            '<w:body>'
            '<w:p><w:r><w:t>段落 1</w:t></w:r></w:p>'
            '<w:p><w:r><w:t>段落 2</w:t></w:r></w:p>'
            '</w:body>'
            '</w:document>'
        )
        def mock_translate(text, attrs):
            return f"[TRANS: {text}]"

        result = mutate_complete_xml_part_dom(
            xml_str=doc_xml,
            tag_prefix="w",
            paragraph_translator=mock_translate,
        )
        self.assertIn('[TRANS: 段落 1]', result)
        self.assertIn('[TRANS: 段落 2]', result)
        self.assertIn('<?xml version=', result)


if __name__ == "__main__":
    unittest.main()
