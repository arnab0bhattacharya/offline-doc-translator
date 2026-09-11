"""
formats/xml_utils.py
====================
Secure Office Open XML (OOXML) parsing and DOM mutation utilities.
Leverages lxml and defusedxml to provide safe, hardened XML manipulation
free from XXE, entity expansion bombs, and DTD exploits.
"""

import re
import warnings
from collections.abc import Callable

from lxml import etree

with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        from defusedxml.lxml import check_docinfo
    except ImportError:
        check_docinfo = None

from engine.errors import ErrorCode, TranslatorError

# Standard OOXML & Microsoft Office Extension Namespaces
NS_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_WORD14 = "http://schemas.microsoft.com/office/word/2010/wordml"
NS_WORD15 = "http://schemas.microsoft.com/office/word/2012/wordml"
NS_WORD16 = "http://schemas.microsoft.com/office/word/2018/wordml"
NS_DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_DRAWING14 = "http://schemas.microsoft.com/office/drawing/2010/main"
NS_PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_SPREADSHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_MARKUP_COMPAT = "http://schemas.openxmlformats.org/markup-compatibility/2006"
NS_XML = "http://www.w3.org/XML/1998/namespace"

OOXML_NAMESPACES: dict[str, str] = {
    # WordprocessingML & Extensions
    "w": NS_WORD,
    "w14": NS_WORD14,
    "w15": NS_WORD15,
    "w16": NS_WORD16,
    "w16cid": "http://schemas.microsoft.com/office/word/2016/wordml/cid",
    "w16se": "http://schemas.microsoft.com/office/word/2015/wordml/symex",
    "w10": "urn:schemas-microsoft-com:office:word",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "wp14": "http://schemas.microsoft.com/office/word/2010/wordprocessingDrawing",
    "wne": "http://schemas.microsoft.com/office/word/2006/wordml",
    # DrawingML & Extensions
    "a": NS_DRAWING,
    "a14": NS_DRAWING14,
    "a15": "http://schemas.microsoft.com/office/drawing/2012/main",
    "a16": "http://schemas.microsoft.com/office/drawing/2014/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
    "c": "http://schemas.openxmlformats.org/drawingml/2006/chart",
    "dgm": "http://schemas.openxmlformats.org/drawingml/2006/diagram",
    # PresentationML & Extensions
    "p": NS_PRESENTATION,
    "p14": "http://schemas.microsoft.com/office/powerpoint/2010/main",
    "p15": "http://schemas.microsoft.com/office/powerpoint/2012/main",
    # SpreadsheetML & Extensions
    "s": NS_SPREADSHEET,
    "x": NS_SPREADSHEET,
    "x14": "http://schemas.microsoft.com/office/spreadsheetml/2009/9/main",
    "x15": "http://schemas.microsoft.com/office/spreadsheetml/2010/11/main",
    # Common, Relationships, VML, Math, Compatibility
    "r": NS_RELATIONSHIPS,
    "r14": "http://schemas.microsoft.com/office/2007/relationships",
    "m": "http://schemas.openxmlformats.org/officeDocument/2006/math",
    "v": "urn:schemas-microsoft-com:vml",
    "mc": NS_MARKUP_COMPAT,
    "o": "urn:schemas-microsoft-com:office:office",
    "xml": NS_XML,
}


def get_secure_xml_parser(recover: bool = False) -> etree.XMLParser:
    """
    Returns a hardened lxml XMLParser that neutralizes XXE and entity attacks:
    - resolve_entities=False (blocks XML entity expansion)
    - no_network=True (blocks fetching external entities or DTDs over network)
    - dtd_validation=False, load_dtd=False (disables loading external DTDs)
    """
    return etree.XMLParser(
        resolve_entities=False,
        no_network=True,
        dtd_validation=False,
        load_dtd=False,
        remove_blank_text=False,
        recover=recover,
    )


def check_xml_safety(tree: etree._ElementTree | etree._Element) -> None:
    """
    Validates that parsed XML does not contain prohibited DTD or entity declarations.
    OOXML parts strictly require no DTD declarations; any DOCTYPE (internal, external,
    or entity-based) is prohibited as defense in depth.
    Raises TranslatorError(ErrorCode.E04) if any DTD or entity declaration is detected.
    """
    element_tree = tree if isinstance(tree, etree._ElementTree) else tree.getroottree()
    if element_tree is not None:
        if check_docinfo is not None:
            try:
                check_docinfo(element_tree, forbid_dtd=True, forbid_entities=True)
            except Exception as e:
                raise TranslatorError(
                    ErrorCode.E04, detail=f"Security violation in XML: {e} (prohibited DTD or entity declaration)."
                ) from e

        # Fallback inspection on docinfo
        docinfo = element_tree.docinfo
        if docinfo is not None:
            doctype_str = (docinfo.doctype or "").strip()
            if doctype_str:
                raise TranslatorError(
                    ErrorCode.E04,
                    detail=f"Security violation in XML: Prohibited DOCTYPE declaration detected ({doctype_str[:50]}).",
                )


def parse_xml_safely(
    source: str | bytes,
    recover_on_error: bool = False,
) -> tuple[etree._Element, bool]:
    """
    Safely parses an XML string or bytes using the secure parser and verifies safety.
    If the XML is a fragment with undeclared namespace prefixes (e.g. <w:p> without xmlns:w),
    it wraps the fragment in a root element declaring standard OOXML namespaces.

    Returns:
        Tuple of (root_element, was_wrapped: bool).
    """
    parser = get_secure_xml_parser(recover=recover_on_error)
    raw_bytes = source.encode("utf-8") if isinstance(source, str) else source

    try:
        root = etree.fromstring(raw_bytes, parser)
        check_xml_safety(root)
        return root, False
    except etree.XMLSyntaxError as err:
        # Check if error is due to an undeclared namespace prefix
        err_msg = str(err)
        if "Namespace prefix" in err_msg or "not defined" in err_msg:
            ns_decls = " ".join([f'xmlns:{k}="{v}"' for k, v in OOXML_NAMESPACES.items()])
            wrapped = f"<_wrap {ns_decls}>\n".encode("utf-8") + raw_bytes + b"\n</_wrap>"
            try:
                wrapped_root = etree.fromstring(wrapped, parser)
                check_xml_safety(wrapped_root)
                return wrapped_root, True
            except etree.XMLSyntaxError:
                pass
        raise TranslatorError(ErrorCode.E04, detail=f"Corrupted or invalid XML: {err}") from err


def serialize_xml_safely(
    elem: etree._Element,
    was_wrapped: bool = False,
    xml_declaration: bool = False,
    standalone: bool = True,
) -> str:
    """
    Serializes an lxml Element back to an XML string.
    If was_wrapped is True, strips the temporary <_wrap> root container.
    """
    if was_wrapped and elem.tag == "_wrap":
        return "".join(etree.tostring(child, encoding="unicode") for child in elem)

    if xml_declaration:
        raw_bytes = etree.tostring(
            elem,
            encoding="utf-8",
            xml_declaration=True,
            standalone=standalone,
        )
        return raw_bytes.decode("utf-8")

    return etree.tostring(elem, encoding="unicode")


def mutate_paragraph_text_nodes_lxml(
    p_content: str,
    tag_prefix: str,
    translated_text: str,
    extra_nsmap: dict[str, str] | None = None,
) -> str | None:
    """
    Mutates text nodes within OOXML paragraph content using lxml DOM.
    Sets the translated text on the first <prefix:t> node with xml:space="preserve",
    and empties subsequent <prefix:t> nodes. Preserves all other tags, attributes,
    runs (<prefix:r>, <prefix:rPr>, <prefix:b/>, etc.), and extended namespaces (w14, w15, a14).

    Returns:
        Mutated paragraph content string, or None if no text nodes exist (documented no-text result).
    Raises:
        TranslatorError(ErrorCode.E04) if XML is malformed or violates security policies.
    """
    active_nsmap = dict(OOXML_NAMESPACES)
    if extra_nsmap:
        active_nsmap.update({pfx: uri for pfx, uri in extra_nsmap.items() if pfx and uri})

    # Detect any undeclared prefixes in p_content to prevent parser failures on fragments
    for pfx in set(re.findall(r"<([a-zA-Z0-9_\-]+):[a-zA-Z0-9_\-]+", p_content)):
        if pfx not in active_nsmap:
            active_nsmap[pfx] = f"urn:schemas-fallback:{pfx}"
    for pfx in set(re.findall(r"\s([a-zA-Z0-9_\-]+):[a-zA-Z0-9_\-]+=", p_content)):
        if pfx not in active_nsmap:
            active_nsmap[pfx] = f"urn:schemas-fallback:{pfx}"

    ns_decls = " ".join([f'xmlns:{k}="{v}"' for k, v in active_nsmap.items() if k and v])
    wrapper_open = f"<_wrap {ns_decls}>"
    wrapper_close = "</_wrap>"
    wrapped = (wrapper_open + p_content + wrapper_close).encode("utf-8")

    parser = get_secure_xml_parser()
    try:
        root = etree.fromstring(wrapped, parser)
        check_xml_safety(root)
    except TranslatorError:
        raise
    except etree.XMLSyntaxError as err:
        raise TranslatorError(ErrorCode.E04, detail=f"Malformed XML in paragraph: {err}") from err
    except Exception as err:
        raise TranslatorError(ErrorCode.E04, detail=f"Unexpected error parsing paragraph XML: {err}") from err

    # Locate all <tag_prefix:t> nodes
    t_nodes = root.xpath(f".//{tag_prefix}:t", namespaces=active_nsmap)
    if not t_nodes:
        # Fallback to local-name if prefix differs
        t_nodes = root.xpath(".//*[local-name()='t']")

    if not t_nodes:
        return None

    # DOM mutation: set first text node, mark xml:space="preserve", empty rest
    for child in list(t_nodes[0]):
        t_nodes[0].remove(child)
    t_nodes[0].text = translated_text
    t_nodes[0].set(f"{{{NS_XML}}}space", "preserve")
    for t in t_nodes[1:]:
        for child in list(t):
            t.remove(child)
        t.text = ""

    # Serialize DOM
    serialized = etree.tostring(root, encoding="unicode")
    # Extract inner content from <_wrap...>...</_wrap>
    start_pos = serialized.index(">") + 1
    end_pos = serialized.rindex("</_wrap>")
    inner = serialized[start_pos:end_pos]

    # Ensure double and single quotes inside <tag:t> nodes are safely entity-encoded
    # (&quot; and &apos;) for strict conformance with downstream expectations
    def t_quote_repl(m):
        t_open = m.group(1)
        t_body = m.group(2).replace('"', "&quot;").replace("'", "&apos;")
        t_close = m.group(3)
        return t_open + t_body + t_close

    inner = re.sub(rf"(<{tag_prefix}:t\b[^>]*>)(.*?)(</{tag_prefix}:t>)", t_quote_repl, inner, flags=re.DOTALL)
    return inner


def mutate_complete_xml_part_dom(
    xml_str: str,
    tag_prefix: str,
    paragraph_translator: Callable[[str, dict[str, str]], str | None],
) -> str:
    """
    Parses a complete OOXML XML part into an lxml ElementTree, selects paragraphs
    via namespace-aware XPath, mutates text nodes in-place, and serializes safely.
    Validates XML safety (XXE / entity explosion protection) prior to mutation.
    """
    root, was_wrapped = parse_xml_safely(xml_str)

    active_nsmap = dict(OOXML_NAMESPACES)
    if hasattr(root, "nsmap"):
        active_nsmap.update({k: v for k, v in root.nsmap.items() if k and v})

    p_nodes = root.xpath(f".//{tag_prefix}:p", namespaces=active_nsmap)
    for p_elem in p_nodes:
        t_nodes = p_elem.xpath(f".//{tag_prefix}:t", namespaces=active_nsmap)
        if not t_nodes:
            t_nodes = p_elem.xpath(".//*[local-name()='t']")
        if not t_nodes:
            continue

        full_text = "".join(t.text or "" for t in t_nodes)
        if not full_text.strip():
            continue

        translated = paragraph_translator(full_text, dict(p_elem.attrib))
        if translated is not None:
            for child in list(t_nodes[0]):
                t_nodes[0].remove(child)
            t_nodes[0].text = translated
            t_nodes[0].set(f"{{{NS_XML}}}space", "preserve")
            for t in t_nodes[1:]:
                for child in list(t):
                    t.remove(child)
                t.text = ""

    return serialize_xml_safely(root, was_wrapped=was_wrapped, xml_declaration=True)


def create_inline_str_cell_dom(
    ref: str,
    style_attr: str,
    translated_text: str,
) -> str:
    """
    Constructs an ECMA-376 inline string cell (<c t="inlineStr">) using lxml DOM.
    Safeguards XML entities and preserves formatting styles.
    """
    attribs = {"r": ref, "t": "inlineStr"}
    if style_attr:
        m = re.search(r's="([^"]+)"', style_attr)
        if m:
            attribs["s"] = m.group(1)

    cell = etree.Element("c", attrib=attribs)
    is_elem = etree.SubElement(cell, "is")
    t_elem = etree.SubElement(is_elem, "t")
    t_elem.set(f"{{{NS_XML}}}space", "preserve")
    t_elem.text = translated_text
    return etree.tostring(cell, encoding="unicode")
