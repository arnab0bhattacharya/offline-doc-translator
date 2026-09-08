"""
formats/xml_utils.py
====================
Secure Office Open XML (OOXML) parsing and DOM mutation utilities.
Leverages lxml and defusedxml to provide safe, hardened XML manipulation
free from XXE, entity expansion bombs, and DTD exploits.
"""

import re
import warnings
from typing import Optional, Tuple, Dict, Any, Union

from lxml import etree
with warnings.catch_warnings():
    warnings.simplefilter("ignore", DeprecationWarning)
    try:
        from defusedxml.lxml import check_docinfo
    except ImportError:
        check_docinfo = None

from engine.errors import ErrorCode, TranslatorError


# Standard OOXML Namespaces
NS_WORD = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
NS_WORD14 = "http://schemas.microsoft.com/office/word/2010/wordml"
NS_DRAWING = "http://schemas.openxmlformats.org/drawingml/2006/main"
NS_PRESENTATION = "http://schemas.openxmlformats.org/presentationml/2006/main"
NS_SPREADSHEET = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_RELATIONSHIPS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
NS_XML = "http://www.w3.org/XML/1998/namespace"

OOXML_NAMESPACES: Dict[str, str] = {
    "w": NS_WORD,
    "w14": NS_WORD14,
    "a": NS_DRAWING,
    "p": NS_PRESENTATION,
    "s": NS_SPREADSHEET,
    "r": NS_RELATIONSHIPS,
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


def check_xml_safety(tree: Union[etree._ElementTree, etree._Element]) -> None:
    """
    Validates that parsed XML does not contain prohibited DTD or entity declarations.
    Raises TranslatorError(ErrorCode.E04) if hostile constructs are detected.
    """
    element_tree = tree if isinstance(tree, etree._ElementTree) else tree.getroottree()
    if element_tree is not None:
        if check_docinfo is not None:
            try:
                check_docinfo(element_tree, forbid_dtd=False, forbid_entities=True)
            except Exception as e:
                raise TranslatorError(
                    ErrorCode.E04,
                    detail=f"Security violation in XML: {e} (prohibited DTD or entity declaration)."
                ) from e

        # Fallback inspection on docinfo
        docinfo = element_tree.docinfo
        if docinfo is not None:
            doctype_str = (docinfo.doctype or "").upper()
            if "ENTITY" in doctype_str or "SYSTEM" in doctype_str or "PUBLIC" in doctype_str:
                raise TranslatorError(
                    ErrorCode.E04,
                    detail="Prohibited DTD or external entity detected in XML part."
                )


def parse_xml_safely(
    source: Union[str, bytes],
    recover_on_error: bool = False,
) -> Tuple[etree._Element, bool]:
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
        # Extract inner XML of children
        pieces = []
        for child in elem:
            pieces.append(etree.tostring(child, encoding="unicode"))
        return "".join(pieces)

    return etree.tostring(
        elem,
        encoding="unicode",
        xml_declaration=xml_declaration,
        standalone=standalone if xml_declaration else None,
    )


def mutate_paragraph_text_nodes_lxml(
    p_content: str,
    tag_prefix: str,
    translated_text: str,
) -> Optional[str]:
    """
    Mutates text nodes within OOXML paragraph content using lxml DOM.
    Sets the translated text on the first <prefix:t> node with xml:space="preserve",
    and empties subsequent <prefix:t> nodes. Preserves all other tags, attributes,
    and runs (<prefix:r>, <prefix:rPr>, <prefix:b/>, etc.).

    Returns:
        Mutated paragraph content string, or None if parsing/mutation fails.
    """
    ns_uri = OOXML_NAMESPACES.get(tag_prefix, NS_WORD)
    wrapper_open = f'<_wrap xmlns:{tag_prefix}="{ns_uri}" xmlns:xml="{NS_XML}">'
    wrapper_close = "</_wrap>"
    wrapped = (wrapper_open + p_content + wrapper_close).encode("utf-8")

    parser = get_secure_xml_parser()
    try:
        root = etree.fromstring(wrapped, parser)
        check_xml_safety(root)

        # Locate all <tag_prefix:t> nodes
        t_nodes = root.xpath(f".//{tag_prefix}:t", namespaces=OOXML_NAMESPACES)
        if not t_nodes:
            # Fallback to local-name if prefix differs
            t_nodes = root.xpath(".//*[local-name()='t']")

        if not t_nodes:
            return None

        # DOM mutation: set first text node, mark xml:space="preserve", empty rest
        t_nodes[0].text = translated_text
        t_nodes[0].set(f"{{{NS_XML}}}space", "preserve")
        for t in t_nodes[1:]:
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
            t_body = m.group(2).replace('"', '&quot;').replace("'", '&apos;')
            t_close = m.group(3)
            return t_open + t_body + t_close

        inner = re.sub(
            rf'(<{tag_prefix}:t\b[^>]*>)(.*?)(</{tag_prefix}:t>)',
            t_quote_repl,
            inner,
            flags=re.DOTALL
        )
        return inner

    except Exception:
        return None


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
