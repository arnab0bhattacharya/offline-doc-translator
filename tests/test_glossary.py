import os
from unittest.mock import MagicMock, patch

from gui.theme import (
    MAX_GLOSSARY_ENTRIES,
    format_glossary_text,
    parse_glossary_text,
)
from main import parse_cli_glossary


def test_parse_glossary_text_delimiters():
    raw = """
    # Comments should be ignored
    term1 -> trans1
    term2 : trans2
    term3 = trans3
    # Another comment
    invalid line without delimiter
    empty :
    : empty
    """
    glossary = parse_glossary_text(raw)
    assert glossary == {
        "term1": "trans1",
        "term2": "trans2",
        "term3": "trans3",
    }


def test_parse_glossary_text_term_length_limit():
    warnings = []
    normal_term = "hello -> world"
    long_src = f"{'a' * 201} -> valid_tgt"
    long_tgt = f"valid_src -> {'b' * 201}"

    text = f"{normal_term}\n{long_src}\n{long_tgt}"
    glossary = parse_glossary_text(text, on_warning=lambda w: warnings.append(w))

    assert "hello" in glossary
    assert glossary["hello"] == "world"
    assert len(glossary) == 1
    assert len(warnings) == 1
    assert "200 characters" in warnings[0]


def test_parse_glossary_text_entry_count_limit():
    warnings = []
    lines = [f"key{i} -> val{i}" for i in range(MAX_GLOSSARY_ENTRIES + 50)]
    text = "\n".join(lines)

    glossary = parse_glossary_text(text, on_warning=lambda w: warnings.append(w))
    assert len(glossary) == MAX_GLOSSARY_ENTRIES
    assert len(warnings) == 1
    assert "exceeded maximum limit" in warnings[0]


def test_format_glossary_text():
    g = {"src1": "tgt1", "src2": "tgt2"}
    text = format_glossary_text(g)
    assert "# Term -> Translation" in text
    assert "src1 -> tgt1" in text
    assert "src2 -> tgt2" in text

    parsed = parse_glossary_text(text)
    assert parsed == g


def test_parse_cli_glossary(tmp_path):
    # From string
    g = parse_cli_glossary("k1:v1, k2->v2, k3=v3")
    assert g == {"k1": "v1", "k2": "v2", "k3": "v3"}

    # From file
    f = tmp_path / "glossary.txt"
    f.write_text("# header\napple -> ringo\nbanana -> banana", encoding="utf-8")
    g_file = parse_cli_glossary(str(f))
    assert g_file == {"apple": "ringo", "banana": "banana"}

    # Limit check in CLI
    long_k = "x" * 205 + ":val"
    g_long = parse_cli_glossary(long_k)
    assert len(g_long) == 0


def test_documents_view_auto_load_and_persist(tmp_path, monkeypatch):
    import gui.views.documents_view as dv_mod
    from gui.views.documents_view import DocumentsView

    glossary_dir = tmp_path / ".offline-translator"
    glossary_dir.mkdir()
    last_file = glossary_dir / "last_glossary.txt"
    last_file.write_text("persisted_term -> persisted_trans\n", encoding="utf-8")

    monkeypatch.setattr(dv_mod, "GLOSSARY_DIR", str(glossary_dir))
    monkeypatch.setattr(dv_mod, "LAST_GLOSSARY_PATH", str(last_file))

    view = MagicMock(spec=DocumentsView)
    view.glossary_text = MagicMock()
    view._glossary_open = False

    # 1. Auto-load
    DocumentsView._auto_load_glossary(view)
    view.glossary_text.delete.assert_called_with("0.0", "end")
    view.glossary_text.insert.assert_called_with("0.0", "persisted_term -> persisted_trans\n")

    # 2. Persist
    view.glossary_text.get.return_value = "new_term -> new_trans\n"
    DocumentsView.persist_glossary(view)
    assert "new_term -> new_trans" in last_file.read_text(encoding="utf-8")


def test_documents_view_load_glossary_file(tmp_path, monkeypatch):
    from gui.views.documents_view import DocumentsView

    view = MagicMock(spec=DocumentsView)
    view.glossary_text = MagicMock()
    view._glossary_open = False
    view.log = MagicMock()
    view.persist_glossary = MagicMock()
    view._toggle_glossary = MagicMock()

    sample_file = tmp_path / "custom_glossary.txt"
    sample_file.write_text("ai -> 人工知能\nml -> 機械学習\n", encoding="utf-8")

    DocumentsView._load_glossary_file(view, str(sample_file))

    view.glossary_text.delete.assert_called_with("0.0", "end")
    view.glossary_text.insert.assert_called_with("0.0", "ai -> 人工知能\nml -> 機械学習\n")
    # Drawer was collapsed, so toggle should have been called
    assert view._toggle_glossary.called
    assert view.log.called
    assert view.persist_glossary.called


def test_documents_view_save_glossary_file(tmp_path, monkeypatch):
    from tkinter import filedialog

    from gui.views.documents_view import DocumentsView

    view = MagicMock(spec=DocumentsView)
    view.glossary_text = MagicMock()
    view.glossary_text.get.return_value = "term1 -> trans1\n"
    view.log = MagicMock()

    out_file = str(tmp_path / "saved_glossary.txt")
    with patch.object(filedialog, "asksaveasfilename", return_value=out_file):
        DocumentsView._save_glossary_file(view)

    assert os.path.isfile(out_file)
    with open(out_file, encoding="utf-8") as f:
        assert "term1 -> trans1" in f.read()
    assert view.log.called


def test_documents_view_drag_and_drop_routing():
    import gui.views.documents_view as dv_mod
    from gui.views.documents_view import DocumentsView

    view = MagicMock(spec=DocumentsView)
    view.glossary_drawer = MagicMock()
    view.glossary_text = MagicMock()
    view.staged_list = MagicMock()
    view._load_glossary_file = MagicMock()

    # 1. Dropped text file routes to _load_glossary_file
    DocumentsView._on_window_drop(view, ["C:/docs/terms.txt"], 100, 100)
    view._load_glossary_file.assert_called_with("C:/docs/terms.txt")

    # 2. Dropped docx routes to staged_list.add_files
    DocumentsView._on_window_drop(view, ["C:/docs/report.docx"], 100, 100)
    view.staged_list.add_files.assert_called_with(["C:/docs/report.docx"])

    # 3. Hit testing in glossary drawer routes even mixed/other files
    with patch.object(dv_mod, "is_point_in_widget", side_effect=lambda w, x, y: w == view.glossary_drawer):
        DocumentsView._on_window_drop(view, ["C:/docs/glossary_data.csv"], 50, 50)
        view._load_glossary_file.assert_called_with("C:/docs/glossary_data.csv")


def test_app_cleanup_persists_glossary():
    from gui.app import TranslatorApp

    app = MagicMock(spec=TranslatorApp)
    app.docs_view = MagicMock()
    app.docs_view.persist_glossary = MagicMock()
    app.controller = MagicMock()

    TranslatorApp.cleanup(app)

    assert app.docs_view.persist_glossary.called
    assert app.controller.shutdown.called
