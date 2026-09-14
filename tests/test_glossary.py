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
    app.quick_view = MagicMock()
    app.quick_view.persist_glossary = MagicMock()
    app.controller = MagicMock()

    TranslatorApp.cleanup(app)

    assert app.docs_view.persist_glossary.called
    assert app.quick_view.persist_glossary.called
    assert app.controller.shutdown.called


def test_quick_view_auto_load_and_persist(tmp_path, monkeypatch):
    import gui.views.quick_view as qv_mod
    from gui.views.quick_view import QuickView

    glossary_dir = tmp_path / ".offline-translator"
    glossary_dir.mkdir()
    last_file = glossary_dir / "last_glossary.txt"
    last_file.write_text("quick_term -> quick_trans\n", encoding="utf-8")

    monkeypatch.setattr(qv_mod, "GLOSSARY_DIR", str(glossary_dir))
    monkeypatch.setattr(qv_mod, "LAST_GLOSSARY_PATH", str(last_file))

    view = MagicMock(spec=QuickView)
    view.glossary_text = MagicMock()
    view._glossary_open = False

    # 1. Auto-load
    QuickView._auto_load_glossary(view)
    view.glossary_text.delete.assert_called_with("0.0", "end")
    view.glossary_text.insert.assert_called_with("0.0", "quick_term -> quick_trans\n")

    # 2. Sync glossary
    view.glossary_text.get.return_value = "old_content\n"
    QuickView.sync_glossary(view)
    view.glossary_text.insert.assert_called_with("0.0", "quick_term -> quick_trans\n")

    # 3. Persist
    view.glossary_text.get.return_value = "new_quick_term -> new_quick_trans\n"
    QuickView.persist_glossary(view)
    assert "new_quick_term -> new_quick_trans" in last_file.read_text(encoding="utf-8")


def test_quick_view_load_and_save_glossary_file(tmp_path, monkeypatch):
    from tkinter import filedialog

    from gui.views.quick_view import QuickView

    view = MagicMock(spec=QuickView)
    view.glossary_text = MagicMock()
    view.quick_status = MagicMock()
    view._glossary_open = False
    view.persist_glossary = MagicMock()
    view._toggle_glossary = MagicMock()

    sample_file = tmp_path / "quick_glossary.txt"
    sample_file.write_text("gpu -> グラフィックス\n", encoding="utf-8")

    QuickView._load_glossary_file(view, str(sample_file))

    view.glossary_text.delete.assert_called_with("0.0", "end")
    view.glossary_text.insert.assert_called_with("0.0", "gpu -> グラフィックス\n")
    assert view._toggle_glossary.called
    assert view.persist_glossary.called

    # Save
    view.glossary_text.get.return_value = "term -> trans\n"
    out_file = str(tmp_path / "saved_quick.txt")
    with patch.object(filedialog, "asksaveasfilename", return_value=out_file):
        QuickView._save_glossary_file(view)

    assert os.path.isfile(out_file)
    with open(out_file, encoding="utf-8") as f:
        assert "term -> trans" in f.read()


def test_quick_view_drag_and_drop_routing():
    import gui.views.quick_view as qv_mod
    from gui.views.quick_view import QuickView

    view = MagicMock(spec=QuickView)
    view.glossary_drawer = MagicMock()
    view.glossary_text = MagicMock()
    view._load_glossary_file = MagicMock()

    # 1. Dropped txt file routes to _load_glossary_file
    QuickView._on_window_drop(view, ["C:/docs/terms.txt"], 100, 100)
    view._load_glossary_file.assert_called_with("C:/docs/terms.txt")

    # 2. Hit testing in glossary drawer
    with patch.object(qv_mod, "is_point_in_widget", side_effect=lambda w, x, y: w == view.glossary_drawer):
        QuickView._on_window_drop(view, ["C:/docs/data.csv"], 50, 50)
        view._load_glossary_file.assert_called_with("C:/docs/data.csv")


def test_quick_view_worker_gate_check():
    from gui.views.quick_view import QuickView

    view = MagicMock(spec=QuickView)
    view.after = MagicMock(side_effect=lambda delay, fn, *args: fn(*args))
    view._set_quick_result = MagicMock()
    view.quick_translate_btn = MagicMock()

    with patch("gui.views.quick_view.check_ollama_status", return_value=True):
        # Direction ja2en, but text is pure English -> should_translate is False
        QuickView._quick_translate_worker(view, "Hello world", "ja2en", "gemma4:e2b-it-qat")

    view._set_quick_result.assert_called_once()
    args = view._set_quick_result.call_args[0]
    assert args[0] == "Hello world"
    assert "does not require translation" in args[1]


def test_quick_view_worker_glossary_and_number_masking():
    from gui.views.quick_view import QuickView

    view = MagicMock(spec=QuickView)
    view.after = MagicMock(side_effect=lambda delay, fn, *args: fn(*args))
    view._set_quick_result = MagicMock()
    view.quick_translate_btn = MagicMock()
    view.glossary_text = MagicMock()
    view.glossary_text.get.return_value = "AI -> 人工知能\n"

    mock_backend = MagicMock()
    # Masked text will have [[GLOSSARY_A]] for AI and [[N0]] for 42
    # Mock backend returns translation containing the exact placeholders
    mock_backend.translate_single.return_value = ("これは[[GLOSSARY_A]]と[[N0]]です。", 1.25)

    with (
        patch("gui.views.quick_view.check_ollama_status", return_value=True),
        patch("gui.views.quick_view.LLMBackend", return_value=mock_backend),
    ):
        QuickView._quick_translate_worker(view, "This is AI and 42.", "en2ja", "gemma4:e2b-it-qat")

    assert mock_backend.translate_single.called
    kwargs = mock_backend.translate_single.call_args.kwargs
    assert "[[GLOSSARY_A]]" in kwargs["masked_text"]
    assert "[[N0]]" in kwargs["masked_text"]
    assert kwargs["number_map"]["[[GLOSSARY_A]]"] == "人工知能"
    assert kwargs["number_map"]["[[N0]]"] == "42"

    view._set_quick_result.assert_called_once()
    final_text, status = view._set_quick_result.call_args[0]
    assert final_text == "これは人工知能と42です。"
    assert "Translated in 1.2s" in status
    assert "(1 glossary term)" in status


def test_app_switch_tab_glossary_sync():
    from gui.app import TranslatorApp

    app = MagicMock(spec=TranslatorApp)
    app.active_tab = "docs"
    app.tab_frames = {"docs": MagicMock(), "quick": MagicMock()}
    app.nav_buttons = {"docs": MagicMock(), "quick": MagicMock()}
    app.docs_view = MagicMock()
    app.docs_view.persist_glossary = MagicMock()
    app.docs_view.sync_glossary = MagicMock()
    app.quick_view = MagicMock()
    app.quick_view.persist_glossary = MagicMock()
    app.quick_view.sync_glossary = MagicMock()

    # Switch from docs to quick
    TranslatorApp._switch_tab(app, "quick")

    assert app.docs_view.persist_glossary.called
    assert app.quick_view.sync_glossary.called

    # Switch from quick back to docs
    app.docs_view.persist_glossary.reset_mock()
    app.quick_view.sync_glossary.reset_mock()

    TranslatorApp._switch_tab(app, "docs")
    assert app.quick_view.persist_glossary.called
    assert app.docs_view.sync_glossary.called
