# Offline Document Translator

A desktop application for translating Japanese ↔ English Office documents **entirely offline** — no cloud API, no subscription, no data leaving your machine.

Supports `.docx`, `.pptx`, `.xlsx`, and `.pdf` files with two translation engines: a local neural machine translation (NMT) model via Argos Translate, and a local LLM via Ollama.

---

## Features

- **Offline-first** — works without an internet connection once set up
- **Dual engine** — fast NMT (Argos/CTranslate2) or high-quality LLM (Ollama/Gemma)
- **Four formats** — Word, PowerPoint, Excel, and PDF
- **Batch translation** — stage multiple files, queue and monitor all jobs
- **Glossary support** — custom term pairs with file save/load and cross-session persistence
- **Encrypted cache** — machine-bound AES cache so repeated segments translate instantly
- **Review workflow** — reverted segments flagged in `.needs_review.log` with locations
- **Quick Translate tab** — side-by-side text lookup with glossary and number masking
- **Structured errors** — every failure has a code, a plain-English title, and a suggested action

---

## Requirements

| Component | Purpose |
|-----------|---------|
| **Python 3.11+** | Runtime |
| **Argos Translate** | Offline NMT engine (ja↔en) |
| **Ollama** | Local LLM server (optional, for LLM mode) |
| A Gemma model via Ollama | e.g. `ollama pull gemma4:e2b-it-qat` |

---

## Installation

### Option 1 — Windows Installer (Recommended)

Download `OfflineTranslatorSetup.exe` from the [Releases](../../releases) page and run it. The installer handles Python dependencies automatically.

### Option 2 — From Source

```bash
git clone https://github.com/YOUR_USERNAME/offline-doc-translator.git
cd offline-doc-translator

pip install -r requirements.txt

# Launch the GUI
python main.py

# Or use the CLI
python main.py --help
```

Install Argos language models from the **System & AI** tab inside the app, or via the CLI:

```bash
python main.py --install-argos ja en
python main.py --install-argos en ja
```

---

## Usage

### GUI

```bash
python main.py
```

1. Drop files into the **Documents** tab staging area (or use the folder picker)
2. Choose engine: **⚡ Fast NMT** (offline, no Ollama needed) or **🧠 Pure LLM** (Ollama)
3. Select translation direction and optional custom glossary
4. Click **▶ Start Translation**

### CLI

```bash
# Translate a single file
python main.py --input report.docx --output report_en.docx --direction ja2en

# With a glossary file
python main.py --input spec.pptx --output spec_en.pptx --direction ja2en --glossary terms.txt

# Use LLM engine
python main.py --input memo.docx --output memo_en.docx --direction ja2en --mode pure_llm --model gemma4:e2b-it-qat
```

Glossary file format (`terms.txt`):
```
# One term per line, delimiter: ->, :, or =
株式会社 -> Corporation
納期 -> Delivery Date
```

---

## Architecture

```
offline-doc-translator/
├── engine/          # Translation core: backends, cache, queue, security policy
│   ├── core.py      # TranslationEngine: masking, glossary, cache, verification
│   ├── backend_nmt.py   # Argos/CTranslate2 NMT backend
│   ├── backend_llm.py   # Ollama LLM backend (2-stage retry)
│   ├── cache.py     # JSONFileCache / EncryptedFileCache / NullCache
│   └── run_job.py   # Shared job executor (CLI + GUI)
├── formats/         # Format handlers: one class per file type
│   ├── base.py      # BaseFormatHandler ABC + shared OOXML helpers
│   ├── docx_handler.py
│   ├── pptx_handler.py
│   ├── xlsx_handler.py
│   ├── pdf_handler.py
│   └── xml_utils.py # Secure lxml DOM mutation utilities
├── gui/             # CustomTkinter desktop UI (MVC)
│   ├── app.py       # Thin shell + sidebar navigation
│   ├── views/       # DocumentsView, QuickView, SystemView
│   ├── controllers/ # TranslationController
│   └── widgets/     # JobRow, StagedFileList
├── tests/           # 240+ unit tests (pytest)
├── installer/       # InnoSetup installer scripts
└── main.py          # CLI entry point
```

---

## Running Tests

```bash
pip install pytest
python -m pytest tests/ -v
```

Static analysis:

```bash
pip install ruff bandit
python lint.py          # Ruff + Bandit check
python lint.py --fix    # Auto-fix safe issues
```

---

## Security

- **Zero external calls** during translation — all inference is local
- **XXE / entity injection blocked** — lxml parser with `resolve_entities=False`, `no_network=True`
- **Decompression bomb protection** — ZIP extraction enforces size, ratio, and disk-space limits
- **Encrypted cache** — machine-bound Fernet key (DPAPI-derived on Windows)
- **Privacy-safe review logs** — source text is never logged; only location, chunk ID, and SHA-256 hash
- **Placeholder integrity** — `Counter`-based multiset check ensures all masked tokens survive translation

---

## Limitations

- **Scanned PDFs are not supported** — only digitally-born PDFs with selectable text. For scanned documents, run OCR first (e.g. with [ocrmypdf](https://ocrmypdf.readthedocs.io/), NAPS2, or Adobe Acrobat) then translate the resulting PDF.
- **Japanese ↔ English only** — the NMT models and LLM prompts are tuned for this pair. Other languages require different Argos models and prompt updates.
- **Windows only** — the GUI uses Windows-specific APIs (`AppUserModelID`, `os.startfile`, DPAPI). The CLI and engine run cross-platform.
- **LLM quality depends on model** — larger Gemma models produce better translations but require more VRAM/RAM.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
