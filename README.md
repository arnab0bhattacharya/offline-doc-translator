# Offline Document Translator

A desktop application for translating Japanese ↔ English Office documents **entirely offline** — no cloud API, no subscription, no data leaving your machine.

Supports `.docx`, `.pptx`, and `.xlsx` files with two translation engines: local Machine Translation (Google MADLAD-400 3B via CTranslate2) and local AI Translation (Gemma 4 via Ollama).

> [!WARNING]
> **PDF TRANSLATION IS NOT OPERATIONAL (UNDER DEVELOPMENT)**
> PDF translation is currently under active development and is **not functional**. **Do not attempt to translate PDF files.** Only Word (`.docx`), PowerPoint (`.pptx`), and Excel (`.xlsx`) files are supported for translation.

---

## Features

- **Offline-first** — works without an internet connection once set up
- **Dual engine** — fast Machine Translation (Google MADLAD-400 3B) or high-quality AI Translation (Ollama / Gemma 4)
- **Supported formats** — Word (`.docx`), PowerPoint (`.pptx`), and Excel (`.xlsx`) *(PDF is under development)*
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
git clone https://github.com/arnab0bhattacharya/offline-doc-translator.git
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

- **PDF translation is NOT operational** — PDF translation is currently non-functional and under active development. **Do not try to translate PDFs.** Use Word (`.docx`), PowerPoint (`.pptx`), or Excel (`.xlsx`) documents instead.
- **Japanese ↔ English only** — the translation models and LLM prompts are optimized for this pair.
- **Windows only** — the desktop GUI uses Windows-specific APIs (`AppUserModelID`, `os.startfile`, DPAPI). The CLI and engine core run cross-platform.
- **LLM quality depends on model** — Gemma 4 E2B QAT is recommended for fast, high-quality offline AI translation.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](LICENSE) file for details.

### Third-Party Components & Notices

- **PyMuPDF (`fitz`)**: Licensed under GNU AGPL v3.0 / Artifex Software Inc.
- **Argos Translate**: Licensed under MIT License (models under Creative Commons / Open Data).
- **CustomTkinter**: Licensed under MIT License.
- **CTranslate2**: Licensed under MIT License.
- **lxml**: Licensed under BSD License.
- **defusedxml**: Licensed under Python Software Foundation License.
- **cryptography**: Licensed under Apache-2.0 / BSD.
