# Offline Document Translator

A desktop application for translating Japanese ↔ English Office documents **entirely offline** — no cloud API, no subscription, no data leaving your machine.

Supports `.docx`, `.pptx`, and `.xlsx` files with two translation engines: local Machine Translation (Google MADLAD-400 3B via CTranslate2) and local AI Translation (Gemma 4 via Ollama).

> [!WARNING]
> **PDF TRANSLATION IS NOT OPERATIONAL (UNDER DEVELOPMENT)**
> PDF translation is currently under active development and is **not functional**. **Do not attempt to translate PDF files.** Only Word (`.docx`), PowerPoint (`.pptx`), and Excel (`.xlsx`) files are supported for translation.

---

## Features

- **Offline-first** — works without an internet connection once models are set up
- **Dual engine** — fast Machine Translation (Google MADLAD-400 3B) or high-quality AI Translation (Ollama / Gemma 4)
- **Supported formats** — Word (`.docx`), PowerPoint (`.pptx`), and Excel (`.xlsx`) *(PDF is under development)*
- **Batch translation queue** — stage multiple files with per-document direction and engine overrides, live visual badges, and progress tracking
- **Glossary support** — custom term pairs with file save/load and cross-session persistence
- **Encrypted cache** — machine-bound AES cache so repeated segments translate instantly
- **Review workflow** — reverted segments flagged in `.needs_review.log` with locations
- **Quick Translate tab** — side-by-side text lookup with glossary injection and number masking
- **Hardware-aware** — automatic CPU core scaling and dynamic memory eviction for smooth operation on 8 GB+ RAM systems
- **Structured errors** — every failure has a structured error code, a plain-English title, and a suggested action

---

## Requirements

| Component | Purpose |
| :--- | :--- |
| **Python 3.11+** | Runtime environment |
| **CTranslate2 & SentencePiece** | Offline Machine Translation engine (Google MADLAD-400 3B) |
| **Ollama** *(Optional)* | Local AI server (required only for AI Translation mode) |
| **Gemma 4 E2B QAT** *(Optional)* | Pinned Ollama model (`ollama pull gemma4:e2b-it-qat`) |

---

## Installation

### Option 1 — Windows Installer (Recommended)

Download `OfflineTranslatorSetup.exe` from the [Releases](../../releases) page and run it. The installer handles desktop shortcuts and dependencies automatically.

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

### Model Setup

- **Machine Translation (MADLAD-400 3B)**: Download the model post-install with a single click from the **System & AI** tab inside the app.
- **AI Translation (Gemma 4 via Ollama)**: Install [Ollama](https://ollama.ai) and pull the model:
  ```bash
  ollama pull gemma4:e2b-it-qat
  ```

---

## Usage

### Desktop GUI

```bash
python main.py
```

1. Drop files into the **Documents** tab staging area (or use the folder picker).
2. Choose your translation direction and engine per-document or globally:
   - **⚡ Machine Translation**: Google MADLAD-400 3B (fast, lightweight, offline).
   - **🧠 AI Translation**: Gemma 4 via Ollama (high fidelity, 4096 context window).
3. Add optional custom glossary terms.
4. Click **▶ Start Translation**.

### Command Line Interface (CLI)

```bash
# Translate a single Word document with Machine Translation
python main.py --input report.docx --output report_en.docx --direction ja2en --mode machine_translation

# Translate PowerPoint with custom glossary
python main.py --input spec.pptx --output spec_en.pptx --direction ja2en --glossary terms.txt

# Translate Excel workbook with AI Translation (Gemma 4)
python main.py --input data.xlsx --output data_en.xlsx --direction en2ja --mode ai_translation
```

Glossary file format (`terms.txt`):
```text
# One term per line, delimiter: ->, :, or =
株式会社 -> Corporation
納期 -> Delivery Date
```

---

## Architecture

```text
offline-doc-translator/
├── engine/               # Translation core: backends, cache, queue, security policy
│   ├── core.py           # TranslationEngine: masking, glossary, cache, verification
│   ├── backend_madlad.py # Google MADLAD-400 3B (CTranslate2) MT backend
│   ├── backend_llm.py    # Ollama LLM backend (Gemma 4, 4096 context)
│   ├── cache.py          # JSONFileCache / EncryptedFileCache / NullCache
│   ├── queue_manager.py  # Thread-safe multi-document queue orchestrator
│   ├── security_policy.py# DocumentSecurityPolicy resource limits
│   └── run_job.py        # Shared job executor (CLI + GUI)
├── formats/              # Format handlers: one class per file type
│   ├── base.py           # BaseFormatHandler ABC + shared OOXML helpers
│   ├── docx_handler.py   # Word (.docx) handler
│   ├── pptx_handler.py   # PowerPoint (.pptx) handler
│   ├── xlsx_handler.py   # Excel (.xlsx) handler
│   ├── pdf_handler.py    # PDF handler (under active development)
│   └── xml_utils.py      # Secure lxml DOM mutation utilities
├── gui/                  # CustomTkinter desktop UI (MVC)
│   ├── app.py            # Main application window & sidebar navigation
│   ├── views/            # DocumentsView, QuickView, SystemView
│   ├── controllers/      # TranslationController
│   └── widgets/          # JobRow, StagedFileList
├── tests/                # 260+ automated unit & integration test suites
└── main.py               # Desktop GUI launcher and CLI entry point
```

---

## Running Tests

```bash
# Run full automated test suite
python -m unittest discover tests

# Or with pytest
python -m pytest tests/ -v
```

Static analysis and linting:

```bash
pip install ruff bandit
python lint.py          # Ruff + Bandit check
python lint.py --fix    # Auto-fix safe issues
```

---

## Security & Privacy

- **Zero external calls** during translation — all inference runs strictly locally on your hardware.
- **XXE / entity injection blocked** — hardened `lxml` parser with `resolve_entities=False` and `no_network=True`.
- **Zip-slip & traversal protection** — strict path validation during OOXML extraction.
- **Encrypted cache** — machine-bound AES key (DPAPI-derived on Windows).
- **Privacy-safe review logs** — source text is never logged; only location, chunk ID, and SHA-256 hash.
- **Placeholder integrity** — `Counter`-based multiset checks ensure all masked numbers and tags survive translation unaltered.

---

## Limitations

- **PDF translation is NOT operational** — PDF translation is currently non-functional and under active development. **Do not try to translate PDFs.** Use Word (`.docx`), PowerPoint (`.pptx`), or Excel (`.xlsx`) documents instead.
- **Japanese ↔ English only** — the translation models and LLM prompts are optimized specifically for this language pair.
- **Windows only** — the desktop GUI uses Windows-specific APIs (`AppUserModelID`, `os.startfile`, DPAPI). The CLI and engine core run cross-platform.
- **AI mode requires Ollama** — AI Translation requires Ollama running with `gemma4:e2b-it-qat` installed.

---

## License

This project is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)** — see the [LICENSE](LICENSE) file for details.

### Third-Party Components & Notices

- **Google MADLAD-400**: Apache 2.0 License.
- **CTranslate2**: MIT License.
- **PyMuPDF (`fitz`)**: GNU AGPL v3.0 / Artifex Software Inc.
- **CustomTkinter**: MIT License.
- **lxml**: BSD License.
- **defusedxml**: Python Software Foundation License.
- **cryptography**: Apache-2.0 / BSD.
