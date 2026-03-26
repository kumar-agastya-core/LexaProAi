# LexaProAi — Lexware Invoice Automation

Automatically processes PDF invoices and posts them as vouchers to Lexware.
Extracts invoice data with Claude AI, matches vendors, classifies accounting
categories, and handles the full Lexware API workflow.

---

## How it works

```
pdfs/inbox/         ← drop invoices here
     │
     ▼  ocr_preprocess.py
     │  checks text layer; runs Tesseract OCR on image-only PDFs
     │
pdfs/inbox_ocr/
     │
     ▼  processor.py  (via main.py)
     │  1. pdfplumber extracts plain text
     │  2. regex pre-scan → DB lookup (zero API cost)
     │  3. Claude API — extraction + category in one call
     │     (vendor cached → extraction-only, ~1 500 tokens)
     │     (new vendor   → combined call,    ~5 000 tokens)
     │  4. lookup / create contact in Lexware
     │  5. POST voucher → attach PDF
     │
     ├──► pdfs/processed/   (success — voucher posted as OPEN)
     └──► pdfs/failed/      (error   — review manually)
```

---

## Project structure

```
LexaProAi/
│
├── main.py               Entry point — run this to process all invoices
├── test.py               Dry-run harness — test a PDF without posting anything
│
├── config.py             All settings (reads .env, exposes cfg singleton)
├── ocr_preprocess.py     Step 1: ensure PDFs have a text layer (Tesseract OCR)
├── processor.py          Step 2: extract, classify, post to Lexware  ← MAIN PIPELINE
├── pipeline.py           Shared models + PDF-binary mode (used by test.py)
│
├── agent/
│   ├── db.py             SQLite helpers + schema initialisation
│   ├── lexware_client.py Lexware REST API client (rate-limited: 1 req/sec)
│   └── sync.py           Sync Lexware data to local DB (hot/cold)
│
├── db/
│   ├── schema.sql        Database schema (7 tables — contacts, categories, history…)
│   └── lexware.db        Runtime database (auto-created, git-ignored)
│
├── pdfs/
│   ├── inbox/            ← drop invoices here
│   ├── inbox_ocr/        OCR output — auto-managed by ocr_preprocess.py
│   ├── processed/        Successfully posted invoices
│   └── failed/           Failed invoices — review manually
│
├── logs/
│   └── automation.log    Runtime log (auto-created, git-ignored)
│
├── requirements.txt      Python dependencies
├── .env.example          Template — copy to .env and fill in API keys
└── Makefile              Shortcuts for common tasks (make help)
```

---

## Setup

**Prerequisites**

- Python 3.10+
- Tesseract OCR installed on the system:
  ```bash
  # macOS
  brew install tesseract tesseract-lang

  # Ubuntu / Debian
  apt-get install tesseract-ocr tesseract-ocr-deu
  ```

**Install**

```bash
# 1. Create virtual environment
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure
cp .env.example .env
# Edit .env — fill in LEXWARE_API_KEY and ANTHROPIC_API_KEY

# 4. Initialise DB and sync Lexware data
make setup
```

---

## Usage

**Process all invoices (full batch)**
```bash
python main.py
# or
make batch
```

**Process a single invoice**
```bash
python main.py "pdfs/inbox/Vendor_Invoice.pdf"
```

**Dry-run test (nothing is posted to Lexware)**
```bash
# Text mode — default, fewer tokens
python test.py --text "pdfs/inbox/Vendor_Invoice.pdf"

# PDF-binary mode — for comparison
python test.py "pdfs/inbox/Vendor_Invoice.pdf"

# Prime the vendor cache after verifying the result is correct
python test.py --text --learn "pdfs/inbox/Vendor_Invoice.pdf"
```

**Sync latest contacts/categories from Lexware**
```bash
make sync-hot    # fast delta sync (run daily)
make sync-cold   # full re-sync (after major Lexware data changes)
```

**OCR pre-processing only**
```bash
python ocr_preprocess.py
```

---

## Configuration (.env)

| Variable            | Required | Default                        | Description                             |
|---------------------|----------|--------------------------------|-----------------------------------------|
| `LEXWARE_API_KEY`   | ✓        | —                              | Lexware Bearer token                    |
| `ANTHROPIC_API_KEY` | ✓        | —                              | Claude API key                          |
| `CLAUDE_MODEL`      |          | `claude-haiku-4-5-20251001`    | Claude model to use                     |
| `DB_PATH`           |          | `db/lexware.db`                | SQLite database path                    |
| `PDF_INBOX`         |          | `pdfs/inbox`                   | Drop invoices here                      |
| `PDF_INBOX_OCR`     |          | `pdfs/inbox_ocr`               | OCR output folder                       |
| `PDF_PROCESSED`     |          | `pdfs/processed`               | Successfully posted invoices            |
| `PDF_FAILED`        |          | `pdfs/failed`                  | Failed invoices                         |
| `LOG_PATH`          |          | `logs/automation.log`          | Log file path                           |
| `LOG_LEVEL`         |          | `INFO`                         | `DEBUG`, `INFO`, `WARNING`, `ERROR`     |
| `FORCE_COLD_SYNC`   |          | `false`                        | Set `true` to force full re-sync        |

---

## Token cost

| Scenario                       | Approx. tokens / invoice |
|--------------------------------|--------------------------|
| New vendor — combined call     | ~5 000                   |
| Known vendor — extraction-only | ~1 500                   |
| Old two-call approach          | ~14 000                  |

The vendor cache (`contact_category_history` table) fills automatically
as invoices are posted. Use `python test.py --text --learn` to prime it
during testing without posting to Lexware.

---

## Rate limiting

Lexware API hard limit: **2 requests/second**.
We enforce **1 request/second** (1.1 s gap) — HTTP 429 errors should never occur.
