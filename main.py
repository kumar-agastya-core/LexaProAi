#!/usr/bin/env python3
"""
main.py
───────
Entry point for the Lexware invoice automation pipeline.

Full pipeline (two steps):

  Step 1 — OCR pre-processing  (pdfs/inbox/ → pdfs/inbox_ocr/)
    Checks every PDF for a searchable text layer. Image-only PDFs are
    converted to searchable PDFs via ocrmypdf (Tesseract, deu+eng) before
    any data extraction is attempted.

  Step 2 — Invoice processing  (pdfs/inbox_ocr/ → pdfs/processed/ or pdfs/failed/)
    Uses processor (text extraction mode — pdfplumber → Claude) for all invoices.
    NEVER sends the raw PDF binary to Claude — text mode uses far fewer tokens.
    Rate-limited to 1 Lexware API request per second, well below the 2 req/sec
    hard limit, so HTTP 429 errors are not a concern.

    For each invoice:
      a. Pre-scan  — check vendor/category in DB (zero API cost)
      b. Claude    — either combined call (extraction + category) or
                     extraction-only call if vendor is cached
      c. Contact   — lookup or create in Lexware
      d. Voucher   — build payload, POST to Lexware, attach PDF
      e. Move      — pdfs/processed/ on success, pdfs/failed/ on error

Usage:
    python main.py                        # Full batch: step 1 + step 2
    python main.py path/to/invoice.pdf    # Single PDF (skips OCR step)
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from config import get_logger
from agent.db import init_db
from agent.sync import hot_sync
from ocr_preprocess import preprocess_inbox
from processor import run_batch, process_invoice

logger = get_logger("main")


def main() -> None:

    if len(sys.argv) > 1:
        # ── Single PDF mode ───────────────────────────────────────────────
        # Skips the OCR step — assumes the PDF is already readable.
        # Useful for testing a specific file or re-processing a failed one.
        pdf_path = Path(sys.argv[1])
        if not pdf_path.exists():
            print(f"Error: PDF not found: {pdf_path}")
            sys.exit(1)

        init_db()
        hot_sync()

        result = process_invoice(pdf_path)

        print(f"\nResult     : {result['status']}")
        if result["voucher_id"]:
            print(f"Voucher ID : {result['voucher_id']}")
        if result["contact_name"]:
            print(f"Vendor     : {result['contact_name']}")
        if result.get("call_type"):
            print(f"Call type  : {result['call_type']}")
        if result["tokens_used"]:
            print(f"Tokens     : {result['tokens_used']}")
        if result["error"]:
            print(f"Error      : {result['error']}")

        sys.exit(0 if result["status"] in ("open", "unchecked", "skipped") else 1)

    else:
        # ── Batch mode ────────────────────────────────────────────────────
        print("\n" + "=" * 58)
        print("  Lexware Invoice Automation")
        print("=" * 58)

        # ── Step 1: OCR pre-processing ────────────────────────────────────
        print("\n[1/2]  OCR pre-processing  (pdfs/inbox/ → pdfs/inbox_ocr/)")

        ocr = preprocess_inbox()

        if ocr["total"] == 0:
            print("       No PDFs found in pdfs/inbox/ — nothing to do.")
            sys.exit(0)

        print(
            f"       total={ocr['total']}  "
            f"already_ocr={ocr['already_ocr']}  "
            f"ocr_applied={ocr['ocr_applied']}  "
            f"ocr_failed={ocr['ocr_failed']}"
        )
        if ocr["ocr_failed"]:
            print(
                f"       WARNING: {ocr['ocr_failed']} PDF(s) could not be OCR'd — "
                "they will be attempted and likely moved to pdfs/failed/"
            )

        # ── Step 2: Invoice processing ────────────────────────────────────
        print("\n[2/2]  Processing invoices  (processor text mode, 1 req/sec)")

        summary = run_batch()

        print("\n" + "=" * 58)
        print(
            f"  Batch complete — "
            f"open={summary['open']}  "
            f"unchecked={summary['unchecked']}  "
            f"failed={summary['failed']}  "
            f"skipped={summary['skipped']}"
        )
        print("=" * 58 + "\n")

        sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
