#!/usr/bin/env python3
"""
ocr_preprocess.py
─────────────────
OCR pre-processing step — runs before the main invoice pipeline.

Scans pdfs/inbox/ for PDF files and ensures every PDF has a searchable
text layer before processor.py tries to extract data from it.

Flow for each PDF:
  1. Check with pdfplumber — does the PDF already have a text layer?
     (≥ 100 characters across all pages counts as "has text")
  2a. Has text  → move directly to pdfs/inbox_ocr/
  2b. No text   → run ocrmypdf (Tesseract backend, German + English)
                  to add a text layer, then write the result to inbox_ocr/
                  and delete the original from inbox/
  3. If OCR fails → move the original file to inbox_ocr/ anyway so the
     main pipeline can still attempt pdfplumber extraction (which may
     yield partial text) and route it to failed/ if truly unusable.

This module is imported and called by main.py before run_batch().
It can also be run standalone:
    python ocr_preprocess.py
"""

import shutil
from pathlib import Path

import pdfplumber

from config import cfg, get_logger

logger = get_logger("ocr_preprocess")

# Minimum extracted characters to consider a page "has text"
_MIN_TEXT_CHARS = 100


def has_text_layer(pdf_path: Path) -> bool:
    """
    Return True if pdfplumber can extract at least _MIN_TEXT_CHARS
    characters of real text from the PDF.

    A low character count (< 100) typically means the PDF is a scanned
    image with no embedded text — it needs OCR before extraction.
    """
    try:
        with pdfplumber.open(str(pdf_path)) as pdf:
            total = sum(len(page.extract_text() or "") for page in pdf.pages)
        result = total >= _MIN_TEXT_CHARS
        logger.debug(
            "%s — %d chars extracted, has_text=%s", pdf_path.name, total, result
        )
        return result
    except Exception as e:
        logger.warning("pdfplumber failed to open %s: %s — assuming no text", pdf_path.name, e)
        return False


def apply_ocr(input_path: Path, output_path: Path) -> bool:
    """
    Use ocrmypdf (Tesseract backend) to add a searchable text layer.

    Options used:
      language="deu+eng"  — German invoices may contain English product names
      skip_text=True      — don't re-OCR pages that already have text
                            (handles mixed PDFs safely)
      progress_bar=False  — suppress the CLI progress bar in batch mode
      deskew=True         — straighten slightly rotated scans for better accuracy

    Returns True on success, False if ocrmypdf raises an exception.
    The caller always moves the file to inbox_ocr/ whether OCR succeeded or not.
    """
    try:
        import ocrmypdf
        ocrmypdf.ocr(
            input_file    = str(input_path),
            output_file   = str(output_path),
            language      = "deu+eng",
            skip_text     = True,
            deskew        = True,
            progress_bar  = False,
        )
        logger.info("OCR applied: %s → %s", input_path.name, output_path.name)
        return True
    except Exception as e:
        logger.error("OCR failed for %s: %s", input_path.name, e)
        return False


def preprocess_inbox() -> dict:
    """
    Process all PDFs in pdfs/inbox/ — add OCR where needed, move to pdfs/inbox_ocr/.

    Returns a summary dict:
      {
        "total":       int,   # PDFs found in inbox/
        "already_ocr": int,   # had text layer, moved directly
        "ocr_applied": int,   # OCR was applied successfully
        "ocr_failed":  int,   # OCR failed (file still moved, may partially work)
      }
    """
    inbox_dir = cfg.PDF_INBOX
    ocr_dir   = cfg.PDF_INBOX_OCR

    inbox_dir.mkdir(parents=True, exist_ok=True)
    ocr_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(inbox_dir.glob("*.pdf"))

    if not pdfs:
        logger.info("No PDFs found in %s", inbox_dir)
        return {"total": 0, "already_ocr": 0, "ocr_applied": 0, "ocr_failed": 0}

    logger.info("OCR pre-check: %d PDF(s) in %s", len(pdfs), inbox_dir)
    summary = {"total": len(pdfs), "already_ocr": 0, "ocr_applied": 0, "ocr_failed": 0}

    for pdf in pdfs:
        dest = ocr_dir / pdf.name

        # Avoid clobbering a file that's already been OCR'd and is waiting
        if dest.exists():
            logger.warning(
                "%s already exists in inbox_ocr/ — skipping (remove it to reprocess)",
                pdf.name,
            )
            summary["already_ocr"] += 1
            continue

        if has_text_layer(pdf):
            # PDF already has a text layer — move it directly
            shutil.move(str(pdf), str(dest))
            logger.info("Text layer present — moved: %s → inbox_ocr/", pdf.name)
            summary["already_ocr"] += 1

        else:
            # No text layer — apply OCR
            logger.info("No text layer — OCR queued: %s", pdf.name)
            ok = apply_ocr(pdf, dest)

            if ok:
                # Remove original; OCR'd copy is in dest
                pdf.unlink()
                summary["ocr_applied"] += 1
            else:
                # OCR failed — move original anyway so processing can continue
                # The invoice will likely end up in pdfs/failed/ if text can't be read,
                # but at least it won't be stuck in inbox/ forever.
                shutil.move(str(pdf), str(dest))
                summary["ocr_failed"] += 1
                logger.warning(
                    "OCR failed — moving original to inbox_ocr/ anyway: %s", pdf.name
                )

    logger.info(
        "OCR complete — total=%d  already_ocr=%d  ocr_applied=%d  ocr_failed=%d",
        summary["total"],
        summary["already_ocr"],
        summary["ocr_applied"],
        summary["ocr_failed"],
    )
    return summary


# ─────────────────────────────────────────────────────────────────────────────
# Standalone entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    from pathlib import Path as _Path

    _ROOT = _Path(__file__).parent
    sys.path.insert(0, str(_ROOT))

    summary = preprocess_inbox()
    print(
        f"\nOCR pre-process complete\n"
        f"  Total found  : {summary['total']}\n"
        f"  Already OCR  : {summary['already_ocr']}\n"
        f"  OCR applied  : {summary['ocr_applied']}\n"
        f"  OCR failed   : {summary['ocr_failed']}\n"
    )
