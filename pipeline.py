#!/usr/bin/env python3
"""
pipeline.py
───────────
Self-contained invoice processing agent.

Scans pdfs/inbox/from-OCR/ for PDF files. For each file:
  1. Sends the PDF directly to Claude AI (no text pre-extraction)
  2. Claude returns structured invoice data as JSON
  3. Looks up the vendor in the local DB — VAT ID first, then IBAN, then fuzzy name
  4. Creates the contact in Lexware if not found, saves UUID to local DB
  5. Resolves posting category from contact history (or falls back to Zu prüfen)
  6. Performs math check on extracted amounts
  7. Builds voucher payload and POSTs to Lexware
  8. Attaches the original PDF to the voucher
  9. Updates the learning DB and writes the audit log
  10. Moves PDF to pdfs/processed/ (success) or pdfs/failed/ (error)

Everything stays in-process — no intermediate files, no data written outside
of the local SQLite DB and the Lexware API.
"""

import base64
import json
import re
import shutil
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic

from config import cfg, get_logger
from agent.db import (
    get_db,
    init_db,
    pdf_hash,
    is_already_processed,
    record_processed_invoice,
    update_category_history,
    update_category_usage_count,
)
from agent.lexware_client import LexwareClient, LexwareAPIError
from agent.sync import hot_sync

logger = get_logger("agent")

# Your company's VAT ID — loaded from .env (OWN_VAT_ID=DE123456789).
# Injected into Claude prompts at import time so it is never returned as vendor.
_OWN_VAT = cfg.OWN_VAT_ID

# ── Inbox path ────────────────────────────────────────────────────────────────
# PDFs land in pdfs/inbox_ocr/ after OCR pre-processing by ocr_preprocess.py
PDF_OCR_INBOX = cfg.PDF_INBOX_OCR

# ── rapidfuzz for fuzzy name matching (optional but recommended) ──────────────
try:
    from rapidfuzz.distance import Levenshtein as _Lev
    def _edit_distance(a: str, b: str) -> int:
        return _Lev.distance(a, b)
except ImportError:
    def _edit_distance(a: str, b: str) -> int:  # type: ignore[misc]
        if a == b:
            return 0
        if a in b or b in a:
            return abs(len(a) - len(b))
        return max(len(a), len(b))


# ─────────────────────────────────────────────────────────────────────────────
# Claude prompts
# ─────────────────────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a German bookkeeping assistant specialising in Lexware.
You are given a PDF invoice (Rechnung). Extract ALL relevant data and return ONLY valid JSON.
No markdown, no explanation, no prose — ONLY the JSON object.
Never return VAT ID "DE322185267" as the vendor — that is our company's own VAT ID. Ignore it completely and find the actual vendor (issuer) VAT ID instead.
Dates must be in YYYY-MM-DD format.
All monetary amounts must be numbers (float), never strings."""

_EXTRACT_PROMPT = """Extract all invoice data from this PDF and return a single JSON object:

{
  "vendor_name": "Full legal company name of the invoice issuer (the sender, NOT the recipient)",
  "iban": "IBAN of the vendor (e.g. DE89370400440532013000) or null",
  "vat_id": "VAT registration ID (Umsatzsteuer-ID) of the vendor (e.g. DE123456789) or null",
  "tax_number": "Steuernummer of the vendor (e.g. 74315/10796) or null",
  "invoice_number": "Invoice number (Rechnungsnummer) as a string or null",
  "invoice_date": "Invoice date in YYYY-MM-DD format or null",
  "due_date": "Payment due date in YYYY-MM-DD format or null",
  "total_gross": 0.00,
  "total_tax": 0.00,
  "tax_items": [
    {"rate": 19.0, "net": 0.00, "tax": 0.00, "gross": 0.00}
  ],
  "tax_type": "gross",
  "category_suggestion": "Brief description of what this invoice is for"
}

Field rules:
- vendor_name: the company or person who ISSUED (sent) the invoice, not the recipient
- vat_id: the ISSUER's VAT ID. If you see DE322185267, IGNORE it — it is ours. Return null if no other VAT ID is present.
- tax_items: one entry per tax rate bucket. If all items are at the same rate, one entry. If multiple rates, one entry per rate.
- tax_type options:
    "gross"                    — normal German invoice with VAT included
    "net"                      — amounts shown net, tax added on top
    "vatfree"                  — Kleinunternehmer or steuerfrei
    "constructionService13b"   — §13b Bauleistungen
    "externalService13b"       — §13b Fremdleistungen
    "intraCommunitySupply"     — innergemeinschaftliche Lieferung
- All amounts must be numeric floats, not strings."""

# Patch prompts with the configured OWN_VAT_ID (replaces hardcoded placeholder)
if _OWN_VAT:
    for _p in ("_SYSTEM_PROMPT", "_EXTRACT_PROMPT"):
        globals()[_p] = globals()[_p].replace("DE322185267", _OWN_VAT)
    del _p


# ─────────────────────────────────────────────────────────────────────────────
# Data structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TaxItem:
    rate:  float
    net:   float
    tax:   float
    gross: float


@dataclass
class InvoiceData:
    vendor_name:         Optional[str]
    iban:                Optional[str]
    vat_id:              Optional[str]
    tax_number:          Optional[str]
    invoice_number:      Optional[str]
    invoice_date:        Optional[str]
    due_date:            Optional[str]
    total_gross:         Optional[float]
    total_tax:           Optional[float]
    tax_items:           list = field(default_factory=list)
    tax_type:            str  = "gross"
    category_suggestion: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Extract invoice data by sending PDF directly to Claude
# ─────────────────────────────────────────────────────────────────────────────

def extract_invoice_via_claude(pdf_path: Path) -> tuple[InvoiceData, int]:
    """
    Send the PDF binary directly to Claude via the documents API.
    Returns (InvoiceData, tokens_used).

    No text pre-extraction — Claude reads the raw PDF and returns
    structured JSON with all fields needed to create a voucher.
    """
    pdf_bytes = pdf_path.read_bytes()
    pdf_b64   = base64.standard_b64encode(pdf_bytes).decode("utf-8")

    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    response = client.messages.create(
        model      = cfg.CLAUDE_MODEL,
        max_tokens = 1500,
        system     = _SYSTEM_PROMPT,
        messages   = [{
            "role":    "user",
            "content": [
                {
                    "type":   "document",
                    "source": {
                        "type":       "base64",
                        "media_type": "application/pdf",
                        "data":       pdf_b64,
                    },
                },
                {
                    "type": "text",
                    "text": _EXTRACT_PROMPT,
                },
            ],
        }],
    )

    tokens = response.usage.input_tokens + response.usage.output_tokens
    raw    = response.content[0].text

    # Strip markdown code fences if Claude wrapped its JSON
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$",          "", cleaned.strip())

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Claude returned invalid JSON for %s — raw: %s", pdf_path.name, raw[:300])
        return InvoiceData(
            vendor_name=None, iban=None, vat_id=None, tax_number=None,
            invoice_number=None, invoice_date=None, due_date=None,
            total_gross=None, total_tax=None,
        ), tokens

    # Parse tax_items list
    tax_items: list[TaxItem] = []
    for item in data.get("tax_items") or []:
        try:
            tax_items.append(TaxItem(
                rate  = float(item.get("rate",  0)),
                net   = float(item.get("net",   0)),
                tax   = float(item.get("tax",   0)),
                gross = float(item.get("gross", 0)),
            ))
        except (TypeError, ValueError) as e:
            logger.warning("Could not parse tax item %s: %s", item, e)

    inv = InvoiceData(
        vendor_name         = data.get("vendor_name"),
        iban                = data.get("iban"),
        vat_id              = data.get("vat_id"),
        tax_number          = data.get("tax_number"),
        invoice_number      = data.get("invoice_number"),
        invoice_date        = data.get("invoice_date"),
        due_date            = data.get("due_date"),
        total_gross         = _safe_float(data.get("total_gross")),
        total_tax           = _safe_float(data.get("total_tax")),
        tax_items           = tax_items,
        tax_type            = data.get("tax_type", "gross"),
        category_suggestion = data.get("category_suggestion"),
    )

    logger.info(
        "Claude extracted: vendor=%r  invoice=%r  date=%r  gross=%.2f  tokens=%d",
        inv.vendor_name, inv.invoice_number, inv.invoice_date,
        inv.total_gross or 0.0, tokens,
    )
    return inv, tokens


def _safe_float(v) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Look up contact in local DB
# ─────────────────────────────────────────────────────────────────────────────

_LEGAL_SUFFIXES = re.compile(
    r'\b(?:GmbH(?:\s*&\s*Co\.?\s*KG)?|UG|AG|KG|OHG|GbR'
    r'|e\.?\s*K\.?|Einzelunternehmen|Ltd\.?|S\.A\.|B\.V\.)',
    re.IGNORECASE,
)


def _norm_name(s: str) -> str:
    """Normalise company name for fuzzy comparison."""
    if not s:
        return ""
    s = s.lower().strip()
    s = _LEGAL_SUFFIXES.sub("", s)
    s = re.sub(r"[^\w\s]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _norm_vat(s: str) -> str:
    return re.sub(r"\s", "", s).upper() if s else ""


def lookup_contact(inv: InvoiceData) -> Optional[str]:
    """
    Look up the vendor in the local contacts DB.
    Returns the Lexware contact UUID if found, else None.

    Priority order (highest confidence first):
      1. VAT ID exact match   — legally unique identifier
      2. IBAN exact match     — globally unique identifier
      3. Company name fuzzy   — normalised, ≤2 Levenshtein edits, score ≥ 60
    """
    inv_vat  = _norm_vat(inv.vat_id or "")
    inv_iban = re.sub(r"\s", "", inv.iban or "").upper()
    inv_name = _norm_name(inv.vendor_name or "")

    with get_db() as db:

        # ── 1. VAT ID exact match ─────────────────────────────────────────
        if inv_vat:
            row = db.execute(
                "SELECT id FROM contacts WHERE UPPER(REPLACE(vat_id,' ','')) = ?",
                (inv_vat,),
            ).fetchone()
            if row:
                logger.info("Contact matched by VAT ID: %s", row["id"])
                return row["id"]

        # ── 2. IBAN exact match ───────────────────────────────────────────
        if inv_iban:
            row = db.execute(
                "SELECT id FROM contacts WHERE UPPER(REPLACE(iban,' ','')) = ?",
                (inv_iban,),
            ).fetchone()
            if row:
                logger.info("Contact matched by IBAN: %s", row["id"])
                return row["id"]

        # ── 3. Fuzzy company name match ───────────────────────────────────
        if inv_name:
            rows = db.execute(
                "SELECT id, name FROM contacts WHERE role_vendor = 1"
            ).fetchall()

            best_id    = None
            best_score = 0

            for row in rows:
                n = _norm_name(row["name"] or "")
                if not n:
                    continue

                dist = _edit_distance(inv_name, n)
                if dist == 0:
                    score = 80
                elif dist <= 2:
                    score = 60
                elif dist <= 5 and len(inv_name) > 8:
                    # Token-level containment for longer names
                    inv_tok = set(inv_name.split())
                    c_tok   = set(n.split())
                    common  = inv_tok & c_tok
                    if len(common) >= 2 and len(common) / max(len(inv_tok), 1) >= 0.6:
                        score = 40
                    else:
                        continue
                else:
                    continue

                if score > best_score:
                    best_score = score
                    best_id    = row["id"]

            if best_id and best_score >= 60:
                logger.info("Contact matched by name (score=%d): %s", best_score, best_id)
                return best_id

    logger.info("No contact match found for vendor %r", inv.vendor_name)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Create new contact in Lexware and save to local DB
# ─────────────────────────────────────────────────────────────────────────────

def create_contact(inv: InvoiceData, client: LexwareClient) -> Optional[str]:
    """
    Create a new vendor contact in Lexware.
    Saves the UUID and key fields to the local DB so future invoices find it.
    Returns the new contact UUID, or None on failure.
    """
    if not inv.vendor_name:
        logger.warning("Cannot create contact — no vendor name extracted from PDF")
        return None

    payload: dict = {
        "version": 0,
        "roles":   {"vendor": {}},
        "company": {
            "name": inv.vendor_name,
        },
    }

    if inv.vat_id:
        payload["company"]["vatRegistrationId"] = inv.vat_id
    if inv.tax_number:
        payload["company"]["taxNumber"] = inv.tax_number

    try:
        response   = client.create_contact(payload)
        contact_id = response["id"]
    except LexwareAPIError as e:
        logger.error("Failed to create Lexware contact for %r: %s", inv.vendor_name, e)
        return None

    # Save to local DB immediately so the next invoice from the same vendor
    # is matched without another API call
    now = datetime.now(timezone.utc).isoformat()
    with get_db() as db:
        db.execute("""
            INSERT OR REPLACE INTO contacts
                (id, name, iban, vat_id, tax_number,
                 role_vendor, last_synced_at, raw_json)
            VALUES (?, ?, ?, ?, ?, 1, ?, ?)
        """, (
            contact_id,
            inv.vendor_name,
            inv.iban,
            inv.vat_id,
            inv.tax_number,
            now,
            json.dumps({
                "company_name": inv.vendor_name,
                "iban":         inv.iban,
                "vat_id":       inv.vat_id,
                "tax_number":   inv.tax_number,
            }),
        ))

    logger.info("Created new contact: %r → id=%s", inv.vendor_name, contact_id)
    return contact_id


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Resolve posting category
# ─────────────────────────────────────────────────────────────────────────────

def resolve_category(contact_id: Optional[str]) -> str:
    """
    Return the best posting category UUID for this contact.

    Checks contact_category_history for the most-used category
    for this vendor (purchaseinvoice type). Falls back to the
    Zu prüfen catch-all if no history exists yet.
    """
    if contact_id:
        with get_db() as db:
            row = db.execute("""
                SELECT category_id
                FROM   contact_category_history
                WHERE  contact_id   = ?
                AND    voucher_type = 'purchaseinvoice'
                ORDER  BY usage_count DESC
                LIMIT  1
            """, (contact_id,)).fetchone()
            if row:
                logger.info("Category from contact history: %s", row["category_id"])
                return row["category_id"]

    logger.info("No category history — defaulting to Zu prüfen (%s)", cfg.ZU_PRUEFEN_CATEGORY_ID)
    return cfg.ZU_PRUEFEN_CATEGORY_ID


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Math validation
# ─────────────────────────────────────────────────────────────────────────────

def math_check(inv: InvoiceData) -> tuple[bool, str]:
    """
    Verify that extracted tax items sum to the invoice totals.
    Tolerance: ±€0.05 to allow for rounding differences.
    Returns (passed, failure_reason).
    """
    if not inv.tax_items:
        return False, "no tax items extracted from PDF"

    calc_gross = round(sum(i.gross for i in inv.tax_items), 2)
    calc_tax   = round(sum(i.tax   for i in inv.tax_items), 2)

    if inv.total_gross is not None:
        diff = abs(calc_gross - inv.total_gross)
        if diff > 0.05:
            return False, (
                f"gross mismatch: items sum to €{calc_gross:.2f} "
                f"but invoice total is €{inv.total_gross:.2f} (diff €{diff:.2f})"
            )

    if inv.total_tax is not None and inv.total_tax > 0:
        diff = abs(calc_tax - inv.total_tax)
        if diff > 0.05:
            return False, (
                f"tax mismatch: items sum to €{calc_tax:.2f} "
                f"but invoice tax total is €{inv.total_tax:.2f} (diff €{diff:.2f})"
            )

    return True, ""


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Build voucher payload, POST to Lexware, attach PDF
# ─────────────────────────────────────────────────────────────────────────────

def build_and_post_voucher(
    inv:         InvoiceData,
    contact_id:  Optional[str],
    category_id: str,
    math_ok:     bool,
    math_reason: str,
    pdf_path:    Path,
    client:      LexwareClient,
) -> tuple[Optional[str], str]:
    """
    Assemble the Lexware voucher payload and POST it.

    Decision logic:
      - Math check failed → unchecked (let bookkeeper review)
      - No contact found  → unchecked (collective contact)
      - Zu prüfen cat     → unchecked (category not yet known)
      - All good          → open

    After posting, attaches the original PDF.
    Returns (voucher_id, status).
    """
    go_unchecked  = not math_ok or not contact_id or category_id == cfg.ZU_PRUEFEN_CATEGORY_ID
    remark_parts: list[str] = []

    if not math_ok:
        remark_parts.append(f"Math check failed: {math_reason}")
    if not contact_id:
        remark_parts.append("No vendor contact resolved — posted to collective contact")
    if category_id == cfg.ZU_PRUEFEN_CATEGORY_ID:
        remark_parts.append("Category not yet known — please assign manually")

    if go_unchecked:
        # ── Unchecked payload — minimal, safe ────────────────────────────
        payload: dict = {
            "type":          "purchaseinvoice",
            "voucherStatus": "unchecked",
            "taxType":       inv.tax_type or "gross",
            "remark":        "Zu prüfen — " + "; ".join(remark_parts),
        }
        if inv.invoice_date:
            payload["voucherDate"] = inv.invoice_date
        if inv.invoice_number and any(c.isdigit() for c in inv.invoice_number):
            payload["voucherNumber"] = inv.invoice_number
        # Only include amounts if they are internally consistent
        if (inv.total_gross and inv.total_gross > 0
                and inv.total_tax is not None
                and 0 <= inv.total_tax < inv.total_gross
                and inv.total_tax / inv.total_gross <= 0.25):
            payload["totalGrossAmount"] = inv.total_gross
            payload["totalTaxAmount"]   = inv.total_tax
        if contact_id:
            payload["contactId"]           = contact_id
            payload["useCollectiveContact"] = False
        else:
            payload["useCollectiveContact"] = True
        status = "unchecked"

    else:
        # ── Open payload — full data ──────────────────────────────────────
        # Build voucherItems (one per tax rate)
        items = []
        for item in inv.tax_items:
            # For single-item invoices use the invoice total to avoid rounding drift
            amount = inv.total_gross if len(inv.tax_items) == 1 and inv.total_gross else item.gross
            items.append({
                "amount":         round(amount, 2),
                "taxAmount":      round(item.tax, 2),
                "taxRatePercent": int(item.rate) if item.rate == int(item.rate) else item.rate,
                "categoryId":     category_id,
            })

        payload = {
            "type":             "purchaseinvoice",
            "voucherStatus":    "open",
            "taxType":          inv.tax_type or "gross",
            "totalGrossAmount": inv.total_gross,
            "totalTaxAmount":   inv.total_tax,
            "voucherItems":     items,
            "contactId":        contact_id,
            "useCollectiveContact": False,
        }
        if inv.invoice_number:
            payload["voucherNumber"] = inv.invoice_number
        if inv.invoice_date:
            payload["voucherDate"] = inv.invoice_date
        if inv.due_date:
            payload["dueDate"] = inv.due_date
        if remark_parts:
            payload["remark"] = "; ".join(remark_parts)
        status = "open"

    logger.info("Posting voucher as %s for %r", status.upper(), inv.vendor_name)

    # ── POST voucher ──────────────────────────────────────────────────────
    try:
        resp       = client.create_voucher(payload)
        voucher_id = resp["id"]
        logger.info("Voucher created: id=%s  status=%s", voucher_id, status)
    except LexwareAPIError as e:
        logger.error("Failed to POST voucher: %s\nPayload: %s", e, payload)
        return None, "failed"
    except Exception as e:
        logger.error("Unexpected error POSTing voucher: %s", e)
        return None, "failed"

    # ── Attach original PDF to the voucher ────────────────────────────────
    try:
        client.attach_pdf(voucher_id, str(pdf_path))
        logger.info("PDF attached to voucher %s", voucher_id)
    except LexwareAPIError as e:
        # Non-fatal — the voucher exists, bookkeeper can attach manually
        logger.warning("PDF attachment failed for voucher %s (non-fatal): %s", voucher_id, e)

    return voucher_id, status


# ─────────────────────────────────────────────────────────────────────────────
# Main per-invoice pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_invoice(pdf_path: Path) -> dict:
    """
    Process a single PDF invoice through the complete pipeline.

    Returns a result dict:
        status       — "open" | "unchecked" | "skipped" | "failed"
        voucher_id   — Lexware UUID if posted, else None
        contact_id   — vendor contact UUID, or None
        contact_name — vendor name as extracted by Claude
        category_id  — posting category UUID used
        tokens_used  — total Anthropic tokens consumed
        error        — error message if failed, else None
    """
    now    = datetime.now(timezone.utc).isoformat()
    result = {
        "status":       "failed",
        "voucher_id":   None,
        "contact_id":   None,
        "contact_name": None,
        "category_id":  None,
        "tokens_used":  0,
        "error":        None,
    }

    logger.info("=" * 60)
    logger.info("Processing: %s", pdf_path.name)
    logger.info("=" * 60)

    # ── Duplicate guard ───────────────────────────────────────────────────
    try:
        file_hash = pdf_hash(pdf_path)
    except FileNotFoundError:
        result["error"] = f"PDF not found: {pdf_path}"
        logger.error(result["error"])
        return result

    if is_already_processed(file_hash):
        logger.info("SKIPPED — already processed (hash %.12s…)", file_hash)
        result["status"] = "skipped"
        return result

    lx_client = LexwareClient()

    # ── Step 1: Send PDF to Claude ────────────────────────────────────────
    try:
        inv, tokens = extract_invoice_via_claude(pdf_path)
        result["tokens_used"]  = tokens
        result["contact_name"] = inv.vendor_name
    except Exception as e:
        result["error"] = f"Claude extraction failed: {e}"
        logger.error(result["error"])
        _move_pdf(pdf_path, cfg.PDF_FAILED)
        _write_audit(result, pdf_path.name, file_hash, now)
        return result

    # ── Step 2: Look up contact ───────────────────────────────────────────
    contact_id = lookup_contact(inv)
    result["contact_id"] = contact_id

    # ── Step 3: Create contact if not in DB ──────────────────────────────
    if not contact_id:
        contact_id          = create_contact(inv, lx_client)
        result["contact_id"] = contact_id

    # ── Step 4: Resolve posting category ─────────────────────────────────
    category_id          = resolve_category(contact_id)
    result["category_id"] = category_id

    # ── Step 5: Math check ────────────────────────────────────────────────
    math_ok, math_reason = math_check(inv)
    if not math_ok:
        logger.warning("Math check failed: %s", math_reason)

    # ── Step 6: Build payload, POST voucher, attach PDF ───────────────────
    try:
        voucher_id, status = build_and_post_voucher(
            inv         = inv,
            contact_id  = contact_id,
            category_id = category_id,
            math_ok     = math_ok,
            math_reason = math_reason,
            pdf_path    = pdf_path,
            client      = lx_client,
        )
    except Exception as e:
        result["error"] = f"Voucher posting failed: {e}"
        logger.error(result["error"])
        status     = "failed"
        voucher_id = None

    result["status"]     = status
    result["voucher_id"] = voucher_id

    # ── Step 7: Update learning DB ────────────────────────────────────────
    if (
        status in ("open", "unchecked")
        and contact_id
        and category_id
        and category_id != cfg.ZU_PRUEFEN_CATEGORY_ID
    ):
        try:
            update_category_history(
                contact_id   = contact_id,
                category_id  = category_id,
                voucher_type = "purchaseinvoice",
                tax_type     = inv.tax_type or "gross",
                used_at      = now[:10],
            )
            update_category_usage_count(category_id)
            logger.info(
                "Learning DB updated: contact=%s → category=%s",
                contact_id, category_id,
            )
        except Exception as e:
            logger.warning("Failed to update learning DB: %s", e)

    # ── Step 8: Write audit log ───────────────────────────────────────────
    _write_audit(result, pdf_path.name, file_hash, now)

    # ── Step 9: Move PDF ──────────────────────────────────────────────────
    if status in ("open", "unchecked"):
        _move_pdf(pdf_path, cfg.PDF_PROCESSED)
    else:
        _move_pdf(pdf_path, cfg.PDF_FAILED)

    logger.info(
        "Done: status=%s  voucher=%s  tokens=%d",
        status, voucher_id, result["tokens_used"],
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _move_pdf(pdf_path: Path, destination: Path) -> None:
    """Move PDF to destination directory, avoiding overwrite collisions."""
    try:
        destination.mkdir(parents=True, exist_ok=True)
        dest = destination / pdf_path.name
        if dest.exists():
            ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
            dest = destination / f"{pdf_path.stem}_{ts}{pdf_path.suffix}"
        shutil.move(str(pdf_path), str(dest))
        logger.info("PDF moved → %s", dest)
    except Exception as e:
        logger.warning("Could not move %s: %s", pdf_path.name, e)


def _write_audit(result: dict, filename: str, file_hash: str, processed_at: str) -> None:
    """Write result to the processed_invoices audit log."""
    try:
        record_processed_invoice(
            pdf_filename   = filename,
            pdf_hash_val   = file_hash,
            voucher_id     = result.get("voucher_id"),
            voucher_status = result.get("status", "failed"),
            contact_id     = result.get("contact_id"),
            contact_name   = result.get("contact_name"),
            category_id    = result.get("category_id"),
            confidence     = 1.0 if result.get("status") == "open" else 0.5,
            match_signals  = [],
            claude_called  = True,
            claude_tokens  = result.get("tokens_used", 0),
            error_message  = result.get("error"),
            processed_at   = processed_at,
        )
    except Exception as e:
        logger.warning("Failed to write audit record: %s", e)


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner — called by main.py
# ─────────────────────────────────────────────────────────────────────────────

def run_batch() -> dict:
    """
    Process all PDFs in pdfs/inbox/from-OCR/.

    Runs a hot sync first to pull any new contacts or categories from Lexware.
    Returns a summary dict: {open, unchecked, failed, skipped}.
    """
    init_db()
    hot_sync()

    PDF_OCR_INBOX.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_OCR_INBOX.glob("*.pdf"))

    if not pdfs:
        logger.info("No PDFs found in %s", PDF_OCR_INBOX)
        return {"open": 0, "unchecked": 0, "failed": 0, "skipped": 0}

    logger.info("Batch: %d PDF(s) queued for processing", len(pdfs))
    summary: dict = {"open": 0, "unchecked": 0, "failed": 0, "skipped": 0}

    for pdf in pdfs:
        try:
            r = process_invoice(pdf)
            s = r.get("status", "failed")
            summary[s] = summary.get(s, 0) + 1
        except Exception as e:
            logger.error("Unexpected error on %s: %s", pdf.name, e)
            summary["failed"] += 1

    logger.info(
        "Batch complete — open=%d  unchecked=%d  failed=%d  skipped=%d",
        summary["open"], summary["unchecked"], summary["failed"], summary["skipped"],
    )
    return summary
