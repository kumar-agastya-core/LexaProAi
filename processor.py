#!/usr/bin/env python3
"""
processor.py
────────────
Main invoice processing pipeline — text extraction mode.

This is the module used by main.py for all invoice processing.

How it works
────────────
  1. pdfplumber extracts plain text from the PDF (no binary upload to Claude)
  2. Vendor pre-scan: regex finds VAT ID / IBAN in the raw text,
     looked up against the local DB before any Claude API call (zero cost)
  3. Claude API call — routed by cache state:
       cache hit  (vendor + category known) → extraction-only  (~1 500 tokens)
       cache miss (new vendor)              → combined call     (~5 000 tokens)
                                               extraction + category in one response
  4. Contact lookup / create in Lexware
  5. Voucher payload built and POSTed to Lexware
  6. Original PDF attached to the voucher
  7. PDF moved to pdfs/processed/ (success) or pdfs/failed/ (error)
  8. Learning DB updated — next invoice from same vendor hits the cache

Token cost comparison
────────────────────
  Naive two-call approach         : ~14 000 tokens / invoice
  Combined call (new vendor)      : ~5 000 tokens  / invoice
  Extraction-only (cached vendor) : ~1 500 tokens  / invoice

Trade-offs vs pipeline.py (PDF-binary mode)
────────────────────────────────────────────
  ✓ Far fewer tokens (text << base64-encoded PDF binary)
  ✓ Works with any Claude model (document API requires claude-3-5+)
  ✓ Resolves the exact posting category UUID from the live DB
  ✗ Accuracy depends on pdfplumber reading the PDF layout correctly
  ✗ Image-only / scanned PDFs need OCR first — run ocr_preprocess.py

Entry points
────────────
  process_invoice(pdf_path)  — process one PDF, return result dict
  run_batch()                — process all PDFs in pdfs/inbox_ocr/
  extract_and_resolve(path)  — extract + classify in one call (used by test.py)
"""

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import anthropic
import pdfplumber

from config import cfg, get_logger
from agent.db import (
    get_db,
    init_db,
    pdf_hash,
    is_already_processed,
    update_category_history,
    update_category_usage_count,
)
from agent.lexware_client import LexwareClient
from agent.sync import hot_sync

# ── Re-use all shared logic from pipeline.py ─────────────────────────────────
from pipeline import (
    TaxItem,
    InvoiceData,
    lookup_contact,
    create_contact,
    math_check,
    build_and_post_voucher,
    _move_pdf,
    _write_audit,
    _safe_float,
    PDF_OCR_INBOX,
)

logger = get_logger("processor")

# Your company's VAT ID — loaded from .env (OWN_VAT_ID=DE123456789).
# Any invoice that carries this ID is from YOU (the buyer) and must never
# be returned as the vendor during AI extraction.
_OWN_VAT = cfg.OWN_VAT_ID


# ─────────────────────────────────────────────────────────────────────────────
# Result dataclasses
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CategoryResult:
    category_id:   str
    category_name: str
    group_name:    str
    method:        str   # "contact_history" | "claude_semantic" | "fallback"
    tokens_used:   int   = 0


@dataclass
class VendorCache:
    """
    Zero-cost pre-scan result — regex + DB lookup before any API call.

    hit=True means both the contact AND their preferred category are already
    known locally. In that case extract_and_resolve() skips the categories
    list entirely and makes only a cheap extraction-only Claude call.
    """
    vat_id_found:   Optional[str]   # VAT ID found in raw text (not ours)
    iban_found:     Optional[str]   # IBAN found in raw text
    contact_id:     Optional[str]   # matched contact UUID
    contact_name:   Optional[str]   # matched contact display name
    category_id:    Optional[str]   # cached category UUID
    category_name:  Optional[str]   # cached category display name
    group_name:     Optional[str]   # cached category group
    hit:            bool = False    # True → extraction-only call is enough


# ─────────────────────────────────────────────────────────────────────────────
# Claude prompts
# ─────────────────────────────────────────────────────────────────────────────

# ── Extraction-only (used when vendor + category are already cached) ──────────

_SYSTEM_PROMPT = """You are a German bookkeeping assistant specialising in Lexware.
You are given the plain text extracted from a PDF invoice (Rechnung).
Extract ALL relevant data and return ONLY valid JSON — no markdown, no explanation, no prose.
Never return VAT ID "DE322185267" as the vendor — that is our company's own VAT ID. Ignore it and find the actual vendor VAT ID instead.
Dates must be in YYYY-MM-DD format.
All monetary amounts must be numbers (float), never strings."""

_EXTRACT_PROMPT = """Extract all invoice data from the text below and return a single JSON object:

{{
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
    {{"rate": 19.0, "net": 0.00, "tax": 0.00, "gross": 0.00}}
  ],
  "tax_type": "gross",
  "category_suggestion": "Brief description of what this invoice is for"
}}

Field rules:
- vendor_name: the company or person who ISSUED (sent) the invoice, not the recipient
- vat_id: the ISSUER's VAT ID. If you see DE322185267, IGNORE it — it is ours. Return null if no other VAT ID is present.
- tax_items: one entry per tax rate bucket. If all items share the same rate, one entry. Multiple rates → one entry per rate.
- tax_type options:
    "gross"                  — normal German invoice, VAT already included in amounts
    "net"                    — amounts shown net, tax added on top
    "vatfree"                — Kleinunternehmer or steuerfrei
    "constructionService13b" — §13b Bauleistungen
    "externalService13b"     — §13b Fremdleistungen
    "intraCommunitySupply"   — innergemeinschaftliche Lieferung
- All amounts must be numeric floats, not strings.

INVOICE TEXT:
{text}"""


# ── Combined extraction + category (used when vendor / category not cached) ───

_COMBINED_SYSTEM = """You are a German bookkeeping assistant and tax expert for Lexware.
Given PDF invoice text and a list of expense categories, extract all invoice data
AND select the single best posting category (Buchungskategorie) UUID.
Return ONLY valid JSON — no markdown, no explanation, no prose.
Never return VAT ID "DE322185267" as the vendor — that is our own VAT ID.
Dates must be YYYY-MM-DD. All amounts must be numeric floats."""

_COMBINED_PROMPT = """Extract all invoice data AND pick the best expense category.
Return a single JSON object:

{{
  "vendor_name": "Full legal company name of the invoice issuer (NOT the recipient)",
  "iban": "Vendor IBAN or null",
  "vat_id": "Vendor VAT ID — NOT DE322185267 — or null",
  "tax_number": "Steuernummer or null",
  "invoice_number": "Invoice number or null",
  "invoice_date": "YYYY-MM-DD or null",
  "due_date": "YYYY-MM-DD or null",
  "total_gross": 0.00,
  "total_tax": 0.00,
  "tax_items": [{{"rate": 19.0, "net": 0.00, "tax": 0.00, "gross": 0.00}}],
  "tax_type": "gross",
  "category_suggestion": "One-line description of what this invoice is for",
  "category_id": "UUID of the best matching category from the list below",
  "category_group": "Group name of that category",
  "category_name": "Name of that category"
}}

Tax type options: gross | net | vatfree | constructionService13b | externalService13b | intraCommunitySupply

CATEGORY SELECTION RULES (apply in order):
1. §13b invoices → choose §13b variant
2. EU purchases (intraCommunitySupply) → prefer "innergemeinschaftlicher" variants
3. Fuel, oil, car wash, charging → Fahrzeug > Kraftstoff/Ladestrom
4. SaaS / cloud subscriptions → Sonstige Ausgaben > Lizenzen und Konzessionen
5. Software maintenance → Sonstige Ausgaben > Wartungskosten für Hard- und Software
6. Mobile phone / internet / landline → Telekommunikation
7. Office supplies, paper, toner → Sonstige Ausgaben > Bürobedarf
8. Accounting / bookkeeping → Beratung > Buchführungskosten
9. Tax advisor → Beratung > Steuerberater
10. Legal / lawyer → Beratung > Rechtsanwalt
11. Advertising, marketing → Werbung
12. Business travel, hotel, taxi → Reisen
13. Raw materials / goods for resale → Material/Waren > Wareneinkauf
14. Subcontractors / freelancers → Fremdleistungen > Freelancer/Freie Mitarbeiter
15. Construction services (§13b) → Fremdleistungen > Bauleistungen §13b
16. Rent / lease → Raumkosten > Miete/Pacht
17. Electricity, gas, water → Raumkosten > Strom, Wasser, Gas
18. Genuinely unclear → use UUID: 8d2e71c6-09d5-439a-a295-a9e71661afcd

AVAILABLE CATEGORIES (group > name = UUID):
{categories}

INVOICE TEXT:
{text}"""


# ── Standalone category (legacy — kept for backward compat / PDF mode) ────────

_CATEGORY_SYSTEM = """You are a German tax expert for small business bookkeeping (EÜR / Bilanz).
Your job: pick the single best Lexware Buchungskategorie (posting category) UUID for an expense.
Return ONLY the UUID — no explanation, no punctuation, no other text whatsoever."""

_CATEGORY_PROMPT = """Match this purchase invoice to exactly one expense category.

INVOICE:
  Vendor        : {vendor_name}
  Description   : {category_suggestion}
  Tax type      : {tax_type}
  Amount        : €{total_gross}

GERMAN BOOKKEEPING RULES (apply in order):
1. §13b invoices → always choose the §13b variant of the matching category
2. EU purchases (intraCommunitySupply) → prefer "innergemeinschaftlicher" variants
3. Fuel, oil, car wash, charging → Fahrzeug > Kraftstoff/Ladestrom
4. SaaS / software subscriptions, cloud services → Sonstige Ausgaben > Lizenzen und Konzessionen
5. Software maintenance / support contracts → Sonstige Ausgaben > Wartungskosten für Hard- und Software
6. New software purchase (capitalized asset) → Anlagevermögen > Software
7. Mobile phone, internet, landline → Telekommunikation (pick the right sub-type)
8. Office supplies, paper, toner → Sonstige Ausgaben > Bürobedarf
9. Accounting, bookkeeping → Beratung > Buchführungskosten
10. Tax advisor → Beratung > Steuerberater
11. Legal / lawyer → Beratung > Rechtsanwalt
12. Advertising, marketing, SEO, print → Werbung
13. Business travel, hotel, taxi, flight → Reisen (pick the right sub-type)
14. Business meals (with clients) → Beschränkt abziehbare Betriebsausgaben > Bewirtungskosten (mit Geschäftspartnern)
15. Business insurance → Versicherungen (betrieblich)
16. Repairs of equipment/machines → Reparaturen > Wartung or Reparaturen
17. Raw materials / goods for resale → Material/Waren > Wareneinkauf
18. Subcontractors / freelancers → Fremdleistungen > Freelancer/Freie Mitarbeiter
19. Construction services (§13b) → Fremdleistungen > Bauleistungen §13b
20. Rent / lease → Raumkosten > Miete/Pacht
21. Electricity, gas, water → Raumkosten > Strom, Wasser, Gas
22. Vehicle lease → Fahrzeug > Mietleasing Kfz
23. Training / courses → Fortbildung > Seminar/Weiterbildung
24. Genuinely unclear → return the Zu prüfen UUID: 8d2e71c6-09d5-439a-a295-a9e71661afcd

AVAILABLE OUTGO CATEGORIES (group > name = UUID):
{categories}

Return ONLY the UUID of the single best match."""


# Inject the company's own VAT ID into every prompt that references it.
# This runs once at import time so the API calls always use the correct value.
if _OWN_VAT:
    for _p in ("_SYSTEM_PROMPT", "_EXTRACT_PROMPT",
               "_COMBINED_SYSTEM", "_COMBINED_PROMPT"):
        globals()[_p] = globals()[_p].replace("DE322185267", _OWN_VAT)
del _p


# ─────────────────────────────────────────────────────────────────────────────
# DB helpers
# ─────────────────────────────────────────────────────────────────────────────

def _load_outgo_categories() -> list[dict]:
    """Load all outgo posting categories from the local DB."""
    with get_db() as db:
        rows = db.execute("""
            SELECT id, name, group_name, split_allowed
            FROM   posting_categories
            WHERE  type = 'outgo'
            ORDER  BY group_name, name
        """).fetchall()
    return [dict(r) for r in rows]


def _format_categories(categories: list[dict]) -> str:
    """Format categories as compact lines for the prompt."""
    lines = []
    for c in categories:
        group = c["group_name"] or "Sonstige"
        lines.append(f"{group} > {c['name']} = {c['id']}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Vendor pre-scan — zero API cost
# ─────────────────────────────────────────────────────────────────────────────

def pre_scan_vendor(raw_text: str) -> VendorCache:
    """
    Regex-scan raw PDF text for VAT ID / IBAN, then check the local DB.
    Zero API cost — runs before any Claude call.

    If both the contact AND their preferred posting category are found in DB,
    sets hit=True so extract_and_resolve() can skip passing the full
    categories list to Claude (saves ~3 500 input tokens per invoice).
    """
    text_compact = re.sub(r"\s", "", raw_text).upper()

    # ── VAT IDs: DE + 9 digits ────────────────────────────────────────────
    vat_matches = re.findall(r"DE\d{9}", text_compact)
    vat_candidates = [v for v in dict.fromkeys(vat_matches) if v != _OWN_VAT]

    # ── IBANs ─────────────────────────────────────────────────────────────
    # German IBAN: DE + 2 check digits + 18 BBAN digits = 22 chars total.
    # All 20 chars after "DE" are purely numeric — this cleanly rejects
    # product names like "Organic Harithaki" that happen to start with a
    # valid country code (HR = Croatia) but contain letters in the BBAN.
    #
    # Match both forms that appear on invoices:
    #   grouped : "DE26 2004 0000 0225 2013 00"
    #   compact : "DE26200400000225201300"
    de_grouped = re.findall(
        r"DE\d{2}(?:\s*\d{4}){4}\s*\d{2}", raw_text.upper()
    )
    de_compact = re.findall(r"DE\d{20}", text_compact)
    iban_candidates = list(dict.fromkeys(
        [re.sub(r"\s", "", i) for i in de_grouped] + de_compact
    ))

    cache = VendorCache(
        vat_id_found  = vat_candidates[0]  if vat_candidates  else None,
        iban_found    = iban_candidates[0] if iban_candidates else None,
        contact_id    = None,
        contact_name  = None,
        category_id   = None,
        category_name = None,
        group_name    = None,
        hit           = False,
    )

    if not vat_candidates and not iban_candidates:
        logger.debug("Pre-scan: no identifiers found in text")
        return cache

    with get_db() as db:
        # ── VAT ID match ──────────────────────────────────────────────────
        for vat in vat_candidates:
            row = db.execute(
                "SELECT id, name FROM contacts "
                "WHERE UPPER(REPLACE(vat_id,' ','')) = ?",
                (vat,),
            ).fetchone()
            if row:
                cache.contact_id   = row["id"]
                cache.contact_name = row["name"]
                break

        # ── IBAN match (if no VAT hit) ────────────────────────────────────
        if not cache.contact_id:
            for iban in iban_candidates:
                row = db.execute(
                    "SELECT id, name FROM contacts "
                    "WHERE UPPER(REPLACE(iban,' ','')) = ?",
                    (iban,),
                ).fetchone()
                if row:
                    cache.contact_id   = row["id"]
                    cache.contact_name = row["name"]
                    break

        # ── Category history (if contact matched) ─────────────────────────
        if cache.contact_id:
            row = db.execute("""
                SELECT cch.category_id, pc.name, pc.group_name
                FROM   contact_category_history cch
                JOIN   posting_categories pc ON pc.id = cch.category_id
                WHERE  cch.contact_id   = ?
                AND    cch.voucher_type = 'purchaseinvoice'
                ORDER  BY cch.usage_count DESC
                LIMIT  1
            """, (cache.contact_id,)).fetchone()
            if row:
                cache.category_id   = row["category_id"]
                cache.category_name = row["name"]
                cache.group_name    = row["group_name"] or ""
                cache.hit           = True

    logger.info(
        "Pre-scan: vat=%r  iban=%r  contact=%r  category=%r  cache_hit=%s",
        cache.vat_id_found, cache.iban_found,
        cache.contact_name, cache.category_name, cache.hit,
    )
    return cache


# ─────────────────────────────────────────────────────────────────────────────
# JSON → InvoiceData parser (shared between extract paths)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_invoice_data(data: dict) -> InvoiceData:
    """Build an InvoiceData from a parsed JSON dict returned by Claude."""
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

    return InvoiceData(
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


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences that Claude sometimes wraps JSON in."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.IGNORECASE)
    return re.sub(r"\s*```$", "", cleaned.strip())


# ─────────────────────────────────────────────────────────────────────────────
# Smart extraction entry point — single optimised call
# ─────────────────────────────────────────────────────────────────────────────

def extract_and_resolve(
    pdf_path: Path,
) -> tuple[InvoiceData, CategoryResult, int, str, VendorCache]:
    """
    Extract invoice data AND resolve the posting category in the minimum
    number of Claude API calls.

    Algorithm
    ─────────
    1. pdfplumber → raw text  (free)
    2. pre_scan_vendor()      (free — regex + DB, no API)
       ├─ cache HIT  → one extraction-only call  (~1 500 tokens)
       │               category comes from DB, no categories list sent
       └─ cache MISS → one combined call          (~5 000 tokens)
                        extraction + category selection in one response

    Returns
    ───────
    (inv, cat, tokens_used, call_type, vendor_cache)

    call_type: "extraction_only" | "combined"
    """
    ZU_PRUEFEN_OUTGO = "8d2e71c6-09d5-439a-a295-a9e71661afcd"

    raw_text = extract_text_from_pdf(pdf_path)
    if not raw_text.strip():
        raise ValueError(
            f"pdfplumber extracted no text from {pdf_path.name}. "
            "This PDF may be image-only / scanned. Use pipeline.py (PDF upload mode) instead."
        )

    trimmed = raw_text[:6000]

    # ── Step 1: Zero-cost pre-scan ────────────────────────────────────────
    cache  = pre_scan_vendor(trimmed)
    client = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)

    # ── Route A: cache hit — extraction-only call (no categories list) ────
    if cache.hit:
        logger.info(
            "Cache HIT — vendor=%r  category=%r → extraction-only call",
            cache.contact_name, cache.category_name,
        )
        response = client.messages.create(
            model      = cfg.CLAUDE_MODEL,
            max_tokens = 1200,
            system     = _SYSTEM_PROMPT,
            messages   = [{"role": "user", "content": _EXTRACT_PROMPT.format(text=trimmed)}],
        )
        tokens    = response.usage.input_tokens + response.usage.output_tokens
        call_type = "extraction_only"

        cleaned = _strip_fences(response.content[0].text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Extraction-only call returned invalid JSON")
            data = {}

        inv = _parse_invoice_data(data)
        cat = CategoryResult(
            category_id   = cache.category_id,
            category_name = cache.category_name,
            group_name    = cache.group_name or "",
            method        = "contact_history",
            tokens_used   = 0,
        )

    # ── Route B: cache miss — combined call (extraction + category) ───────
    else:
        categories = _load_outgo_categories()
        prompt = _COMBINED_PROMPT.format(
            text       = trimmed,
            categories = _format_categories(categories),
        )
        response = client.messages.create(
            model      = cfg.CLAUDE_MODEL,
            max_tokens = 1500,
            system     = _COMBINED_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        tokens    = response.usage.input_tokens + response.usage.output_tokens
        call_type = "combined"

        cleaned = _strip_fences(response.content[0].text)
        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.error("Combined call returned invalid JSON: %s", response.content[0].text[:300])
            inv = _parse_invoice_data({})
            cat = CategoryResult(
                category_id   = ZU_PRUEFEN_OUTGO,
                category_name = "Zu prüfen",
                group_name    = "Zu prüfen",
                method        = "fallback",
                tokens_used   = tokens,
            )
            return inv, cat, tokens, call_type, cache

        inv = _parse_invoice_data(data)

        # ── Validate returned category UUID ───────────────────────────────
        raw_uuid  = (data.get("category_id") or "").strip().strip('"')
        valid_ids = {c["id"] for c in categories}
        valid_ids.add(ZU_PRUEFEN_OUTGO)

        if raw_uuid in valid_ids:
            match = next((c for c in categories if c["id"] == raw_uuid), None)
            cat = CategoryResult(
                category_id   = raw_uuid,
                category_name = match["name"]       if match else "Zu prüfen",
                group_name    = (match["group_name"] if match else "Zu prüfen") or "",
                method        = "claude_semantic",
                tokens_used   = tokens,
            )
            logger.info(
                "Combined call category: %s > %s (%s)",
                cat.group_name, cat.category_name, raw_uuid,
            )
        else:
            logger.warning("Combined call returned unknown UUID %r — falling back", raw_uuid)
            cat = CategoryResult(
                category_id   = ZU_PRUEFEN_OUTGO,
                category_name = "Zu prüfen",
                group_name    = "Zu prüfen",
                method        = "fallback",
                tokens_used   = tokens,
            )

    logger.info(
        "extract_and_resolve: call_type=%s  vendor=%r  category=%r  tokens=%d",
        call_type, inv.vendor_name, cat.category_name, tokens,
    )
    return inv, cat, tokens, call_type, cache


# ─────────────────────────────────────────────────────────────────────────────
# Text extraction via pdfplumber
# ─────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Extract all text from a PDF using pdfplumber.
    Returns an empty string if pdfplumber finds nothing (scanned PDF).
    """
    pages = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                pages.append(text.strip())

    full_text = "\n\n".join(pages)
    logger.debug("pdfplumber extracted %d characters from %s", len(full_text), pdf_path.name)
    return full_text


# ─────────────────────────────────────────────────────────────────────────────
# Legacy single-step functions (kept for backward compatibility / PDF mode)
# ─────────────────────────────────────────────────────────────────────────────

def extract_invoice_via_text(pdf_path: Path) -> tuple[InvoiceData, int, str]:
    """
    Extract text with pdfplumber, send text to Claude (extraction-only).
    Returns (InvoiceData, tokens_used, raw_text).

    For the optimised path that also resolves the category in one call,
    use extract_and_resolve() instead.
    """
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        raise ValueError(
            f"pdfplumber extracted no text from {pdf_path.name}. "
            "This PDF may be image-only / scanned. Use pipeline.py instead."
        )

    trimmed = raw_text[:6000]

    client   = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model      = cfg.CLAUDE_MODEL,
        max_tokens = 1200,
        system     = _SYSTEM_PROMPT,
        messages   = [{"role": "user", "content": _EXTRACT_PROMPT.format(text=trimmed)}],
    )

    tokens  = response.usage.input_tokens + response.usage.output_tokens
    cleaned = _strip_fences(response.content[0].text)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        logger.error("Claude returned invalid JSON — raw: %s", response.content[0].text[:300])
        return _parse_invoice_data({}), tokens, raw_text

    inv = _parse_invoice_data(data)
    logger.info(
        "Text-mode extracted: vendor=%r  invoice=%r  date=%r  gross=%.2f  tokens=%d",
        inv.vendor_name, inv.invoice_number, inv.invoice_date,
        inv.total_gross or 0.0, tokens,
    )
    return inv, tokens, raw_text


def resolve_category_smart(inv: InvoiceData, contact_id: Optional[str]) -> CategoryResult:
    """
    Resolve the best Lexware posting category for this invoice.

    Priority:
      1. Contact history (free)
      2. Claude semantic matching (standalone category call)
      3. Fallback → Zu prüfen

    For new invoices, prefer extract_and_resolve() which combines extraction
    and category resolution into a single API call and is more efficient.
    """
    ZU_PRUEFEN_OUTGO = "8d2e71c6-09d5-439a-a295-a9e71661afcd"

    # ── 1. Contact history ────────────────────────────────────────────────
    if contact_id:
        with get_db() as db:
            row = db.execute("""
                SELECT cch.category_id, pc.name, pc.group_name
                FROM   contact_category_history cch
                JOIN   posting_categories pc ON pc.id = cch.category_id
                WHERE  cch.contact_id   = ?
                AND    cch.voucher_type = 'purchaseinvoice'
                ORDER  BY cch.usage_count DESC
                LIMIT  1
            """, (contact_id,)).fetchone()
            if row:
                logger.info(
                    "Category from contact history: %s > %s",
                    row["group_name"], row["name"],
                )
                return CategoryResult(
                    category_id   = row["category_id"],
                    category_name = row["name"],
                    group_name    = row["group_name"] or "",
                    method        = "contact_history",
                )

    # ── 2. Claude semantic matching ───────────────────────────────────────
    categories = _load_outgo_categories()
    if not categories:
        return CategoryResult(
            category_id   = ZU_PRUEFEN_OUTGO,
            category_name = "Zu prüfen",
            group_name    = "Zu prüfen",
            method        = "fallback",
        )

    prompt = _CATEGORY_PROMPT.format(
        vendor_name         = inv.vendor_name         or "Unknown",
        category_suggestion = inv.category_suggestion or "General business expense",
        tax_type            = inv.tax_type            or "gross",
        total_gross         = f"{inv.total_gross:.2f}" if inv.total_gross else "unknown",
        categories          = _format_categories(categories),
    )

    try:
        client   = anthropic.Anthropic(api_key=cfg.ANTHROPIC_API_KEY)
        response = client.messages.create(
            model      = cfg.CLAUDE_MODEL,
            max_tokens = 50,
            system     = _CATEGORY_SYSTEM,
            messages   = [{"role": "user", "content": prompt}],
        )
        tokens   = response.usage.input_tokens + response.usage.output_tokens
        raw_uuid = response.content[0].text.strip().strip('"').strip()

        valid_ids = {c["id"] for c in categories}
        valid_ids.add(ZU_PRUEFEN_OUTGO)

        if raw_uuid in valid_ids:
            match = next((c for c in categories if c["id"] == raw_uuid), None)
            name  = match["name"]       if match else "Zu prüfen"
            group = match["group_name"] if match else "Zu prüfen"
            logger.info(
                "Category from Claude semantic: %s > %s (%s)  tokens=%d",
                group, name, raw_uuid, tokens,
            )
            return CategoryResult(
                category_id   = raw_uuid,
                category_name = name,
                group_name    = group or "",
                method        = "claude_semantic",
                tokens_used   = tokens,
            )
        else:
            logger.warning("Claude returned unknown UUID %r — falling back", raw_uuid)

    except Exception as e:
        logger.error("Category Claude call failed: %s", e)

    # ── 3. Fallback ───────────────────────────────────────────────────────
    return CategoryResult(
        category_id   = ZU_PRUEFEN_OUTGO,
        category_name = "Zu prüfen",
        group_name    = "Zu prüfen",
        method        = "fallback",
        tokens_used   = 0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Main per-invoice pipeline
# ─────────────────────────────────────────────────────────────────────────────

def process_invoice(pdf_path: Path) -> dict:
    """
    Process a single PDF using the optimised text-extraction pipeline.
    Returns same result dict as pipeline.process_invoice().
    """
    now    = datetime.now(timezone.utc).isoformat()
    result = {
        "status":       "failed",
        "voucher_id":   None,
        "contact_id":   None,
        "contact_name": None,
        "category_id":  None,
        "tokens_used":  0,
        "call_type":    None,
        "error":        None,
        "mode":         "text",
    }

    logger.info("=" * 60)
    logger.info("Processing (text mode): %s", pdf_path.name)
    logger.info("=" * 60)

    # ── Duplicate guard ───────────────────────────────────────────────────
    try:
        file_hash = pdf_hash(pdf_path)
    except FileNotFoundError:
        result["error"] = f"PDF not found: {pdf_path}"
        return result

    if is_already_processed(file_hash):
        logger.info("SKIPPED — already processed (hash %.12s…)", file_hash)
        result["status"] = "skipped"
        return result

    lx_client = LexwareClient()

    # ── Step 1: Smart extraction (pre-scan + optimal Claude call) ─────────
    try:
        inv, cat_result, tokens, call_type, _ = extract_and_resolve(pdf_path)
        result["tokens_used"]  = tokens
        result["call_type"]    = call_type
        result["contact_name"] = inv.vendor_name
    except Exception as e:
        result["error"] = f"Text extraction failed: {e}"
        logger.error(result["error"])
        _move_pdf(pdf_path, cfg.PDF_FAILED)
        _write_audit(result, pdf_path.name, file_hash, now)
        return result

    # ── Step 2: Look up contact ───────────────────────────────────────────
    contact_id           = lookup_contact(inv)
    result["contact_id"] = contact_id

    # ── Step 3: Create contact if not in DB ──────────────────────────────
    if not contact_id:
        contact_id           = create_contact(inv, lx_client)
        result["contact_id"] = contact_id

    # ── Step 4: Use resolved category ────────────────────────────────────
    # If cat_result came from contact_history but was pre-scanned before
    # contact lookup, confirm it matches the now-resolved contact_id.
    # (In practice the pre-scan contact_id === lookup_contact result.)
    category_id           = cat_result.category_id
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
        except Exception as e:
            logger.warning("Failed to update learning DB: %s", e)

    # ── Step 8: Audit log ─────────────────────────────────────────────────
    _write_audit(result, pdf_path.name, file_hash, now)

    # ── Step 9: Move PDF ──────────────────────────────────────────────────
    if status in ("open", "unchecked"):
        _move_pdf(pdf_path, cfg.PDF_PROCESSED)
    else:
        _move_pdf(pdf_path, cfg.PDF_FAILED)

    logger.info(
        "Done: status=%s  voucher=%s  call_type=%s  tokens=%d",
        status, voucher_id, call_type, tokens,
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Batch runner
# ─────────────────────────────────────────────────────────────────────────────

def run_batch() -> dict:
    """Process all PDFs in pdfs/inbox_ocr/ using text-extraction mode."""
    init_db()
    hot_sync()

    PDF_OCR_INBOX.mkdir(parents=True, exist_ok=True)
    pdfs = sorted(PDF_OCR_INBOX.glob("*.pdf"))

    if not pdfs:
        logger.info("No PDFs found in %s", PDF_OCR_INBOX)
        return {"open": 0, "unchecked": 0, "failed": 0, "skipped": 0}

    logger.info("Batch (text mode): %d PDF(s) queued", len(pdfs))
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
