#!/usr/bin/env python3
"""
test.py
───────
Dry-run test harness for pipeline.py (PDF mode) and processor.py (text mode).

Sends a PDF to Claude and prints every extracted field, the contact match
result, the math check, and the full voucher payload — without writing
anything to Lexware or moving the PDF.

Usage:
    python test.py path/to/invoice.pdf           # PDF upload mode  (pipeline.py)
    python test.py --text path/to/invoice.pdf    # Text extract mode (processor.py)

Flags:
  --text    Use pdfplumber text extraction instead of uploading the raw PDF.
            Fewer tokens, but may miss data on scanned / image-only PDFs.
            Also uses the optimised single-call path (extraction + category
            in one Claude call, or extraction-only when vendor is cached).
  --learn   After resolving the category, write it to contact_category_history
            so future runs for the same vendor hit the cache (extraction-only).
            Does NOT post anything to Lexware. Safe to use with --text.

Output sections:
  [0] PRE-SCAN      — vendor lookup in DB before any API call (--text only)
  [1] EXTRACTION    — every field Claude pulled from the PDF
  [2] CONTACT MATCH — whether the vendor was found in DB (and how)
  [3] CATEGORY      — Buchungskategorie resolved (method + UUID)
  [4] MATH CHECK    — whether the amounts add up
  [5] VOUCHER PAYLOAD — the exact JSON that would be sent to Lexware
  [6] RATING CARD   — summary checklist + token cost breakdown
"""

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).parent
sys.path.insert(0, str(_ROOT))

from config import cfg, get_logger
from pipeline import (
    lookup_contact,
    math_check,
    _norm_vat,
    _norm_name,
    _edit_distance,
    TaxItem,
    InvoiceData,
)
from agent.db import init_db, get_db

logger = get_logger("test")

# ── ANSI colours (works on macOS/Linux terminals) ─────────────────────────────
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_RED    = "\033[91m"
_CYAN   = "\033[96m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"
_RESET  = "\033[0m"

def _ok(s):   return f"{_GREEN}✓  {s}{_RESET}"
def _warn(s): return f"{_YELLOW}⚠  {s}{_RESET}"
def _fail(s): return f"{_RED}✗  {s}{_RESET}"
def _info(s): return f"{_CYAN}→  {s}{_RESET}"
def _h(s):    return f"{_BOLD}{s}{_RESET}"
def _dim(s):  return f"{_DIM}{s}{_RESET}"

SEP  = "─" * 62
SEP2 = "═" * 62


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _val(v, unit: str = "") -> str:
    """Format a value — red if None/empty, green otherwise."""
    if v is None or v == "" or v == []:
        return _fail("null")
    return f"{_GREEN}{v}{unit}{_RESET}"


def _rating(label: str, passed: bool, note: str = "") -> str:
    icon = _ok(label) if passed else _fail(label)
    return f"  {icon}" + (f"  {_dim(note)}" if note else "")


# ─────────────────────────────────────────────────────────────────────────────
# Section printers
# ─────────────────────────────────────────────────────────────────────────────

def print_prescan(cache, call_type: str, tokens: int) -> None:
    """
    Section [0] — show vendor pre-scan results and the call routing decision.
    Only shown in --text mode.
    """
    print(f"\n{SEP}")
    print(_h(" [0]  PRE-SCAN  —  DB lookup before any API call"))
    print(SEP)

    print(f"  VAT ID found  : {_val(cache.vat_id_found)}")
    print(f"  IBAN found    : {_val(cache.iban_found)}")
    print()

    if cache.contact_id:
        print(_ok(f"Contact pre-matched  → {cache.contact_name}"))
        print(f"  {_dim(f'UUID: {cache.contact_id}')}")
        if cache.hit:
            print(_ok(f"Category cached      → {cache.group_name} > {cache.category_name}"))
            print(f"  {_dim(f'UUID: {cache.category_id}')}")
        else:
            print(_warn("Category not in cache  — will be resolved via Claude"))
    else:
        print(_warn("Vendor not found in DB  — full combined call required"))

    print()

    call_colours = {
        "extraction_only": _GREEN,
        "combined":        _CYAN,
    }
    colour = call_colours.get(call_type, _YELLOW)

    call_descriptions = {
        "extraction_only": "extraction-only  (category from cache — no categories list sent)",
        "combined":        "combined  (extraction + category in one call)",
    }
    desc = call_descriptions.get(call_type, call_type)

    print(f"  Call type  : {colour}{_BOLD}{desc}{_RESET}")
    print(f"  Tokens     : {tokens}")

    # Estimate savings vs naive two-call baseline
    NAIVE_TWO_CALL_TOKENS = 14_000
    saved = max(0, NAIVE_TWO_CALL_TOKENS - tokens)
    if saved > 0:
        pct = round(saved / NAIVE_TWO_CALL_TOKENS * 100)
        print(f"  {_GREEN}~{saved:,} tokens saved vs naive two-call baseline  ({pct}% reduction){_RESET}")


def print_extraction(inv: InvoiceData, tokens: int) -> None:
    print(f"\n{SEP}")
    print(_h(" [1]  EXTRACTION  —  what Claude read from the PDF"))
    print(SEP)

    rows = [
        ("Vendor name",     inv.vendor_name),
        ("VAT ID",          inv.vat_id),
        ("IBAN",            inv.iban),
        ("Tax number",      inv.tax_number),
        ("Invoice number",  inv.invoice_number),
        ("Invoice date",    inv.invoice_date),
        ("Due date",        inv.due_date),
        ("Total gross",     f"€ {inv.total_gross:.2f}" if inv.total_gross is not None else None),
        ("Total tax",       f"€ {inv.total_tax:.2f}"   if inv.total_tax   is not None else None),
        ("Tax type",        inv.tax_type),
        ("Category hint",   inv.category_suggestion),
    ]
    col = 18
    for label, val in rows:
        print(f"  {label:<{col}} {_val(val)}")

    print(f"\n  {'Tax items':<{col}}", end="")
    if inv.tax_items:
        for i, item in enumerate(inv.tax_items):
            prefix = " " * (col + 2) if i > 0 else ""
            print(
                f"{prefix}{_GREEN}{item.rate:.0f}%{_RESET}"
                f"  net=€{item.net:.2f}"
                f"  tax=€{item.tax:.2f}"
                f"  gross=€{item.gross:.2f}"
            )
    else:
        print(_fail("none extracted"))

    print(f"\n  {_dim(f'Claude tokens used: {tokens}')}")


def print_contact_match(inv: InvoiceData) -> tuple[str | None, str]:
    """
    Print contact match details and return (contact_id, match_method).
    Runs a verbose version of the lookup so every step is visible.
    """
    print(f"\n{SEP}")
    print(_h(" [2]  CONTACT MATCH  —  vendor lookup in local DB"))
    print(SEP)

    inv_vat  = _norm_vat(inv.vat_id or "")
    import re as _re
    inv_iban = _re.sub(r"\s", "", inv.iban or "").upper()
    inv_name = _norm_name(inv.vendor_name or "")

    contact_id   = None
    match_method = "none"
    contact_name_db = None

    with get_db() as db:
        total = db.execute("SELECT COUNT(*) FROM contacts").fetchone()[0]
        print(f"  Contacts in DB : {total}")
        print(f"  Searching for  : VAT={_val(inv.vat_id)}  IBAN={_val(inv.iban)}")
        print(f"  Normalised name: {_val(inv_name)}")
        print()

        # ── VAT ID ────────────────────────────────────────────────────────
        if inv_vat:
            row = db.execute(
                "SELECT id, name, vat_id, iban FROM contacts "
                "WHERE UPPER(REPLACE(vat_id,' ','')) = ?",
                (inv_vat,),
            ).fetchone()
            if row:
                contact_id      = row["id"]
                contact_name_db = row["name"]
                match_method    = "VAT_ID"
                print(_ok(f"VAT ID match  → {row['name']}"))
                print(f"  {_dim(f'UUID: {contact_id}')}")
            else:
                print(_warn(f"VAT ID {inv.vat_id!r} not found in DB"))
        else:
            print(_dim("  VAT ID not available — skipping VAT lookup"))

        # ── IBAN ──────────────────────────────────────────────────────────
        if not contact_id and inv_iban:
            row = db.execute(
                "SELECT id, name, iban FROM contacts "
                "WHERE UPPER(REPLACE(iban,' ','')) = ?",
                (inv_iban,),
            ).fetchone()
            if row:
                contact_id      = row["id"]
                contact_name_db = row["name"]
                match_method    = "IBAN"
                print(_ok(f"IBAN match    → {row['name']}"))
                print(f"  {_dim(f'UUID: {contact_id}')}")
            else:
                print(_warn(f"IBAN {inv.iban!r} not found in DB"))
        elif not contact_id:
            print(_dim("  IBAN not available — skipping IBAN lookup"))

        # ── Fuzzy name ────────────────────────────────────────────────────
        if not contact_id and inv_name:
            rows = db.execute(
                "SELECT id, name FROM contacts WHERE role_vendor = 1"
            ).fetchall()

            candidates = []
            for r in rows:
                n = _norm_name(r["name"] or "")
                if not n:
                    continue
                dist = _edit_distance(inv_name, n)
                if dist == 0:
                    score = 80
                elif dist <= 2:
                    score = 60
                elif dist <= 5 and len(inv_name) > 8:
                    import re as _re2
                    inv_tok = set(inv_name.split())
                    c_tok   = set(n.split())
                    common  = inv_tok & c_tok
                    if len(common) >= 2 and len(common) / max(len(inv_tok), 1) >= 0.6:
                        score = 40
                    else:
                        continue
                else:
                    continue
                candidates.append((score, r["id"], r["name"], n, dist))

            candidates.sort(reverse=True)

            print(f"\n  Fuzzy name search against {len(rows)} vendor contacts:")
            if candidates:
                for score, cid, raw_name, norm, dist in candidates[:5]:
                    marker = _ok if score >= 60 else _warn if score >= 40 else _dim
                    print(f"    score={score:3d}  dist={dist}  {marker(raw_name)}")
                    if score >= 60 and not contact_id:
                        contact_id      = cid
                        contact_name_db = raw_name
                        match_method    = f"NAME_FUZZY (score={score}, dist={dist})"
                if contact_id:
                    print(f"\n  {_ok(f'Fuzzy match accepted → {contact_name_db}')}")
                    print(f"  {_dim(f'UUID: {contact_id}')}")
                else:
                    print(f"\n  {_warn('Best fuzzy score below threshold (60) — no match')}")
            else:
                print(f"    {_dim('No candidates found')}")

        elif not contact_id:
            print(_dim("  Vendor name not available — skipping fuzzy search"))

    # ── Final verdict ─────────────────────────────────────────────────────
    print()
    if contact_id:
        print(_ok(f"MATCH FOUND   method={match_method}"))
        print(f"  DB name  : {_GREEN}{contact_name_db}{_RESET}")
        print(f"  UUID     : {_GREEN}{contact_id}{_RESET}")
    else:
        print(_warn("NO MATCH — a new contact would be created in Lexware"))
        print(f"  Payload  : {{\"company\": {{\"name\": {inv.vendor_name!r}, "
              f"\"vatRegistrationId\": {inv.vat_id!r}}}}}")

    return contact_id, match_method


def print_category_resolution(inv: InvoiceData, contact_id: str | None,
                               pre_resolved=None):
    """
    Section [3] — display the resolved posting category.

    If pre_resolved is given (a CategoryResult from extract_and_resolve),
    it is displayed directly without making another Claude call.
    Otherwise resolve_category_smart() is called (used in PDF mode).
    """
    print(f"\n{SEP}")
    print(_h(" [3]  CATEGORY RESOLUTION  —  Buchungskategorie"))
    print(SEP)

    if pre_resolved is not None:
        # Category was resolved as part of the combined extraction call
        method_note = {
            "contact_history": "from contact cache  (no extra API call)",
            "claude_semantic":  "resolved by Claude  (part of combined call — no extra API call)",
            "fallback":         "fallback — could not determine automatically",
        }.get(pre_resolved.method, pre_resolved.method)
        print(f"  {_dim(method_note)}")
        cat = pre_resolved
    else:
        from processor import resolve_category_smart
        if contact_id:
            print(f"  {_dim('Checking contact history first, then Claude taxonomy if needed…')}")
        else:
            print(f"  {_dim('No contact match — going straight to Claude taxonomy…')}")
        cat = resolve_category_smart(inv, contact_id)

    method_colour = _GREEN if cat.method == "contact_history" else \
                    _CYAN  if cat.method == "claude_semantic"  else _YELLOW

    print(f"\n  Method      : {method_colour}{cat.method}{_RESET}")
    print(f"  Group       : {_val(cat.group_name)}")
    print(f"  Category    : {_val(cat.category_name)}")
    print(f"  UUID        : {_GREEN}{cat.category_id}{_RESET}")
    if cat.tokens_used:
        print(f"  Tokens used : {_dim(str(cat.tokens_used))}")

    if cat.category_name == "Zu prüfen":
        print(f"\n  {_warn('Fell back to Zu prüfen — bookkeeper must assign category manually')}")
    else:
        print(f"\n  {_ok(f'{cat.group_name} > {cat.category_name}')}")

    return cat


def print_math_check(inv: InvoiceData) -> bool:
    print(f"\n{SEP}")
    print(_h(" [4]  MATH CHECK  —  do the numbers add up?"))
    print(SEP)

    ok, reason = math_check(inv)

    if inv.tax_items:
        calc_gross = round(sum(i.gross for i in inv.tax_items), 2)
        calc_tax   = round(sum(i.tax   for i in inv.tax_items), 2)
        print(f"  Sum of tax items (gross) : € {calc_gross:.2f}")
        print(f"  Invoice total gross      : € {inv.total_gross:.2f}" if inv.total_gross else "  Invoice total gross      : null")
        print(f"  Sum of tax items (tax)   : € {calc_tax:.2f}")
        print(f"  Invoice total tax        : € {inv.total_tax:.2f}"   if inv.total_tax   else "  Invoice total tax        : null")
    else:
        print(_fail("No tax items — cannot verify"))

    print()
    if ok:
        print(_ok("Math check PASSED — amounts are consistent (±€0.05)"))
    else:
        print(_fail(f"Math check FAILED — {reason}"))
        print(_warn("Voucher would be posted as UNCHECKED for manual review"))

    return ok


def print_voucher_payload(inv: InvoiceData, contact_id: str | None,
                          category_id: str, math_ok: bool) -> None:
    print(f"\n{SEP}")
    print(_h(" [5]  VOUCHER PAYLOAD  —  what would be sent to Lexware"))
    print(SEP)

    go_unchecked = not math_ok or not contact_id or category_id == cfg.ZU_PRUEFEN_CATEGORY_ID
    remark_parts = []
    if not math_ok:
        remark_parts.append("Math check failed")
    if not contact_id:
        remark_parts.append("No vendor contact resolved")
    if category_id == cfg.ZU_PRUEFEN_CATEGORY_ID:
        remark_parts.append("Category unknown — to be reviewed")

    if go_unchecked:
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
        if (inv.total_gross and inv.total_gross > 0
                and inv.total_tax is not None
                and 0 <= inv.total_tax < inv.total_gross
                and inv.total_tax / inv.total_gross <= 0.25):
            payload["totalGrossAmount"] = inv.total_gross
            payload["totalTaxAmount"]   = inv.total_tax
        payload["contactId" if contact_id else "useCollectiveContact"] = (
            contact_id if contact_id else True
        )
        if contact_id:
            payload["useCollectiveContact"] = False
    else:
        items = []
        for item in inv.tax_items:
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

    status_colour = _GREEN if payload["voucherStatus"] == "open" else _YELLOW
    print(f"  Would post as: {status_colour}{_BOLD}{payload['voucherStatus'].upper()}{_RESET}\n")
    print(json.dumps(payload, indent=4, ensure_ascii=False))


def print_rating_card(inv: InvoiceData, contact_id: str | None,
                      match_method: str, math_ok: bool, tokens: int,
                      cat=None, call_type: str | None = None) -> None:
    print(f"\n{SEP2}")
    print(_h(" [6]  RATING CARD  —  manual verification checklist"))
    print(SEP2)

    fields_extracted = sum(1 for v in [
        inv.vendor_name, inv.vat_id, inv.iban,
        inv.invoice_number, inv.invoice_date, inv.total_gross, inv.total_tax,
    ] if v is not None)

    cat_resolved = (
        cat is not None
        and hasattr(cat, "category_name")
        and cat.category_name != "Zu prüfen"
    )
    cat_label = (
        f"{cat.group_name} > {cat.category_name}" if cat_resolved
        else "Zu prüfen — manual review needed"
    )

    print(f"\n  Extraction quality")
    print(_rating("Vendor name extracted",    bool(inv.vendor_name),    inv.vendor_name or ""))
    print(_rating("VAT ID extracted",         bool(inv.vat_id),         inv.vat_id or "not on invoice"))
    print(_rating("IBAN extracted",           bool(inv.iban),           inv.iban or "not on invoice"))
    print(_rating("Invoice number extracted", bool(inv.invoice_number), inv.invoice_number or ""))
    print(_rating("Invoice date extracted",   bool(inv.invoice_date),   inv.invoice_date or ""))
    print(_rating("Total gross extracted",    bool(inv.total_gross),    f"€ {inv.total_gross:.2f}" if inv.total_gross else ""))
    print(_rating("Total tax extracted",      bool(inv.total_tax),      f"€ {inv.total_tax:.2f}"   if inv.total_tax   else ""))
    print(_rating("Tax items extracted",      bool(inv.tax_items),      f"{len(inv.tax_items)} rate bucket(s)"))

    print(f"\n  Contact matching")
    print(_rating("Contact found in DB",      bool(contact_id),         match_method if contact_id else "will create new"))

    print(f"\n  Category classification")
    print(_rating("Category resolved",        cat_resolved,             cat_label))

    print(f"\n  Amount validation")
    print(_rating("Math check passed",        math_ok,                  "amounts are consistent" if math_ok else "review amounts manually"))

    # Score: 7 extraction fields + contact + category + math = 10
    score = fields_extracted + (1 if contact_id else 0) + (1 if cat_resolved else 0) + (1 if math_ok else 0)
    max_score = 10
    pct = round(score / max_score * 100)
    colour = _GREEN if pct >= 80 else _YELLOW if pct >= 50 else _RED

    print(f"\n  {SEP}")
    print(f"  Extraction score : {colour}{_BOLD}{score}/{max_score} ({pct}%){_RESET}")
    print(f"  Tokens used      : {tokens}")

    if call_type:
        NAIVE_TOKENS = 14_000
        saved = max(0, NAIVE_TOKENS - tokens)
        if saved > 0:
            pct_saved = round(saved / NAIVE_TOKENS * 100)
            print(f"  Token savings    : {_GREEN}~{saved:,} vs naive two-call ({pct_saved}% reduction){_RESET}")
        call_labels = {
            "extraction_only": f"{_GREEN}extraction_only  (vendor cached){_RESET}",
            "combined":        f"{_CYAN}combined  (extraction + category){_RESET}",
        }
        print(f"  Claude calls     : {call_labels.get(call_type, call_type)}")

    print(f"  {_dim('Rate this run: was the data accurate? Y/N/partial')}")
    print(f"  {SEP}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _write_cache(contact_id: str, category_id: str, tax_type: str) -> None:
    """Write vendor → category mapping to DB so future runs hit the cache."""
    from agent.db import update_category_history, update_category_usage_count
    from datetime import date
    update_category_history(
        contact_id   = contact_id,
        category_id  = category_id,
        voucher_type = "purchaseinvoice",
        tax_type     = tax_type or "gross",
        used_at      = date.today().isoformat(),
    )
    update_category_usage_count(category_id)


def main() -> None:
    args = sys.argv[1:]

    # Parse flags
    text_mode  = "--text"  in args
    learn_mode = "--learn" in args
    args = [a for a in args if a not in ("--text", "--learn")]

    if not args:
        print("Usage: python test.py [--text] path/to/invoice.pdf")
        print("  (no flag) — PDF upload mode   — sends raw PDF to Claude")
        print("  --text    — Text extract mode  — pdfplumber → plain text → Claude")
        sys.exit(1)

    pdf_path = Path(args[0])
    if not pdf_path.exists():
        print(f"{_fail(f'PDF not found: {pdf_path}')}")
        sys.exit(1)

    mode_label = "TEXT EXTRACT (processor)"  if text_mode else "PDF UPLOAD (pipeline)"
    mode_note  = "pdfplumber → plain text → Claude (optimised single-call)" if text_mode \
                 else "raw PDF binary → Claude document API"

    print(f"\n{SEP2}")
    print(_h(f" DRY RUN — {pdf_path.name}"))
    print(f"  Mode : {_CYAN}{mode_label}{_RESET}")
    print(f"  How  : {_dim(mode_note)}")
    learn_note = "  Category will be saved to cache (--learn)" if learn_mode else \
                 "  Nothing is written to Lexware or moved on disk"
    print(_dim(learn_note))
    print(SEP2)

    init_db()

    # ─────────────────────────────────────────────────────────────────────
    # TEXT MODE — optimised path: extract_and_resolve() does pre-scan +
    # either extraction-only (cache hit) or combined call (cache miss)
    # ─────────────────────────────────────────────────────────────────────
    if text_mode:
        from processor import extract_and_resolve
        print(f"\n{_info('Pre-scanning vendor in DB, then calling Claude…')}")
        try:
            inv, cat, tokens, call_type, cache = extract_and_resolve(pdf_path)
        except Exception as e:
            print(f"\n{_fail(f'Extraction failed: {e}')}")
            sys.exit(1)

        # [0] Pre-scan results + routing decision
        print_prescan(cache, call_type, tokens)

        # [1] Extraction
        print_extraction(inv, tokens)

        # [2] Contact match (full verbose lookup)
        contact_id, match_method = print_contact_match(inv)

        # [3] Category — already resolved, just display
        cat = print_category_resolution(inv, contact_id, pre_resolved=cat)
        category_id = cat.category_id

        # [4] Math check
        math_ok = print_math_check(inv)

        # [5] Voucher payload
        print_voucher_payload(inv, contact_id, category_id, math_ok)

        # [6] Rating card with token savings
        print_rating_card(inv, contact_id, match_method, math_ok, tokens, cat, call_type)

        # --learn: write category to DB so the next run hits the cache
        ZU_PRUEFEN_OUTGO = "8d2e71c6-09d5-439a-a295-a9e71661afcd"
        if learn_mode and contact_id and category_id and category_id != ZU_PRUEFEN_OUTGO:
            _write_cache(contact_id, category_id, inv.tax_type)
            print(f"\n{_ok(f'Cache updated — {cat.group_name} > {cat.category_name} saved for {contact_id[:8]}…')}")
            print(f"  {_dim('Next run for this vendor will use extraction-only (~1 500 tokens)')}")
        elif learn_mode and (not contact_id or category_id == ZU_PRUEFEN_OUTGO):
            print(f"\n{_warn('--learn skipped — contact not matched or category is Zu prüfen')}")

    # ─────────────────────────────────────────────────────────────────────
    # PDF MODE — binary upload to Claude, separate category call
    # ─────────────────────────────────────────────────────────────────────
    else:
        from pipeline import extract_invoice_via_claude
        print(f"\n{_info('Sending PDF binary to Claude…')}")
        try:
            inv, tokens = extract_invoice_via_claude(pdf_path)
        except Exception as e:
            print(f"\n{_fail(f'Claude extraction failed: {e}')}")
            sys.exit(1)

        # [1] Extraction
        print_extraction(inv, tokens)

        # [2] Contact match
        contact_id, match_method = print_contact_match(inv)

        # [3] Category — calls resolve_category_smart (separate call)
        cat = print_category_resolution(inv, contact_id)
        category_id = cat.category_id if hasattr(cat, "category_id") else cat
        cat_tokens  = cat.tokens_used if hasattr(cat, "tokens_used") else 0

        # [4] Math check
        math_ok = print_math_check(inv)

        # [5] Voucher payload
        print_voucher_payload(inv, contact_id, category_id, math_ok)

        # [6] Rating card (no call_type in PDF mode)
        print_rating_card(inv, contact_id, match_method, math_ok, tokens + cat_tokens, cat)


if __name__ == "__main__":
    main()
