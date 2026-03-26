"""
agent/sync.py
─────────────
Cold sync and hot sync engine.

Cold sync  — first ever run. Pulls complete Lexware history:
             contacts, posting-categories, articles,
             payment-conditions, and all voucher detail
             to build the contact_category_history table.

Hot sync   — every session start. Uses updatedDateFrom filters
             to pull only the delta since last sync.
             Fast — usually completes in seconds.

Usage:
    from agent.sync import cold_sync, hot_sync
    stats = cold_sync()          # first run
    hot_sync()                   # every subsequent session
"""
"""
═══════════════════════════════════════════════════════════════════════════════
EXECUTION COMMANDS — LEXAPRO AI PIPELINE
═══════════════════════════════════════════════════════════════════════════════

# ── SYNC OPERATIONS ──────────────────────────────────────────────────────────

# 1. HOT SYNC (incremental update)
# Pulls only new/updated data from Lexware
python -m agent.sync

# 2. COLD SYNC (full re-fetch, keeps existing DB)
# Rebuilds local state but respects existing data
python -m agent.sync cold

# 3. RESET DATABASE (wipe all runtime data)
# Deletes contacts, history, processed invoices, sync state
python -m agent.sync reset

# 4. RESET + COLD SYNC (FULL FRESH REBUILD)  ← RECOMMENDED
# Completely wipes DB and rebuilds from Lexware
python -m agent.sync reset_cold


# ── OCR PREPROCESSING ───────────────────────────────────────────────────────

# 5. MANUAL OCR RUN
# Scans inbox/ and sends non-text PDFs → inbox_ocr/
python preprocess.py

# 6. WATCHER MODE (AUTO OCR ON FILE DROP)
# Monitors inbox/ and triggers OCR automatically
python watcher.py


# ── DEBUG / INSPECTION ──────────────────────────────────────────────────────

# 7. DATABASE DUMP (JSON OUTPUT)
# Prints full DB state (contacts, invoices, sync state, etc.)
python inspect_db.py


# ── RECOMMENDED WORKFLOWS ────────────────────────────────────────────────────

# FULL CLEAN SYSTEM RESET + REBUILD
python -m agent.sync reset_cold
python preprocess.py

# DAILY OPERATION (INCREMENTAL)
python -m agent.sync
python preprocess.py


# ── NOTES ───────────────────────────────────────────────────────────────────

# • Always run commands from project root (LexaProAi/)
# • Use module execution (-m) for all agent scripts
# • Do NOT run files like: python agent/sync.py  (breaks imports)
# • OCR output is stored separately (non-destructive processing)
# • reset_cold = safest way to recover from bad data or corruption

═══════════════════════════════════════════════════════════════════════════════
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Ensure project root is on sys.path when executed directly
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import cfg, get_logger
from agent.db import get_db, get_sync_state, set_sync_state
from agent.lexware_client import LexwareClient, LexwareAPIError
from agent.db import reset_database

logger = get_logger(__name__)

# Voucher types we learn category history from
_BOOKKEEPING_TYPES = {
    "purchaseinvoice",
    "salesinvoice",
    "purchasecreditnote",
    "salescreditnote",
}


# ── Internal sync helpers ─────────────────────────────────────────────────────

def _sync_contacts(client: LexwareClient, since: Optional[str] = None) -> int:
    """
    Pull all contacts (or delta since `since`) and upsert into local DB.
    Returns count of contacts processed.
    """
    logger.info("Syncing contacts%s", f" (since {since})" if since else " (full)")
    contacts = client.get_all_contacts(updated_since=since)
    count = 0

    with get_db() as db:
        for c in contacts:
            company  = c.get("company") or {}
            person   = c.get("person")  or {}
            roles    = c.get("roles")   or {}
            billing  = (c.get("addresses", {}).get("billing") or [{}])[0]
            emails   = c.get("emailAddresses", {})
            phones   = c.get("phoneNumbers", {})

            # Build normalised name
            if company.get("name"):
                name = company["name"]
            else:
                first = person.get("firstName", "")
                last  = person.get("lastName", "")
                name  = f"{first} {last}".strip()

            # Extract first available email / phone
            email = (
                (emails.get("business") or [None])[0] or
                (emails.get("office")   or [None])[0] or
                (emails.get("private")  or [None])[0]
            )
            phone = (
                (phones.get("business") or [None])[0] or
                (phones.get("office")   or [None])[0] or
                (phones.get("mobile")   or [None])[0]
            )

            db.execute("""
                INSERT INTO contacts (
                    id, name, vat_id, tax_number,
                    street, zip, city, country_code,
                    email, phone,
                    role_customer, role_vendor,
                    allow_tax_free, version,
                    last_synced_at, raw_json
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name           = excluded.name,
                    vat_id         = excluded.vat_id,
                    tax_number     = excluded.tax_number,
                    street         = excluded.street,
                    zip            = excluded.zip,
                    city           = excluded.city,
                    country_code   = excluded.country_code,
                    email          = excluded.email,
                    phone          = excluded.phone,
                    role_customer  = excluded.role_customer,
                    role_vendor    = excluded.role_vendor,
                    allow_tax_free = excluded.allow_tax_free,
                    version        = excluded.version,
                    last_synced_at = excluded.last_synced_at,
                    raw_json       = excluded.raw_json
            """, (
                c["id"],
                name,
                company.get("vatRegistrationId"),
                company.get("taxNumber"),
                billing.get("street"),
                billing.get("zip"),
                billing.get("city"),
                billing.get("countryCode", "DE"),
                email,
                phone,
                1 if "customer" in roles else 0,
                1 if "vendor"   in roles else 0,
                1 if company.get("allowTaxFreeInvoices") else 0,
                c.get("version", 0),
                datetime.now(timezone.utc).isoformat(),
                json.dumps(c),
            ))
            count += 1

    logger.info("Contacts synced: %d", count)
    return count


def _sync_posting_categories(client: LexwareClient) -> int:
    """
    Pull posting categories and upsert. Always a full refresh
    (no delta support on this endpoint).
    """
    logger.info("Syncing posting categories")
    categories = client.get_posting_categories()

    with get_db() as db:
        for cat in categories:
            db.execute("""
                INSERT INTO posting_categories
                    (id, name, type, split_allowed, group_name, contact_required)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    name             = excluded.name,
                    type             = excluded.type,
                    split_allowed    = excluded.split_allowed,
                    group_name       = excluded.group_name,
                    contact_required = excluded.contact_required
            """, (
                cat["id"],
                cat["name"],
                cat["type"],
                1 if cat.get("splitAllowed")    else 0,
                cat.get("groupName"),
                1 if cat.get("contactRequired") else 0,
            ))

    logger.info("Posting categories synced: %d", len(categories))
    return len(categories)


def _sync_articles(client: LexwareClient) -> int:
    """Pull all articles and upsert."""
    logger.info("Syncing articles")
    articles = client.get_all_articles()
    now = datetime.now(timezone.utc).isoformat()

    with get_db() as db:
        for a in articles:
            price = a.get("price") or {}
            db.execute("""
                INSERT INTO articles
                    (id, title, type, unit_name, tax_rate,
                     net_price, gross_price, article_number, last_synced_at)
                VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    title          = excluded.title,
                    type           = excluded.type,
                    unit_name      = excluded.unit_name,
                    tax_rate       = excluded.tax_rate,
                    net_price      = excluded.net_price,
                    gross_price    = excluded.gross_price,
                    article_number = excluded.article_number,
                    last_synced_at = excluded.last_synced_at
            """, (
                a["id"],
                a.get("title"),
                a.get("type"),
                a.get("unitName"),
                price.get("taxRate"),
                price.get("netPrice"),
                price.get("grossPrice"),
                a.get("articleNumber"),
                now,
            ))

    logger.info("Articles synced: %d", len(articles))
    return len(articles)


def _sync_payment_conditions(client: LexwareClient) -> int:
    """Pull payment conditions and upsert."""
    logger.info("Syncing payment conditions")
    conditions = client.get_payment_conditions()

    with get_db() as db:
        for pc in conditions:
            discount = pc.get("paymentDiscountConditions") or {}
            db.execute("""
                INSERT INTO payment_conditions
                    (id, label_template, payment_term_days,
                     discount_percentage, discount_range_days, is_org_default)
                VALUES (?,?,?,?,?,?)
                ON CONFLICT(id) DO UPDATE SET
                    label_template      = excluded.label_template,
                    payment_term_days   = excluded.payment_term_days,
                    discount_percentage = excluded.discount_percentage,
                    discount_range_days = excluded.discount_range_days,
                    is_org_default      = excluded.is_org_default
            """, (
                pc["id"],
                pc.get("paymentTermLabelTemplate"),
                pc.get("paymentTermDuration"),
                discount.get("discountPercentage"),
                discount.get("discountRange"),
                1 if pc.get("organizationDefault") else 0,
            ))

    logger.info("Payment conditions synced: %d", len(conditions))
    return len(conditions)


def _sync_voucher_history(
    client: LexwareClient, since: Optional[str] = None
) -> int:
    """
    Pull voucherlist → fetch full detail for bookkeeping vouchers →
    build contact_category_history.

    This is the most expensive sync operation (many API calls).
    On cold sync it processes the entire history.
    On hot sync it only processes vouchers updated since last sync.
    """
    logger.info(
        "Syncing voucher history%s", f" (since {since})" if since else " (full)"
    )
    learned = 0
    skipped = 0

    voucher_iter = client.get_voucherlist(updated_since=since)

    for v in voucher_iter:
        vtype      = v.get("voucherType", "")
        contact_id = v.get("contactId")

        # Only learn from bookkeeping vouchers with a proper contact
        if vtype not in _BOOKKEEPING_TYPES or not contact_id:
            skipped += 1
            continue

        try:
            detail = client.get_voucher(v["id"])
        except LexwareAPIError as e:
            logger.warning("Could not fetch voucher %s: %s", v["id"], e)
            continue

        tax_type = detail.get("taxType", "gross")
        used_at  = (v.get("voucherDate") or "")[:10]  # YYYY-MM-DD

        with get_db() as db:
            for item in detail.get("voucherItems", []):
                cat_id = item.get("categoryId")
                if not cat_id:
                    continue

                db.execute("""
                    INSERT INTO contact_category_history
                        (contact_id, category_id, voucher_type,
                         tax_type, usage_count, last_used_at)
                    VALUES (?,?,?,?,1,?)
                    ON CONFLICT(contact_id, category_id, voucher_type)
                    DO UPDATE SET
                        usage_count  = usage_count + 1,
                        tax_type     = excluded.tax_type,
                        last_used_at = excluded.last_used_at
                """, (contact_id, cat_id, vtype, tax_type, used_at))

                # Keep contact.default_category_id pointing to highest-usage cat
                top = db.execute("""
                    SELECT category_id FROM contact_category_history
                    WHERE  contact_id   = ?
                    AND    voucher_type = ?
                    ORDER  BY usage_count DESC
                    LIMIT  1
                """, (contact_id, vtype)).fetchone()

                if top:
                    db.execute("""
                        UPDATE contacts
                        SET    default_category_id = ?
                        WHERE  id = ?
                    """, (top["category_id"], contact_id))

        learned += 1

        # Polite logging every 100 vouchers so the user sees progress
        if learned % 100 == 0:
            logger.info("  Voucher history: %d learned so far...", learned)

    logger.info(
        "Voucher history sync done. Learned: %d  Skipped: %d",
        learned, skipped
    )
    return learned


# ── Public API ────────────────────────────────────────────────────────────────

def cold_sync() -> dict:
    """
    Full historical sync. Pulls everything from Lexware from the beginning.
    Should only be run once (or when FORCE_COLD_SYNC=true in .env).

    Returns a stats dict:
        {contacts, categories, articles, payment_conditions, vouchers_learned}
    """
    logger.info("=" * 50)
    logger.info("COLD SYNC starting")
    logger.info("=" * 50)
    start = time.time()
    client = LexwareClient()

    stats: dict = {}

    stats["contacts"]           = _sync_contacts(client)
    stats["categories"]         = _sync_posting_categories(client)
    stats["articles"]           = _sync_articles(client)
    stats["payment_conditions"] = _sync_payment_conditions(client)
    stats["vouchers_learned"]   = _sync_voucher_history(client)

    now = datetime.now(timezone.utc).isoformat()
    set_sync_state("last_full_sync_at",      now)
    set_sync_state("last_contact_sync_at",   now)
    set_sync_state("last_voucher_sync_at",   now)
    set_sync_state("total_vouchers_learned", str(stats["vouchers_learned"]))

    elapsed = round(time.time() - start, 1)
    logger.info("COLD SYNC complete in %ss. Stats: %s", elapsed, stats)
    return stats


def hot_sync() -> dict:
    """
    Delta sync. Pulls only what changed since last sync.
    Called at the start of every invoice processing session.

    Returns a stats dict:
        {contacts_updated, vouchers_learned}
    """
    logger.info("Hot sync starting")
    start  = time.time()
    client = LexwareClient()

    # Check if cold sync has ever been done
    last_full = get_sync_state("last_full_sync_at")
    if not last_full:
        logger.warning(
            "No cold sync found — running cold sync first. "
            "This may take a while."
        )
        return cold_sync()

    since_contacts = get_sync_state("last_contact_sync_at")
    since_vouchers = get_sync_state("last_voucher_sync_at")

    # Lexware voucherlist and contacts filters only accept yyyy-MM-dd,
    # not full ISO timestamps — truncate to date portion only
    if since_contacts:
        since_contacts = since_contacts[:10]
    if since_vouchers:
        since_vouchers = since_vouchers[:10]

    stats: dict = {}

    # Contacts delta
    stats["contacts_updated"] = _sync_contacts(client, since=since_contacts)

    # Categories always refreshed (they can change without notification)
    _sync_posting_categories(client)

    # Voucher history delta
    stats["vouchers_learned"] = _sync_voucher_history(
        client, since=since_vouchers
    )

    now = datetime.now(timezone.utc).isoformat()
    set_sync_state("last_contact_sync_at", now)
    set_sync_state("last_voucher_sync_at", now)

    # Update total learned count
    prev_total = int(get_sync_state("total_vouchers_learned") or "0")
    set_sync_state(
        "total_vouchers_learned",
        str(prev_total + stats["vouchers_learned"])
    )

    elapsed = round(time.time() - start, 1)
    logger.info("Hot sync complete in %ss. Stats: %s", elapsed, stats)
    return stats


# ── Reset helpers ────────────────────────────────────────────────────────────

def clear_logs() -> None:
    """Delete all log files under the configured log directory."""
    log_dir = cfg.LOG_PATH.parent
    if not log_dir.exists():
        return

    removed = 0
    for log_file in log_dir.glob("*.log*"):
        log_file.unlink(missing_ok=True)
        removed += 1
    logger.info("Cleared %d log file(s) in %s", removed, log_dir)


def reset_and_cold_sync() -> dict:
    """
    Clear logs, wipe the local DB cache, then run a fresh cold sync.
    Intended for “start from scratch” scenarios.
    """
    logger.warning("Resetting logs and database before cold sync...")
    clear_logs()
    reset_database()
    return cold_sync()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Lexware sync utilities")
    parser.add_argument("--cold", action="store_true", help="Run a full cold sync")
    parser.add_argument("--hot",  action="store_true", help="Run a delta hot sync")
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Clear logs and database, then run a cold sync",
    )

    args = parser.parse_args()

    if args.reset:
        stats = reset_and_cold_sync()
    elif args.cold:
        stats = cold_sync()
    elif args.hot:
        stats = hot_sync()
    else:
        parser.print_help()
        raise SystemExit(1)

    logger.info("Sync finished. Stats: %s", stats)


