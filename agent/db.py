"""
agent/db.py
───────────
Database connection manager and low-level helpers.
All agent modules import get_db() from here — never open
sqlite3 connections directly.

Usage:
    from agent.db import get_db
    with get_db() as db:
        rows = db.execute("SELECT ...").fetchall()
"""

import sqlite3
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from config import cfg, get_logger

logger = get_logger(__name__)


# ── Connection factory ────────────────────────────────────────────────────────
@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """
    Context manager that yields a sqlite3 connection with
    WAL mode, foreign keys enabled, and row_factory set to
    sqlite3.Row (columns accessible by name).

    Commits on clean exit, rolls back on exception.

    Example:
        with get_db() as db:
            db.execute("INSERT INTO ...")
        # auto-committed
    """
    cfg.DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(cfg.DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── Schema initialisation ─────────────────────────────────────────────────────
def init_db() -> None:
    """
    Create all tables from schema.sql if they don't exist.
    Safe to call on every startup — uses CREATE TABLE IF NOT EXISTS.
    """
    schema_path = Path(__file__).parent.parent / "db" / "schema.sql"
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")

    sql = schema_path.read_text(encoding="utf-8")
    with get_db() as db:
        db.executescript(sql)
    logger.info("Database schema initialised: %s", cfg.DB_PATH)


def reset_database() -> None:
    """
    Drop the current SQLite database (including WAL/SHM) and recreate schema.
    Use with care — this deletes all cached sync data.
    """
    for suffix in ("", "-wal", "-shm"):
        path = Path(str(cfg.DB_PATH) + suffix)
        if path.exists():
            path.unlink()
            logger.info("Removed database file: %s", path)

    # Recreate schema so downstream code can run immediately
    init_db()


# ── Sync state helpers ────────────────────────────────────────────────────────
def get_sync_state(key: str) -> str | None:
    with get_db() as db:
        row = db.execute(
            "SELECT value FROM sync_state WHERE key = ?", (key,)
        ).fetchone()
    return row["value"] if row else None


def set_sync_state(key: str, value: str) -> None:
    with get_db() as db:
        db.execute(
            "INSERT OR REPLACE INTO sync_state (key, value) VALUES (?, ?)",
            (key, value)
        )


# ── Duplicate detection ───────────────────────────────────────────────────────
def pdf_hash(pdf_path: str | Path) -> str:
    """Return the SHA-256 hex digest of a PDF file."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def is_already_processed(file_hash: str) -> bool:
    """Return True if this PDF (by hash) was already successfully processed."""
    with get_db() as db:
        row = db.execute(
            "SELECT voucher_status FROM processed_invoices WHERE pdf_hash = ?",
            (file_hash,)
        ).fetchone()
    # Only skip if it was posted (open or unchecked) — retry if it failed
    if row and row["voucher_status"] in ("open", "unchecked"):
        return True
    return False


def record_processed_invoice(
    pdf_filename:   str,
    pdf_hash_val:   str,
    voucher_id:     str | None,
    voucher_status: str,
    contact_id:     str | None,
    contact_name:   str | None,
    category_id:    str | None,
    confidence:     float,
    match_signals:  list,
    claude_called:  bool,
    claude_tokens:  int,
    error_message:  str | None,
    processed_at:   str,
) -> None:
    with get_db() as db:
        db.execute("""
            INSERT OR REPLACE INTO processed_invoices (
                pdf_filename, pdf_hash, voucher_id, voucher_status,
                contact_id, contact_name, category_id, confidence,
                match_signals, claude_called, claude_tokens,
                error_message, processed_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            pdf_filename, pdf_hash_val, voucher_id, voucher_status,
            contact_id, contact_name, category_id, confidence,
            json.dumps(match_signals), 1 if claude_called else 0,
            claude_tokens, error_message, processed_at
        ))


# ── Learning helpers ──────────────────────────────────────────────────────────
def update_category_history(
    contact_id:   str,
    category_id:  str,
    voucher_type: str,
    tax_type:     str,
    used_at:      str,
) -> None:
    """
    Increment usage count for a (contact, category, voucher_type) triplet.
    Called after every successfully posted voucher.
    """
    with get_db() as db:
        db.execute("""
            INSERT INTO contact_category_history
                (contact_id, category_id, voucher_type, tax_type, usage_count, last_used_at)
            VALUES (?, ?, ?, ?, 1, ?)
            ON CONFLICT(contact_id, category_id, voucher_type) DO UPDATE SET
                usage_count  = usage_count + 1,
                tax_type     = excluded.tax_type,
                last_used_at = excluded.last_used_at
        """, (contact_id, category_id, voucher_type, tax_type, used_at))

        # Update default_category_id on contacts to the highest-count category
        top = db.execute("""
            SELECT category_id FROM contact_category_history
            WHERE  contact_id = ? AND voucher_type = ?
            ORDER  BY usage_count DESC
            LIMIT  1
        """, (contact_id, voucher_type)).fetchone()

        if top:
            db.execute("""
                UPDATE contacts
                SET    default_category_id = ?,
                       invoice_count       = invoice_count + 1
                WHERE  id = ?
            """, (top["category_id"], contact_id))


def update_category_usage_count(category_id: str) -> None:
    """Increment org-wide usage count for a posting category."""
    with get_db() as db:
        db.execute("""
            UPDATE posting_categories
            SET usage_count = usage_count + 1
            WHERE id = ?
        """, (category_id,))
