"""
config.py
─────────
Central configuration loader. Reads .env file and exposes
all settings as a typed Config dataclass. Import this
at the top of every agent module.

Usage:
    from config import cfg
    print(cfg.LEXWARE_API_KEY)
"""

import os
import logging
from pathlib import Path
from dataclasses import dataclass
from dotenv import load_dotenv

# Load .env from project root (two levels up from agent/)
_root = Path(__file__).parent
load_dotenv(_root / ".env")


@dataclass
class Config:
    # ── API keys ───────────────────────────────────────────────────────────
    LEXWARE_API_KEY:    str
    ANTHROPIC_API_KEY:  str

    # ── Paths ──────────────────────────────────────────────────────────────
    DB_PATH:         Path
    PDF_INBOX:       Path   # raw PDFs land here (before OCR)
    PDF_INBOX_OCR:   Path   # PDFs after OCR pre-processing (ready for processor)
    PDF_PROCESSED:   Path
    PDF_FAILED:      Path
    LOG_PATH:        Path

    # ── Behaviour ──────────────────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float   # below this → post as unchecked
    CLAUDE_MODEL:         str
    FORCE_COLD_SYNC:      bool

    # ── Lexware ────────────────────────────────────────────────────────────
    LEXWARE_BASE_URL: str

    # ── Logging ────────────────────────────────────────────────────────────
    LOG_LEVEL: str

    # ── Your company ───────────────────────────────────────────────────────
    # Your company's VAT ID — invoices carrying this ID are from YOU (the buyer)
    # and must never be returned as the vendor during AI extraction.
    OWN_VAT_ID: str = ""

    # ── Constants (not from .env) ──────────────────────────────────────────
    # "Zu prüfen" categoryId — the catch-all for uncertain vouchers
    ZU_PRUEFEN_CATEGORY_ID: str = "32b4c1d5-050f-4b80-9377-a8e98384ebee"

    # Contact match score thresholds
    MATCH_SCORE_AUTO:   int = 100   # IBAN / VAT ID → instant auto-match
    MATCH_SCORE_ACCEPT: int = 60    # Fuzzy name / address → accepted match
    MATCH_SCORE_MIN:    int = 40    # Below this → no match, create new contact


def _bool(val: str) -> bool:
    return val.strip().lower() in ("1", "true", "yes")


def _load() -> Config:
    missing = []
    for key in ("LEXWARE_API_KEY", "ANTHROPIC_API_KEY"):
        if not os.getenv(key):
            missing.append(key)
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}\n"
            f"Copy .env.example → .env and fill in your API keys."
        )

    root = Path(__file__).parent
    return Config(
        LEXWARE_API_KEY    = os.environ["LEXWARE_API_KEY"].strip(),
        ANTHROPIC_API_KEY  = os.environ["ANTHROPIC_API_KEY"].strip(),

        DB_PATH        = root / os.getenv("DB_PATH",        "db/lexware.db"),
        PDF_INBOX      = root / os.getenv("PDF_INBOX",      "pdfs/inbox"),
        PDF_INBOX_OCR  = root / os.getenv("PDF_INBOX_OCR",  "pdfs/inbox_ocr"),
        PDF_PROCESSED  = root / os.getenv("PDF_PROCESSED",  "pdfs/processed"),
        PDF_FAILED     = root / os.getenv("PDF_FAILED",     "pdfs/failed"),
        LOG_PATH       = root / os.getenv("LOG_PATH",       "logs/automation.log"),

        OWN_VAT_ID           = os.getenv("OWN_VAT_ID", "").strip(),

        CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.75")),
        CLAUDE_MODEL         = os.getenv("CLAUDE_MODEL", "claude-haiku-4-5-20251001"),
        FORCE_COLD_SYNC      = _bool(os.getenv("FORCE_COLD_SYNC", "false")),

        LEXWARE_BASE_URL = "https://api.lexware.io/v1",

        LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper(),
    )


# ── Singleton ─────────────────────────────────────────────────────────────────
cfg = _load()


# ── Logger factory ────────────────────────────────────────────────────────────
def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger that writes to both console and the log file.
    Call once per module:  logger = get_logger(__name__)
    """
    cfg.LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    if logger.handlers:          # avoid duplicate handlers on re-import
        return logger

    logger.setLevel(getattr(logging, cfg.LOG_LEVEL, logging.INFO))

    fmt = logging.Formatter(
        "%(asctime)s  %(levelname)-8s  %(name)-20s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # file
    fh = logging.FileHandler(cfg.LOG_PATH, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger
