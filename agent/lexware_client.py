"""
agent/lexware_client.py
───────────────────────
Thin wrapper around the Lexware REST API.

Rate limiting (from official Lexware API docs):
  - Hard limit:  2 requests per second (token bucket algorithm)
  - Our target:  1 req/sec = 1.1s between requests (45% buffer below limit)
                 This guarantees we never approach the hard limit even under
                 bursts or network jitter.
  - On 429:      exponential backoff — wait, then retry up to 5 times
  - Voucher detail fetches (cold sync): 1.2s gap (burst protection)

Lexware docs state:
  "Enforcing the mentioned limits without any buffer will commonly
   result in rate limited requests."

Usage:
    from agent.lexware_client import LexwareClient
    client = LexwareClient()
    contacts = client.get_all_contacts()
"""

import time
import json
import requests
from typing import Any, Generator

from config import cfg, get_logger

logger = get_logger(__name__)

# ── Rate limit constants (aligned with Lexware API docs) ─────────────────────
# Lexware hard limit: 2 req/sec. We target 1 req/sec (45% buffer) for safety.
_REQUEST_DELAY_SECONDS  = 1.10   # standard calls (~1 req/sec — well below the 2/sec limit)
_VOUCHER_DETAIL_DELAY   = 1.20   # voucher/{id} calls during cold sync (slightly more gap)

# Exponential backoff for 429 responses
_BACKOFF_BASE           = 2.0    # seconds for first retry
_BACKOFF_MULTIPLIER     = 2.0    # doubles each retry: 2s, 4s, 8s, 16s, 32s
_BACKOFF_MAX_RETRIES    = 5      # give up after 5 retries (~62s total wait)
_BACKOFF_MAX_WAIT       = 32.0   # cap individual wait at 32s


class LexwareAPIError(Exception):
    """Raised when the Lexware API returns a non-2xx response."""
    def __init__(self, status_code: int, message: str):
        self.status_code = status_code
        super().__init__(f"Lexware API {status_code}: {message}")


class LexwareClient:
    """
    Authenticated Lexware API client.
    Instantiate once and reuse — headers are set on init.
    Handles rate limiting and exponential backoff automatically.
    """

    def __init__(self, api_key: str | None = None):
        key = api_key or cfg.LEXWARE_API_KEY
        self._headers = {
            "Authorization": f"Bearer {key}",
            "Accept":        "application/json",
            "Content-Type":  "application/json",
        }
        self._base = cfg.LEXWARE_BASE_URL

    # ── Core request with exponential backoff ─────────────────────────────
    def _request_with_backoff(
        self,
        method:  str,
        url:     str,
        delay:   float = _REQUEST_DELAY_SECONDS,
        **kwargs,
    ) -> requests.Response:
        """
        Make an HTTP request respecting the Lexware rate limit.

        - Always sleeps `delay` seconds before the request (token bucket style)
        - On 429: retries with exponential backoff up to _BACKOFF_MAX_RETRIES
        - On other non-2xx: raises LexwareAPIError immediately
        """
        time.sleep(delay)

        wait = _BACKOFF_BASE
        for attempt in range(_BACKOFF_MAX_RETRIES + 1):
            r = requests.request(method, url, headers=self._headers,
                                 timeout=30, **kwargs)
            if r.status_code == 429:
                if attempt >= _BACKOFF_MAX_RETRIES:
                    raise LexwareAPIError(
                        429,
                        f"Rate limit exceeded after {_BACKOFF_MAX_RETRIES} retries"
                    )
                actual_wait = min(wait, _BACKOFF_MAX_WAIT)
                logger.warning(
                    "Rate limit hit (429) — backing off %.1fs (attempt %d/%d)",
                    actual_wait, attempt + 1, _BACKOFF_MAX_RETRIES
                )
                time.sleep(actual_wait)
                wait *= _BACKOFF_MULTIPLIER
                time.sleep(delay)  # respect rate limit after backoff too
                continue
            return r

        # Should never reach here
        raise LexwareAPIError(429, "Max retries exceeded")

    # ── Private helpers ───────────────────────────────────────────────────
    def _get(
        self,
        endpoint: str,
        params:   dict | None = None,
        delay:    float = _REQUEST_DELAY_SECONDS,
    ) -> Any:
        url = f"{self._base}/{endpoint.lstrip('/')}"
        r = self._request_with_backoff("GET", url, delay=delay,
                                        params=params or {})
        if not r.ok:
            raise LexwareAPIError(r.status_code, r.text[:300])
        return r.json()

    def _post(self, endpoint: str, payload: dict) -> Any:
        url = f"{self._base}/{endpoint.lstrip('/')}"
        r = self._request_with_backoff(
            "POST", url,
            delay=_REQUEST_DELAY_SECONDS,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        )
        if not r.ok:
            raise LexwareAPIError(r.status_code, r.text[:500])
        return r.json()

    def _post_file(self, endpoint: str, pdf_path: str) -> Any:
        """Upload a PDF file to a voucher."""
        url = f"{self._base}/{endpoint.lstrip('/')}"
        # Remove Content-Type — requests sets multipart boundary automatically
        headers = {k: v for k, v in self._headers.items()
                   if k != "Content-Type"}
        time.sleep(_REQUEST_DELAY_SECONDS)
        with open(pdf_path, "rb") as f:
            r = requests.post(
                url, headers=headers,
                files={"file": (pdf_path, f, "application/pdf")},
                timeout=60,
            )
        if not r.ok:
            raise LexwareAPIError(r.status_code, r.text[:300])
        return r.json() if r.text else {}

    def _paginate(
        self,
        endpoint:  str,
        params:    dict | None = None,
        page_size: int = 250,
        delay:     float = _REQUEST_DELAY_SECONDS,
    ) -> Generator[dict, None, None]:
        """
        Yield every item from a paged Lexware endpoint.
        Handles both list responses and {content: [...]} responses.
        Uses max page size (250) to minimise number of requests made.
        """
        p = dict(params or {})
        p["size"] = page_size
        page = 0
        while True:
            p["page"] = page
            data = self._get(endpoint, params=p, delay=delay)

            # Some endpoints (e.g. posting-categories) return a plain list
            if isinstance(data, list):
                yield from data
                break

            items = data.get("content", [])
            yield from items

            if data.get("last", True) or not items:
                break
            page += 1

    # ── Contacts ──────────────────────────────────────────────────────────
    def get_all_contacts(self, updated_since: str | None = None) -> list[dict]:
        params = {}
        if updated_since:
            params["updatedDateFrom"] = updated_since
        return list(self._paginate("contacts", params))

    def create_contact(self, payload: dict) -> dict:
        """POST /v1/contacts — returns {id, resourceUri, ...}"""
        return self._post("contacts", payload)

    # ── Posting categories ────────────────────────────────────────────────
    def get_posting_categories(self) -> list[dict]:
        return list(self._paginate("posting-categories"))

    # ── Articles ──────────────────────────────────────────────────────────
    def get_all_articles(self) -> list[dict]:
        return list(self._paginate("articles"))

    # ── Payment conditions ────────────────────────────────────────────────
    def get_payment_conditions(self) -> list[dict]:
        data = self._get("payment-conditions")
        return data if isinstance(data, list) else data.get("content", [])

    # ── Voucherlist ───────────────────────────────────────────────────────
    def get_voucherlist(
        self,
        voucher_type:   str = "any",
        voucher_status: str = "any",
        updated_since:  str | None = None,
    ) -> Generator[dict, None, None]:
        params: dict = {
            "voucherType":   voucher_type,
            "voucherStatus": voucher_status,
        }
        if updated_since:
            params["updatedDateFrom"] = updated_since
        yield from self._paginate("voucherlist", params)

    # ── Individual voucher detail ─────────────────────────────────────────
    def get_voucher(self, voucher_id: str) -> dict:
        """
        Fetch full voucher detail.
        Uses a slightly longer delay (_VOUCHER_DETAIL_DELAY) because cold
        sync calls this in a tight loop for every voucher — most likely
        to hit the rate limit.
        """
        return self._get(
            f"vouchers/{voucher_id}",
            delay=_VOUCHER_DETAIL_DELAY,
        )

    # ── Create voucher ────────────────────────────────────────────────────
    def create_voucher(self, payload: dict) -> dict:
        """POST /v1/vouchers — returns {id, resourceUri, ...}"""
        logger.debug(
            "Creating voucher: type=%s status=%s",
            payload.get("type"), payload.get("voucherStatus")
        )
        return self._post("vouchers", payload)

    # ── Attach PDF to voucher ─────────────────────────────────────────────
    def attach_pdf(self, voucher_id: str, pdf_path: str) -> dict:
        """POST /v1/vouchers/{id}/files"""
        logger.debug("Attaching PDF to voucher %s", voucher_id)
        return self._post_file(f"vouchers/{voucher_id}/files", pdf_path)

    # ── Profile ───────────────────────────────────────────────────────────
    def get_profile(self) -> dict:
        return self._get("profile")