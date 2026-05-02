import gzip
import json
import logging
import os
import random
import re
import string
import time
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional
import html as html_module


logger = logging.getLogger(__name__)

EDGAR_BASE = "https://www.sec.gov"
DATA_API = "https://data.sec.gov"

_CIK_CACHE = {}
_TICKERS_CACHE = None
_TICKERS_FETCH_TIME = 0
_MIN_TICKERS_CACHE_AGE = 3600

_FILING_CACHE_DIR = Path(os.path.expanduser("~"), ".tradingagents", "cache", "sec_filings")
_FILING_CACHE_MAX_AGE = 90 * 24 * 3600  # 90 days in seconds

_FIREFOX_VERSIONS = ["120.0", "121.0", "122.0", "123.0", "124.0", "125.0", "126.0", "127.0", "128.0", "129.0"]
_PLATFORMS = [
    "X11; Linux x86_64",
    "X11; Ubuntu; Linux x86_64",
    "Windows NT 10.0; Win64; x64",
    "Macintosh; Intel Mac OS X 10.15",
    "Macintosh; Intel Mac OS X 14.0",
]


def _make_user_agent() -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=8))
    name = random.choice(["TradingAgents", "FinAnalysis", "MarketData", "StockScreener", "EquityResearch"])
    return f"{name}/{random.randint(1,9)}.{random.randint(0,9)} sample{suffix}@example.com"


def _sec_request(url: str, max_retries: int = 3) -> bytes:
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": _make_user_agent()})
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read()
                if resp.headers.get("Content-Encoding") == "gzip":
                    raw = gzip.decompress(raw)
                return raw
        except urllib.error.HTTPError as e:
            if e.code == 429 or e.code == 403:
                wait = 2 ** (attempt + 1) + 1
                logger.warning("SEC rate limit hit on %s, waiting %ds (attempt %d/%d)", url, wait, attempt + 1, max_retries)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise
    raise RuntimeError(f"Failed to fetch {url} after {max_retries} retries")


def _load_tickers_map() -> Optional[dict]:
    global _TICKERS_CACHE, _TICKERS_FETCH_TIME
    now = time.time()
    if _TICKERS_CACHE is not None and (now - _TICKERS_FETCH_TIME) < _MIN_TICKERS_CACHE_AGE:
        return _TICKERS_CACHE
    try:
        raw = _sec_request(f"{EDGAR_BASE}/files/company_tickers.json")
        _TICKERS_CACHE = json.loads(raw.decode("utf-8"))
        _TICKERS_FETCH_TIME = now
        return _TICKERS_CACHE
    except Exception:
        return _TICKERS_CACHE


def _ticker_to_cik(ticker: str) -> Optional[str]:
    global _CIK_CACHE

    ticker = ticker.upper().strip().lstrip("0")

    if ticker in _CIK_CACHE:
        return _CIK_CACHE[ticker]

    try:
        int(ticker)
        cik = ticker.zfill(10)
        _CIK_CACHE[ticker] = cik
        return cik
    except ValueError:
        pass

    all_tickers = _load_tickers_map()
    if all_tickers:
        for entry in all_tickers.values():
            if entry.get("ticker", "").upper() == ticker:
                cik = str(entry["cik_str"]).zfill(10)
                _CIK_CACHE[ticker] = cik
                return cik

    _CIK_CACHE[ticker] = None
    return None


def _find_latest_annual_filing(cik: str) -> Optional[dict]:
    url = f"{DATA_API}/submissions/CIK{cik}.json"
    try:
        data = json.loads(_sec_request(url))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return None

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A"}

    for i, form in enumerate(forms):
        if form in annual_forms:
            accession_clean = accessions[i].replace("-", "")
            doc_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
                f"{accession_clean}/{primary_docs[i]}"
            )
            return {
                "form_type": form,
                "filing_date": filing_dates[i],
                "accession": accessions[i],
                "document_url": doc_url,
            }

    return None


def _filing_cache_path(ticker: str, accession: str) -> Path:
    safe_ticker = ticker.upper().strip().lstrip("0").replace(".", "_")
    safe_accession = accession.replace("-", "").replace(" ", "_")
    return _FILING_CACHE_DIR / f"{safe_ticker}_{safe_accession}.json"


def _load_filing_cache(ticker: str, accession: str) -> Optional[str]:
    path = _filing_cache_path(ticker, accession)
    if not path.exists():
        return None
    try:
        age = time.time() - path.stat().st_mtime
        if age > _FILING_CACHE_MAX_AGE:
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("accession") != accession:
            return None
        return data.get("result")
    except (json.JSONDecodeError, OSError):
        return None


def _save_filing_cache(ticker: str, accession: str, result: str) -> None:
    try:
        _FILING_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        path = _filing_cache_path(ticker, accession)
        path.write_text(
            json.dumps({"accession": accession, "result": result}),
            encoding="utf-8",
        )
    except OSError:
        pass


def _html_to_text(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"</?(p|div|li|tr|h[1-6])[^>]*>", "\n", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_all_item_positions(text: str) -> list:
    all_items = []
    for m in re.finditer(
        r"item\s+(\d+[a-z]?)\s*[\.\)]\s+(.{0,200})", text, re.IGNORECASE
    ):
        item_num = m.group(1).lower()
        label = m.group(2).strip()
        all_items.append((m.start(), item_num, label))
    return all_items


def _extract_sections(text: str, max_chars: int = 6000) -> dict:
    all_items = _find_all_item_positions(text)

    section_defs = {
        "1": "Item 1: Business",
        "1a": "Item 1A: Risk Factors",
        "1b": "Item 1B: Unresolved Staff Comments",
        "2": "Item 2: Properties",
        "6": "Item 6: Selected Financial Data",
        "7": "Item 7: Management's Discussion & Analysis",
        "7a": "Item 7A: Quantitative Disclosures",
        "8": "Item 8: Financial Statements",
    }

    section_starts = {}
    for pos, item_num, label in all_items:
        if item_num in section_defs:
            section_starts[item_num] = pos

    if not section_starts:
        return {"Full Filing (truncated)": text[:max_chars]}

    sorted_items = sorted(all_items, key=lambda x: x[0])

    sections = {}
    for num, start_pos in section_starts.items():
        end_pos = len(text)
        for item_pos, item_num, _ in sorted_items:
            if item_pos > start_pos + 500 and item_num != num:
                end_pos = item_pos
                break

        chunk = text[start_pos:end_pos].strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars] + "\n\n[... section truncated ...]"
        sections[section_defs[num]] = chunk

    return sections


def get_10k_filing_data(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Fetch the most recent 10-K (or 20-F for foreign companies) annual report
    filing from SEC EDGAR. Returns key sections including Business, Risk Factors,
    and Management's Discussion & Analysis.
    """
    try:
        cik = _ticker_to_cik(ticker)
        if not cik:
            return (
                f"Could not find SEC CIK for ticker '{ticker}'. "
                f"The company may not file with the SEC (e.g., foreign private issuer "
                f"not on EDGAR, or a privately held company)."
            )

        filing = _find_latest_annual_filing(cik)
        if not filing:
            return (
                f"No 10-K or 20-F annual filings found for {ticker} (CIK: {cik}) "
                f"on SEC EDGAR. The company may be newly listed or may not file annual reports."
            )

        cached = _load_filing_cache(ticker, filing["accession"])
        if cached is not None:
            return cached

        html_data = _sec_request(filing["document_url"]).decode("utf-8", errors="replace")
        plain_text = _html_to_text(html_data)

        sections = _extract_sections(plain_text)

        lines = [
            f"# SEC Annual Filing: {ticker.upper()} ({filing['form_type']})",
            f"Filing Date: {filing['filing_date']}",
            f"Accession: {filing['accession']}",
            f"Source: {filing['document_url']}",
            f"Retrieved: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
        ]

        for section_name, content in sections.items():
            lines.append(f"## {section_name}")
            lines.append(content)
            lines.append("")

        result = "\n".join(lines)

        _save_filing_cache(ticker, filing["accession"], result)

        return result

    except urllib.error.URLError as e:
        return f"Error fetching SEC filing for {ticker}: Network error - {e}"
    except Exception as e:
        return f"Error fetching SEC filing for {ticker}: {e}"
