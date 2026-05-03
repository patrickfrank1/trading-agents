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


def _find_filings(
    cik: str,
    form_types: set,
    max_filings: int = 2,
    before_date: str = None,
) -> list:
    url = f"{DATA_API}/submissions/CIK{cik}.json"
    try:
        data = json.loads(_sec_request(url))
    except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError):
        return []

    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])
    filing_dates = recent.get("filingDate", [])

    results = []
    seen_accessions = set()

    for i, form in enumerate(forms):
        if form in form_types:
            if filing_dates[i] in seen_accessions:
                continue
            if before_date and filing_dates[i] > before_date:
                continue
            accession_clean = accessions[i].replace("-", "")
            doc_url = (
                f"{EDGAR_BASE}/Archives/edgar/data/{cik}/"
                f"{accession_clean}/{primary_docs[i]}"
            )
            results.append({
                "form_type": form,
                "filing_date": filing_dates[i],
                "accession": accessions[i],
                "document_url": doc_url,
            })
            seen_accessions.add(filing_dates[i])
            if len(results) >= max_filings:
                break

    return results


def _find_latest_annual_filing(cik: str) -> Optional[dict]:
    filings = _find_filings(cik, {"10-K", "10-K/A", "20-F", "20-F/A"}, max_filings=1)
    return filings[0] if filings else None


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
    text = re.sub(r"<div[^>]*display\s*:\s*none[^>]*>.*?</div>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<ix:.*?>", "", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = html_module.unescape(text)
    text = re.sub(r"\u00a0", " ", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _find_all_item_headers(text: str) -> list:
    positions = []
    for m in re.finditer(r"item\s+(\d+[a-z]?)\s*\.\s*", text, re.IGNORECASE):
        item_num = m.group(1).lower()
        positions.append((m.start(), m.end(), item_num))
    return positions


def _extract_sections(text: str, max_chars: int = 6000, section_defs: dict = None) -> dict:
    if section_defs is None:
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

    headers = _find_all_item_headers(text)

    if not headers:
        return {"Full Filing (truncated)": text[:max_chars]}

    sorted_headers = sorted(headers, key=lambda x: x[0])

    def section_length(start: int) -> int:
        for h_start, _, h_num in sorted_headers:
            if h_start > start + 200:
                return h_start - start
        return len(text) - start

    best = {}
    for pos, end, num in headers:
        if num not in section_defs:
            continue
        slen = section_length(end)
        if num not in best or slen > best[num][1]:
            best[num] = (pos, slen)

    if not best:
        return {"Full Filing (truncated)": text[:max_chars]}

    sections = {}
    for num, (pos, _) in best.items():
        end_pos = len(text)
        for h_start, _, h_num in sorted_headers:
            if h_start > pos + 200 and h_num != num:
                end_pos = h_start
                break

        chunk = text[pos:end_pos].strip()
        if len(chunk) > max_chars:
            chunk = chunk[:max_chars] + "\n\n[... section truncated ...]"
        sections[section_defs[num]] = chunk

    return sections


_10Q_SECTION_DEFS = {
    "1": "Item 1: Financial Statements",
    "2": "Item 2: Management's Discussion & Analysis",
    "3": "Item 3: Quantitative and Qualitative Disclosures About Market Risk",
    "4": "Item 4: Controls and Procedures",
    "1a": "Part I, Item 1A: Risk Factors (if included)",
}

_8K_SECTION_DEFS = {
    "1.01": "Item 1.01: Entry into a Material Definitive Agreement",
    "1.02": "Item 1.02: Termination of a Material Definitive Agreement",
    "2.01": "Item 2.01: Completion of Acquisition or Disposition of Assets",
    "2.02": "Item 2.02: Results of Operations and Financial Condition",
    "2.03": "Item 2.03: Creation of a Direct Financial Obligation",
    "2.04": "Item 2.04: Triggering Events That Accelerate or Increase a Direct Financial Obligation",
    "2.05": "Item 2.05: Costs Associated with Exit or Disposal Activities",
    "2.06": "Item 2.06: Material Impairments",
    "5.02": "Item 5.02: Departure of Directors or Certain Officers",
    "5.03": "Item 5.03: Amendments to Articles of Incorporation or Bylaws",
    "7.01": "Item 7.01: Regulation FD Disclosure",
    "8.01": "Item 8.01: Other Events",
    "9.01": "Item 9.01: Financial Statements and Exhibits",
}

_8K_ITEM_PATTERN = re.compile(
    r"item\s+(\d+\.\d+)\s*[\.\)]\s+(.{0,200})", re.IGNORECASE
)


def _fetch_and_format_filing(ticker: str, filing: dict, section_defs: dict, max_chars: int) -> str:
    cached = _load_filing_cache(ticker, filing["accession"])
    if cached is not None:
        return cached

    html_data = _sec_request(filing["document_url"]).decode("utf-8", errors="replace")
    plain_text = _html_to_text(html_data)
    sections = _extract_sections(plain_text, max_chars=max_chars, section_defs=section_defs)

    lines = [
        f"# SEC Filing: {ticker.upper()} ({filing['form_type']})",
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


def get_10k_filing_data(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Fetch the last 2 available 10-K (or 20-F for foreign companies) annual report
    filings from SEC EDGAR. Returns key sections including Business, Risk Factors,
    Management's Discussion & Analysis, and Financial Statements.
    """
    try:
        cik = _ticker_to_cik(ticker)
        if not cik:
            return (
                f"Could not find SEC CIK for ticker '{ticker}'. "
                f"The company may not file with the SEC (e.g., foreign private issuer "
                f"not on EDGAR, or a privately held company)."
            )

        filings = _find_filings(
            cik,
            {"10-K", "20-F"},
            max_filings=2,
            before_date=curr_date,
        )

        if not filings:
            return (
                f"No 10-K or 20-F annual filings found for {ticker} (CIK: {cik}) "
                f"on SEC EDGAR. The company may be newly listed or may not file annual reports."
            )

        all_results = []
        for filing in filings:
            all_results.append(
                _fetch_and_format_filing(
                    ticker, filing,
                    section_defs=None,
                    max_chars=6000,
                )
            )

        return "\n\n---\n\n".join(all_results)

    except urllib.error.URLError as e:
        return f"Error fetching SEC filing for {ticker}: Network error - {e}"
    except Exception as e:
        return f"Error fetching SEC filing for {ticker}: {e}"


def get_10q_filing_data(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Fetch the last 2 available 10-Q quarterly report filings from SEC EDGAR.
    Returns Financial Statements, MD&A, and Risk Factor sections.
    """
    try:
        cik = _ticker_to_cik(ticker)
        if not cik:
            return (
                f"Could not find SEC CIK for ticker '{ticker}'. "
                f"The company may not file with the SEC."
            )

        filings = _find_filings(
            cik,
            {"10-Q", "10-Q/A"},
            max_filings=2,
            before_date=curr_date,
        )

        if not filings:
            return (
                f"No 10-Q quarterly filings found for {ticker} (CIK: {cik}) "
                f"on SEC EDGAR."
            )

        all_results = []
        for filing in filings:
            all_results.append(
                _fetch_and_format_filing(
                    ticker, filing,
                    section_defs=_10Q_SECTION_DEFS,
                    max_chars=6000,
                )
            )

        return "\n\n---\n\n".join(all_results)

    except urllib.error.URLError as e:
        return f"Error fetching SEC 10-Q filing for {ticker}: Network error - {e}"
    except Exception as e:
        return f"Error fetching SEC 10-Q filing for {ticker}: {e}"


def get_8k_filing_data(
    ticker: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "current date in YYYY-MM-DD format"] = None,
) -> str:
    """Fetch the last 2 available 8-K current report filings from SEC EDGAR.
    8-K filings disclose major corporate events (earnings, acquisitions, leadership
    changes, defaults, etc.). Returns all identified sections from each filing.
    """
    try:
        cik = _ticker_to_cik(ticker)
        if not cik:
            return (
                f"Could not find SEC CIK for ticker '{ticker}'. "
                f"The company may not file with the SEC."
            )

        filings = _find_filings(
            cik,
            {"8-K", "8-K/A"},
            max_filings=2,
            before_date=curr_date,
        )

        if not filings:
            return (
                f"No 8-K current report filings found for {ticker} (CIK: {cik}) "
                f"on SEC EDGAR."
            )

        all_results = []
        for filing in filings:
            cached = _load_filing_cache(ticker, filing["accession"])
            if cached is not None:
                all_results.append(cached)
                continue

            html_data = _sec_request(filing["document_url"]).decode("utf-8", errors="replace")
            plain_text = _html_to_text(html_data)

            sections = {}
            for m in _8K_ITEM_PATTERN.finditer(plain_text):
                item_num = m.group(1).lower()
                if item_num in _8K_SECTION_DEFS:
                    section_start = m.start()

                    next_m = None
                    for nm in _8K_ITEM_PATTERN.finditer(plain_text[m.start() + 10:]):
                        next_m = nm
                        break

                    if next_m:
                        section_end = m.start() + 10 + next_m.start()
                    else:
                        section_end = len(plain_text)

                    chunk = plain_text[section_start:section_end].strip()
                    if len(chunk) > 3000:
                        chunk = chunk[:3000] + "\n\n[... section truncated ...]"
                    sections[_8K_SECTION_DEFS[item_num]] = chunk

            if not sections:
                truncated = plain_text[:6000]
                if len(plain_text) > 6000:
                    truncated += "\n\n[... filing truncated ...]"
                sections = {"Full Filing (truncated)": truncated}

            lines = [
                f"# SEC 8-K Filing: {ticker.upper()} ({filing['form_type']})",
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
            all_results.append(result)

        return "\n\n---\n\n".join(all_results)

    except urllib.error.URLError as e:
        return f"Error fetching SEC 8-K filing for {ticker}: Network error - {e}"
    except Exception as e:
        return f"Error fetching SEC 8-K filing for {ticker}: {e}"
