"""No-login earnings-call transcript tool.

Earnings-call transcripts expose management guidance quality, tone shifts,
and execution vs. prior promises — none of which the filing or news tools
cover. There is no keyless public transcript API, so this module scrapes
The Motley Fool's public transcript pages (no login required):

1. ``_search_transcript_urls`` resolves transcript URLs for a ticker via
   DuckDuckGo (``ddgs`` package, already a dependency of ``web_search``).
2. ``_fetch_page`` downloads the page HTML with a browser-like UA.
3. ``_parse_transcript`` extracts the structured sections Motley Fool
   publishes (DATE, CALL PARTICIPANTS, RISKS, TAKEAWAYS) plus the full
   call body, then picks a bounded excerpt of prepared remarks with a
   bias toward guidance/outlook paragraphs.

Every step is defensive: if Motley Fool changes its layout, the parser
degrades to "N/A" instead of crashing the pipeline (``@safe_tool``).
"""

from __future__ import annotations

import logging
import re
from typing import Annotated

from langchain_core.tools import tool

from tradingagents.agents.utils.tool_errors import safe_tool

logger = logging.getLogger("tradingagents.tools.transcripts")

_UA = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)

_TRANSCRIPT_URL_RE = re.compile(
    r"^https?://(?:www\.)?fool\.com/earnings/call-transcripts/[^?#]+/?$",
    re.IGNORECASE,
)

#: Paragraphs containing any of these keywords are always kept in the
#: remarks excerpt (guidance / outlook / capital allocation).
_GUIDANCE_RE = re.compile(
    r"\b(guidance|outlook|we expect|we anticipate|next quarter|full year|"
    r"fiscal year|we forecast|capital allocation|share repurchas|buyback|"
    r"margins? (?:will|are expected)|headwind|tailwind)\b",
    re.IGNORECASE,
)

#: Rough output budget per transcript so N transcripts stay within the
#: context window.
_MAX_CHARS_PER_TRANSCRIPT = 7000
_MAX_REMARK_PARAGRAPHS = 12


def _search_transcript_urls(ticker: str, limit: int) -> list[str]:
    """Resolve Motley Fool transcript URLs for *ticker* via DuckDuckGo.

    Returns up to *limit* URLs in relevance order (newest first, as DDG
    ranks recency high for this query).
    """
    try:
        from ddgs import DDGS
    except ImportError:
        return []

    query = (
        f"site:fool.com/earnings/call-transcripts {ticker.upper()} "
        "earnings call transcript"
    )
    urls: list[str] = []
    seen: set[str] = set()
    try:
        raw = DDGS().text(query, max_results=max(limit * 4, 16))
    except Exception as exc:
        logger.warning("Transcript URL search failed for %s: %s", ticker, exc)
        return []
    for item in raw or []:
        url = (item.get("href") or item.get("url") or "").strip()
        if url and url not in seen and _TRANSCRIPT_URL_RE.match(url):
            seen.add(url)
            urls.append(url)
        if len(urls) >= limit:
            break
    return urls


def _fetch_page(url: str) -> str:
    import requests

    resp = requests.get(url, headers={"User-Agent": _UA}, timeout=20)
    resp.raise_for_status()
    return resp.text


def _parse_transcript(html: str, url: str) -> dict | None:
    """Parse a Motley Fool transcript page into structured sections.

    Returns ``None`` when the expected container is missing (layout
    change) so the caller can degrade gracefully.
    """
    from parsel import Selector

    sel = Selector(text=html)
    container = sel.css("#article-body-transcript")
    if not container:
        return None
    body = container[0]

    title = (sel.css("title::text").get() or "").strip()
    title = re.sub(r"\s*\|\s*The Motley Fool\s*$", "", title)

    # Newer Motley Fool layouts publish participants/takeaways/risks as
    # <ul><li> items; older ones use <p>. Capture both.
    elements = body.xpath("./h2|./p|./ul/li")
    sections: dict[str, list[str]] = {}
    current = ""
    full_call_started = False
    full_call: list[str] = []
    for el in elements:
        tag = el.root.tag
        text = (el.xpath("string(.)").get() or "").strip()
        if not text:
            continue
        if tag == "h2":
            current = text.strip()
            if "Full Conference Call" in current:
                full_call_started = True
            sections.setdefault(current, [])
            continue
        if full_call_started:
            full_call.append(text)
        elif current:
            sections.setdefault(current, []).append(text)

    if not full_call:
        return None

    return {
        "title": title or url,
        "url": url,
        "date": " ".join(sections.get("DATE", [])),
        "participants": sections.get("CALL PARTICIPANTS", []),
        "risks": sections.get("RISKS", []),
        "takeaways": sections.get("TAKEAWAYS", []),
        "full_call": full_call,
    }


def _pick_remarks(full_call: list[str], max_paragraphs: int) -> list[str]:
    """Select a bounded excerpt from the full call body.

    Keeps the opening speaker paragraphs (prepared remarks set the stage)
    plus any later paragraph that talks about guidance, outlook, or
    capital allocation. Skips operator lines.
    """
    kept: list[str] = []
    speaker_so_far = 0
    for text in full_call:
        is_operator = text.lower().startswith("operator")
        is_speaker_turn = bool(re.match(r"^[A-Z][\w.'\- ]{2,40}:", text))
        if is_operator:
            continue
        if speaker_so_far < max_paragraphs:
            kept.append(text)
            if is_speaker_turn:
                speaker_so_far += 1
        elif _GUIDANCE_RE.search(text):
            kept.append(text)
        if len(kept) >= max_paragraphs * 2:
            break
    return kept


def _format_transcript(parsed: dict, max_chars: int) -> str:
    lines = [f"## {parsed['title']}", f"URL: {parsed['url']}"]
    if parsed.get("date"):
        lines.append(f"Call date: {parsed['date']}")
    if parsed.get("participants"):
        lines.append("\n**Call participants:**")
        lines.extend(f"- {p}" for p in parsed["participants"][:8])
    if parsed.get("takeaways"):
        lines.append("\n**Key takeaways:**")
        lines.extend(f"- {t}" for t in parsed["takeaways"][:18])
    if parsed.get("risks"):
        lines.append("\n**Risks highlighted:**")
        lines.extend(f"- {t}" for t in parsed["risks"][:6])
    remarks = _pick_remarks(parsed["full_call"], _MAX_REMARK_PARAGRAPHS)
    if remarks:
        lines.append("\n**Prepared-remarks excerpt (guidance-weighted):**")
        for r in remarks:
            lines.append(f"> {r}")
    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[:max_chars].rsplit("\n", 1)[0] + "\n[...truncated...]"
    return text


@tool
@safe_tool
def get_earnings_call_transcripts(
    ticker: Annotated[str, "ticker symbol"],
    quarters: Annotated[int, "number of most recent quarterly calls to return (1-4)"] = 2,
) -> str:
    """Retrieve the most recent quarterly earnings-call transcripts for a
    company, including key takeaways, highlighted risks, call participants,
    and a guidance-weighted excerpt of management's prepared remarks.

    Use this to assess management guidance quality, tone shifts across
    quarters, and execution versus prior promises. Source: The Motley Fool
    public transcript pages (no login required); coverage depends on their
    archive — for obscure or foreign tickers it may return nothing.

    Args:
        ticker (str): Ticker symbol of the company
        quarters (int): Number of most recent quarterly calls to return (default 2, max 4)
    Returns:
        str: Formatted transcripts report with takeaways, risks, and remarks excerpts
    """
    ticker = ticker.upper().strip()
    try:
        quarters = int(quarters)
    except (TypeError, ValueError):
        quarters = 2
    quarters = max(1, min(quarters, 4))

    urls = _search_transcript_urls(ticker, quarters)
    if not urls:
        return (
            f"[get_earnings_call_transcripts] No transcripts found for "
            f"{ticker} on fool.com. Coverage may be unavailable for this ticker."
        )

    blocks: list[str] = [f"# Earnings Call Transcripts for {ticker}"]
    for url in urls:
        try:
            parsed = _parse_transcript(_fetch_page(url), url)
        except Exception as exc:
            logger.warning("Transcript fetch failed for %s: %s", url, exc)
            parsed = None
        if parsed is None:
            blocks.append(f"\n## Transcript could not be parsed\nURL: {url}")
            continue
        blocks.append("\n" + _format_transcript(parsed, _MAX_CHARS_PER_TRANSCRIPT))

    return "\n".join(blocks)
