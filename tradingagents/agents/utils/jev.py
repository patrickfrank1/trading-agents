"""Jev decision tool for the Portfolio Manager.

Jev is TypeSafe AI's hosted "System One" model. Unlike an LLM it does not
generate prose: you hand it a ``state`` plus a set of typed questions and it
returns constrained answers (``Choice`` / ``Score`` / ``Noul``) with
calibrated probabilities. The Portfolio Manager calls it as a mandatory step,
passing the canonical facts snapshot plus the analyst reports, and asks for:

- an intrinsic-value estimate,
- a low/high confidence interval around it,
- a target portfolio allocation, and
- a 5-tier rating.

Jev is explicitly *not* a calculator, so intrinsic value and its bounds are
expressed as a ``Choice`` over price bands derived in code from the current
price; the chosen bands are mapped back to numbers here. Jev's rating is
authoritative and drives the final ``PortfolioDecision`` rating.

Every failure mode (missing SDK, missing API key, network error, malformed
answer) degrades gracefully to the existing LLM-only decision path by
returning ``None``.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from tradingagents.agents.schemas import PortfolioRating

logger = logging.getLogger(__name__)

try:  # pragma: no cover - exercised via the availability flag in tests
    from typesafe_sdk import TypeSafeClient

    _TYPESAFE_AVAILABLE = True
except Exception as _exc:  # noqa: BLE001 - any import failure should degrade
    TypeSafeClient = None  # type: ignore[assignment]
    _TYPESAFE_AVAILABLE = False
    _SDK_IMPORT_ERROR: Optional[Exception] = _exc
else:
    _SDK_IMPORT_ERROR = None


# ---------------------------------------------------------------------------
# Question definitions
# ---------------------------------------------------------------------------

# Multiplicative ladder used to express intrinsic value relative to the
# current price. Jev picks a band; we turn the band back into a number.
_LADDER_STEPS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)

# Ordered (label, midpoint factor) allocation bands.
_ALLOCATION_BANDS: tuple[tuple[str, float], ...] = (
    ("0% of NAV (exit or avoid)", 0.0),
    ("1-2% of NAV", 1.5),
    ("2-5% of NAV", 3.5),
    ("5-10% of NAV", 7.5),
    ("10-15% of NAV", 12.5),
    ("more than 15% of NAV", 15.0),
)

_RATING_DESCRIPTIONS = {
    PortfolioRating.BUY.value: "Strong conviction to enter or add to the position.",
    PortfolioRating.OVERWEIGHT.value: "Favourable outlook; gradually increase exposure.",
    PortfolioRating.HOLD.value: "Maintain the current position; no action needed.",
    PortfolioRating.UNDERWEIGHT.value: "Reduce exposure; take partial profits.",
    PortfolioRating.SELL.value: "Exit the position or avoid entry.",
}


@dataclass
class JevAssessment:
    """Structured result of one Jev decision call."""

    rating: PortfolioRating
    allocation_pct: float
    intrinsic_value: Optional[float] = None
    intrinsic_value_low: Optional[float] = None
    intrinsic_value_high: Optional[float] = None
    confidence: Optional[float] = None
    model: str = ""
    current_price: Optional[float] = None
    answers: dict[str, Any] = field(default_factory=dict)


def _price_bands() -> tuple[tuple[str, float, float], ...]:
    """Return ordered ``(label, low_factor, high_factor)`` intrinsic-value bands."""
    bands: list[tuple[str, float, float]] = [
        ("below 0.50x of current price", 0.40, 0.50)
    ]
    for low, high in zip(_LADDER_STEPS, _LADDER_STEPS[1:]):
        bands.append((f"{low:.2f}x-{high:.2f}x of current price", low, high))
    bands.append(("above 1.50x of current price", 1.50, 1.60))
    return tuple(bands)


_PRICE_BANDS = _price_bands()
_PRICE_BAND_FACTOR = {
    label: (low + high) / 2.0 for label, low, high in _PRICE_BANDS
}
_PRICE_BAND_CRITERIA = {
    label: f"Intrinsic value is {label}." for label, _, _ in _PRICE_BANDS
}
_ALLOCATION_VALUE = dict(_ALLOCATION_BANDS)
_ALLOCATION_CRITERIA = {
    label: f"Target position size {label}."
    for label, _ in _ALLOCATION_BANDS
}


def extract_current_price(state: dict) -> Optional[float]:
    """Best-effort parse of the current share price from the collected reports.

    Prefers an explicit "current price" mention, then other price phrasing,
    then a bare dollar amount. Returns ``None`` when no plausible price is
    found, in which case the numeric intrinsic-value questions are skipped.
    """
    if not state:
        return None
    text = " ".join(
        str(state.get(key, "") or "")
        for key in (
            "facts_snapshot",
            "fundamentals_report",
            "business_report",
            "market_report",
        )
    )
    number = r"([0-9][0-9,]*(?:\.[0-9]+)?)"
    patterns = (
        rf"current price[^0-9$]{{0,40}}\$?\s*{number}",
        rf"(?:trading at|trades at|last close|share price|price of)[^0-9$]{{0,30}}\$?\s*{number}",
        rf"\$\s*{number}",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue
        try:
            value = float(match.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value
    return None


def _build_state_payload(state: dict, max_state_chars: int) -> dict:
    """Assemble the state handed to Jev: facts in full, reports capped."""
    payload: dict[str, Any] = {
        "ticker": state.get("company_of_interest", ""),
        "trade_date": state.get("trade_date", ""),
        "canonical_facts": state.get("facts_snapshot", ""),
    }
    reports = []
    for key in (
        "business_report",
        "fundamentals_report",
        "sector_report",
        "macro_report",
        "news_report",
        "market_report",
    ):
        text = state.get(key, "")
        if text:
            reports.append(f"--- {key} ---\n{text}")
    joined = "\n\n".join(reports)
    if max_state_chars and len(joined) > max_state_chars:
        joined = joined[:max_state_chars] + "\n...[truncated]"
    if joined:
        payload["analyst_reports"] = joined
    return payload


def _build_questions(value_questions: bool) -> dict[str, dict]:
    questions: dict[str, dict] = {
        "rating": {
            "type": "choice",
            "instructions": (
                "Based only on the collected facts and analyst reports in the "
                "state, what is the correct portfolio rating for this "
                "instrument over a 12-month horizon? Reserve Hold for genuinely "
                "balanced evidence."
            ),
            "criteria": _RATING_DESCRIPTIONS,
        },
        "allocation": {
            "type": "choice",
            "instructions": (
                "What target portfolio allocation (as a share of NAV) does this "
                "instrument warrant, consistent with the rating? Use 0% for a "
                "Sell/avoid decision."
            ),
            "criteria": _ALLOCATION_CRITERIA,
        },
    }
    if value_questions:
        questions["intrinsic_value"] = {
            "type": "choice",
            "instructions": (
                "Which band best estimates the company's intrinsic value per "
                "share? Derive it from fundamentals in the state "
                "(earnings trajectory, margins, justified multiples, cash flows) "
                "— do not anchor on sell-side price targets or 52-week extremes."
            ),
            "criteria": _PRICE_BAND_CRITERIA,
        }
        questions["intrinsic_value_low"] = {
            "type": "choice",
            "instructions": (
                "Which band is the conservative (bear-case) lower bound of a "
                "plausible intrinsic-value range for this company?"
            ),
            "criteria": _PRICE_BAND_CRITERIA,
        }
        questions["intrinsic_value_high"] = {
            "type": "choice",
            "instructions": (
                "Which band is the optimistic (bull-case) upper bound of a "
                "plausible intrinsic-value range for this company?"
            ),
            "criteria": _PRICE_BAND_CRITERIA,
        }
    return questions


def _choice_value(answer: Any) -> Optional[str]:
    if answer is None:
        return None
    value = getattr(answer, "choice", None)
    if value is None and isinstance(answer, dict):
        value = answer.get("choice")
    return value


def _confidence_value(answer: Any) -> Optional[float]:
    if answer is None:
        return None
    value = getattr(answer, "confidence", None)
    if value is None and isinstance(answer, dict):
        value = answer.get("confidence")
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _answer_for(response: Any, name: str) -> Any:
    answers = getattr(response, "answers", None)
    if answers is None and isinstance(response, dict):
        answers = response.get("answers")
    if not answers:
        return None
    return answers.get(name)


def _factor_to_number(factor: Optional[float], price: Optional[float]) -> Optional[float]:
    if factor is None or price is None:
        return None
    return round(price * factor, 2)


def _coerce_answers(response: Any) -> tuple[Optional[PortfolioRating], dict]:
    """Parse a response into a rating plus raw answer map; None if unusable."""
    rating_raw = _choice_value(_answer_for(response, "rating"))
    if rating_raw is None:
        return None, {}
    try:
        rating = PortfolioRating(rating_raw)
    except ValueError:
        logger.warning("Jev returned an unrecognised rating %r; ignoring assessment", rating_raw)
        return None, {}
    return rating, {
        "intrinsic_value": _answer_for(response, "intrinsic_value"),
        "intrinsic_value_low": _answer_for(response, "intrinsic_value_low"),
        "intrinsic_value_high": _answer_for(response, "intrinsic_value_high"),
        "allocation": _answer_for(response, "allocation"),
    }


def _build_assessment(
    response: Any,
    price: Optional[float],
    model: str,
) -> Optional[JevAssessment]:
    rating, answers = _coerce_answers(response)
    if rating is None:
        return None

    allocation_label = _choice_value(answers["allocation"])
    allocation = _ALLOCATION_VALUE.get(allocation_label, 0.0)
    # Keep an exit rating from carrying a positive allocation.
    if rating == PortfolioRating.SELL:
        allocation = 0.0
    elif rating == PortfolioRating.UNDERWEIGHT:
        allocation = min(allocation, 2.5)

    value_factor = _PRICE_BAND_FACTOR.get(
        _choice_value(answers["intrinsic_value"]) or "", None
    )
    low_factor = _PRICE_BAND_FACTOR.get(
        _choice_value(answers["intrinsic_value_low"]) or "", None
    )
    high_factor = _PRICE_BAND_FACTOR.get(
        _choice_value(answers["intrinsic_value_high"]) or "", None
    )

    # Force the interval to contain the point estimate even if Jev's three
    # independent band answers are mutually inconsistent.
    if value_factor is not None:
        low_factor = value_factor if low_factor is None else min(low_factor, value_factor)
        high_factor = value_factor if high_factor is None else max(high_factor, value_factor)

    raw_answers = {
        key: _choice_value(answer)
        for key, answer in answers.items()
        if key != "allocation"
    }
    raw_answers["allocation"] = allocation_label

    return JevAssessment(
        rating=rating,
        allocation_pct=allocation,
        intrinsic_value=_factor_to_number(value_factor, price),
        intrinsic_value_low=_factor_to_number(low_factor, price),
        intrinsic_value_high=_factor_to_number(high_factor, price),
        confidence=_confidence_value(_answer_for(response, "rating")),
        model=str(getattr(response, "model", "") or model),
        current_price=price,
        answers=raw_answers,
    )


def assess_with_jev(
    state: dict,
    *,
    model: str = "jev-latest",
    current_price: Optional[float] = None,
    max_state_chars: int = 24000,
    client: Any = None,
) -> Optional[JevAssessment]:
    """Query Jev for an intrinsic-value / allocation / rating assessment.

    Returns ``None`` (never raises) when Jev is unavailable or misbehaves, so
    the Portfolio Manager can fall back to its LLM-only path. Pass ``client``
    to inject a fake in tests.
    """
    if client is None:
        if not _TYPESAFE_AVAILABLE:
            logger.debug("typesafe-sdk unavailable; skipping Jev decision tool (%s)", _SDK_IMPORT_ERROR)
            return None
        if not os.environ.get("TYPESAFE_API_KEY", "").strip():
            logger.debug("TYPESAFE_API_KEY not set; skipping Jev decision tool")
            return None

    if current_price is None:
        current_price = extract_current_price(state)
    payload = _build_state_payload(state, max_state_chars)
    if current_price is not None:
        payload["current_price"] = current_price
    questions = _build_questions(value_questions=current_price is not None)

    try:
        if client is not None:
            response = client.system_one(state=payload, questions=questions)
        else:
            with TypeSafeClient(model=model) as sdk_client:  # type: ignore[misc]
                response = sdk_client.system_one(state=payload, questions=questions)
    except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully
        logger.warning("Jev decision tool failed (%s); falling back to LLM-only PM decision", exc)
        return None

    try:
        assessment = _build_assessment(response, current_price, model)
    except Exception as exc:  # noqa: BLE001 - malformed answer must not break the run
        logger.warning("Jev response could not be interpreted (%s); ignoring assessment", exc)
        return None
    if assessment is None:
        logger.warning("Jev did not return a usable rating; ignoring assessment")
    return assessment


def render_jev_assessment(assessment: JevAssessment) -> str:
    """Render a JevAssessment to markdown for the PM decision and reports."""
    confidence = (
        f" (confidence {assessment.confidence:.0%})"
        if assessment.confidence is not None
        else ""
    )
    lines = [
        "**Jev Decision Tool (TypeSafe System One)**",
        f"- Rating: **{assessment.rating.value}**{confidence}",
        f"- Target allocation: **{assessment.allocation_pct:g}% of NAV**",
    ]
    if assessment.intrinsic_value is not None:
        lines.append(
            "- Intrinsic value: "
            f"**{assessment.intrinsic_value:,.2f}** per share "
            f"(low {assessment.intrinsic_value_low:,.2f}, "
            f"high {assessment.intrinsic_value_high:,.2f})"
        )
    else:
        lines.append(
            "- Intrinsic value: unavailable (current price could not be parsed "
            "from the reports)"
        )
    if assessment.model:
        lines.append(f"- Model: {assessment.model}")
    return "\n".join(lines)
