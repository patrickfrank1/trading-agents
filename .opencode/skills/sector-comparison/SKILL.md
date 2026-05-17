---
name: sector-comparison
description: Use when the user asks to compare multiple companies in a sector, compile a fundamentals comparison table, rank them by investment quality, or generate a sector comparison report from TradingAgents reports in reports/sent/. Trigger keywords and phrases: "compare [tickers/sector]", "comparison report", "rank these companies", "sector comparison", "which is the best investment", "compare fundamentals", "side-by-side comparison". Use ONLY when the user provides 3+ tickers/companies AND asks to compare, rank, or generate a report about them. Do NOT use for single-company analysis or when the user only wants to read a report.
---

# Sector Company Comparison Report Generator

Generates a comprehensive side-by-side comparison of companies within a sector from TradingAgents analysis reports stored in `reports/sent/`. Automatically adapts sector-specific analysis dimensions (drug pipelines, tech products, bank loan books, REIT properties, insurance underwriting, etc.).

## Workflow

### Step 1: Locate Reports

Reports are stored under `reports/sent/` in timestamped directories: `TICKER_YYYYMMDD_HHMMSS/`. Each contains a `complete_report.md`.

- Run `ls reports/sent/` to list all available report directories
- Match each user-provided ticker to its directory (e.g. `PFE` → `PFE_20260505_210459/`, `AAPL` → `AAPL_20260505_123456/`)
- If multiple directories exist for a ticker, use the one with the **highest timestamp** (most recent)
- If a ticker has no match, try case-insensitive matching; if still not found, exclude it and note this to the user

### Step 2: Determine the Sector

Before extracting data, identify the sector these companies belong to. The sector determines what business-specific assets to extract:

| Sector | Business-Specific Assets to Extract |
|--------|-------------------------------------|
| **Pharma/Biotech** | Drug pipeline (candidates, phases, indications, PDUFA dates), patent cliffs, key franchises |
| **Technology/Software** | Product portfolio, platform ecosystem, cloud/ARR metrics, R&D pipeline, competitive moats (network effects, switching costs), AI capabilities |
| **Banking/Financials** | Loan book composition, deposit base, net interest margin, CET1 capital ratio, credit quality (NPL ratio), fee income streams |
| **Insurance** | Underwriting profitability (combined ratio), float, policyholder base, investment portfolio yield, catastrophe exposure |
| **REITs/Real Estate** | Property portfolio (sectors, locations, occupancy rates), lease duration/WALT, acquisition pipeline, development projects, NAV per share |
| **Energy/Oil & Gas** | Reserves/production, acreage, exploration pipeline, production costs/breakeven, refining capacity, renewable transition assets |
| **Consumer/Retail** | Brand portfolio, store footprint/e-commerce mix, private label penetration, supply chain infrastructure, customer loyalty metrics |
| **Industrials** | Manufacturing footprint, order backlog, aftermarket/service revenue, IP/patent portfolio, infrastructure assets |
| **Telecom/Media** | Spectrum licenses, subscriber base, network infrastructure, content IP/library, churn rates |

If the sector is ambiguous or mixed, read the first ~100 lines of ONE report to identify the sector from the Fundamentals Analyst section, then proceed.

### Step 3: Extract Data in Parallel

For **each** company, launch a **task subagent of type `general`** to read the full `complete_report.md`. Launch **ALL agents simultaneously** in one message for maximum parallelism.

Each agent must extract and return structured data covering:

1. **Fundamentals** (universal across all sectors):
   - Market cap, current price, 52-week range, beta
   - Forward P/E, trailing P/E, EV/EBITDA, P/B, P/S, PEG
   - Revenue (TTM), revenue growth YoY, revenue growth trajectory (multi-year)
   - Gross margin, operating margin, net margin, EBITDA margin
   - Net income (TTM), diluted EPS, ROE, ROA
   - Free cash flow (annual), FCF yield, operating cash flow, FCF margin
   - Total debt, net debt, debt/equity, net debt/EBITDA
   - Cash & equivalents, current ratio, quick ratio, working capital
   - Total assets, stockholders' equity, tangible book value, goodwill & intangibles
   - Dividend yield, annual dividend, payout ratio (earnings and FCF), buybacks
   - R&D or CapEx spend (whichever is material to the sector)
   - DCF base case fair value, DCF pessimistic/optimistic, EPV, comps fair value, analyst consensus target
   - Interest coverage ratio

2. **Business-Specific Assets & Competitive Position** (sector-adapted):
   - Identify the sector using the table above, then extract ALL relevant items
   - For each key asset/product/franchise: name, stage/maturity, revenue contribution (%), competitive differentiator, upcoming catalysts
   - The single biggest "cliff" or structural threat (patent cliff, platform shift, regulatory change, commodity cycle, etc.)
   - Moat assessment mentioned by the Business Analyst and Fundamentals Analyst

3. **Growth Estimates**:
   - Historical revenue growth (trailing 3-5 years if available)
   - Forward revenue growth guidance, EPS growth implied by forward P/E gap
   - Segment-level growth dynamics (which segments growing, which declining)
   - DCF revenue growth assumptions (base, pessimistic, optimistic)
   - Bull vs bear growth probability assessments from the Business Analyst

4. **Portfolio Manager Decision**:
   - Final rating exactly as stated: BUY / OVERWEIGHT / HOLD / UNDERWEIGHT / SELL
   - Price target (explicit if given, otherwise the DCF base case or analyst consensus referenced)
   - Execution instructions (entry zone, stop-loss levels, position sizing guidance)
   - Conditional upgrade/downgrade triggers mentioned by the PM

5. **Bull & Bear Case Summaries**:
   - 2-3 sentence distillation of each side, preserving the key quantitative anchors (multiples, growth rates, fair values) from the original report

**Subagent instructions**: Be thorough. Read every section of the report (Analysts, Research, Trading, Risk, **Portfolio Manager**). The Portfolio Manager section is critical — it contains the final rating, entry/exit instructions, and the PM's holistic understanding of the business model, competitive position, and investment thesis. Read it in full to understand the company's business model and risk/reward profile, not just to extract the rating. Return raw data — do not summarize or editorialize. If a metric is mentioned in multiple places with different values, report all values and note the discrepancy.

### Step 4: Compile the Report

Create a markdown file at `reports/sector_comparison_<SECTOR>_<YYYYMMDD>.md` (e.g. `reports/sector_comparison_pharma_20260514.md`). If the sector span is ambiguous, use `cross_sector` as the sector name.

#### Report Structure

**Title:**
```markdown
# [Sector] Sector Comparison Report
**Generated:** YYYY-MM-DD | **Source:** TradingAgents multi-agent analysis reports
**Companies:** N | **Tickers:** TICKER1, TICKER2, ...
```

#### Section 1: Fundamentals Comparison Table
Columns (all values in USD):
| Ticker | Company | Price ($) | Mkt Cap ($B) | Fwd P/E | EV/EBITDA | FCF ($B) | FCF Yield | Net Debt ($B) | D/E | Cur. Ratio | Div Yield | Rev TTM ($B) | Rev Growth | Op Margin | ROE |

Add a footnote noting the FX conversion rate if any non-USD tickers.

#### Section 2: Business Assets & Competitive Position (sector-adapted title)
- **For Pharma**: "Pipeline & Key Drug Comparison"
- **For Tech**: "Product Portfolio & Platform Comparison"
- **For Banks**: "Balance Sheet & Lending Franchise Comparison"
- **For REITs**: "Property Portfolio Comparison"
- **For Insurance**: "Underwriting & Float Comparison"
- **Default**: "Business Assets & Competitive Position"

Columns:
| Ticker | Key Assets/Franchises | Development Pipeline / Late-Stage Assets | Biggest Structural Risk |

#### Section 3: Portfolio Manager Ratings Summary
| Ticker | PM Rating | Fair Value / Target | Key Rationale |

#### Section 4: Ranking — Best to Worst Investment

Rank all companies from 1 to N. Use these weighted criteria:

1. **PM final rating** (primary tiebreaker): BUY > OVERWEIGHT > HOLD > UNDERWEIGHT > SELL
2. **Valuation attractiveness**: lower forward P/E, higher FCF yield, lower EV/EBITDA, discount to DCF/EPV
3. **Financial health**: lower D/E, higher current ratio, net cash preferred, positive tangible book, comfortable interest coverage
4. **Competitive strength**: moat durability, ability to withstand sector-specific structural threats, asset/platform quality
5. **Growth trajectory**: revenue growth rate, earnings momentum, margin expansion trend

**Tier structure:**
- **Tier 1**: BUY + strong OVERWEIGHT with excellent valuation and financials
- **Tier 2**: OVERWEIGHT with catalysts but manageable risk
- **Tier 3**: HOLD (quality companies, wait for better entry or catalyst)
- **Tier 4**: HOLD with significant structural risks
- **Tier 5**: UNDERWEIGHT / SELL

For each ranked company, write:
- **Bold one-line thesis** (e.g. "The best risk/reward in the sector")
- 3-4 sentence synthesis of why it's ranked here, weaving together valuation, financials, competitive position, and growth
- Single biggest risk and single biggest catalyst

#### Section 5: Summary Ranking Table
| Rank | Ticker | Company | Rating | Fwd P/E | FCF Yield | Rev Growth | Net Debt/EBITDA |

#### Section 6: Conclusion
- **Top Picks** (1 paragraph): the strongest 1-3 names and why
- **Best Value Plays**: companies trading below intrinsic value with catalysts
- **Avoid or Reduce**: companies with structural/existential risks
- **Disclaimer**: "*All data sourced from TradingAgents multi-agent analysis reports. This is not financial advice. Reports reflect analysis as of their generation dates, which may differ from the current market.*"

### Step 5: Quality Checks

Before finalizing, verify:
- ALL user-requested tickers appear in the report (note any missing ones)
- Every number traces to a specific report — **never fabricate data**. Mark missing metrics as `—`
- Rankings are internally consistent: a lower-ranked company should not have strictly better metrics than a higher-ranked one. If metrics conflict with PM rating, explain the tension in the analysis
- Currency conversions are documented: note the EUR→USD and CHF→USD rates used
- The filename follows the convention `reports/sector_comparison_<SECTOR>_<YYYYMMDD>.md`

### Important Rules

- **All data comes exclusively from `reports/sent/` files** — never fetch external data, never guess
- **Use task subagents for parallelism** — launch all company extractions simultaneously
- **Be sector-aware not sector-blind** — the asset comparison table must reflect what actually matters for that industry
- **Always read the Portfolio Manager report in full** — the PM section is where the business model, competitive position, and investment thesis are synthesized. Do not extract just the rating; understand the business.
- **Always rank the companies** — every comparison report MUST include a ranked ordering from best to worst investment (Section 4: Ranking). Rankings should synthesize PM ratings, valuation, financial health, competitive strength, and growth trajectory.
- **Convert all currencies to USD** for comparison tables, noting the rate used
- **Preserve file naming**: `reports/sector_comparison_<sector>_<YYYYMMDD>.md`
