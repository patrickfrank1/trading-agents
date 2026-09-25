# Bayesian Causal Model for Gold, Equities, Bonds, and Real Estate

## Core upgrade: cash-flow shocks vs discount-rate shocks

The model should explicitly distinguish **cash-flow shocks** from **discount-rate shocks**. The same movement in interest rates can have opposite effects depending on why rates moved.

- **Gold:** opportunity cost, real rates, risk/safe-haven demand, liquidity, structural demand.
- **Equities:** expected cash flows + discount rate + equity risk premium.
- **Property:** rents/NOI + financing cost + cap rate + credit availability + physical supply.
- **Bonds:** expected policy path + term premium + inflation/risk compensation.

This prevents a naive rule such as `real yield ↑ → SPX ↓` from misreading a productivity shock that raises both growth and real yields.

---

## 1. Overall architecture

Use a **dynamic structural Bayesian network / state-space model**, not a static regression:

\[
X_t=A X_{t-1}+B U_t+\epsilon_t
\]

where `X` is the endogenous macro-financial state and `U` contains observed or identified shocks.

Asset layer:

`Gold, S&P 500, Treasury bonds, Residential RE, CRE/REITs`

with lagged feedback from asset prices into wealth, credit, and financial conditions.

High-level structure:

```text
Fiscal Policy → Primary Balance → Debt/GDP
       ↓                         ↓
 Fiscal Impulse → Growth → Inflation → Fed Policy
       ↓              ↓             ↓
Financial Conditions ← Credit Conditions ← Treasury Yields
       ↓              ↓                  ↓
   Housing      Corporate Earnings   Real Yields
       ↓              ↓                  ↓
 Wealth/Collateral → SPX             Gold
```

Global growth, risk, liquidity, geopolitics and structural gold demand feed all relevant asset markets.

Arrows are candidate structural relationships; identification requires external shocks/instruments.

---

## 2. Shock taxonomy

### Growth / productivity shock

`Productivity ↑ → Potential Growth ↑ → Expected Earnings ↑ → Rents/NOI ↑`

and potentially:

`Growth ↑ → Neutral Real Rate ↑ → Real Yield ↑`

So equities/property can rise even while real yields rise.

### Monetary / discount-rate shock

`Inflation pressure ↑ → Fed tightening ↑ → Expected policy rate ↑ → Real yield ↑ → Discount rate ↑`

If cash flows do not rise enough, equities and property valuations fall and gold faces higher opportunity cost.

### Risk-premium shock

`Financial stress ↑ → Equity risk premium ↑ + Credit spreads ↑ → SPX/property ↓`

while safe-haven demand can raise gold.

### Funding/liquidity shock

`Funding stress ↑ → Forced liquidation ↑ → SPX/property ↓`

Gold can also fall temporarily because it is liquidated to meet margin/funding needs. Therefore **funding liquidity** should be distinct from ordinary risk aversion.

---

## 3. Fiscal system

### Nodes

- Government spending
- Tax receipts
- Primary balance
- Fiscal impulse
- Debt/GDP
- Interest expense
- Treasury issuance
- Maturity structure
- Fiscal credibility

### Relationships

\[
Debt_t=\frac{1+r_t}{1+g_t}Debt_{t-1}-PrimaryBalance_t+\epsilon_t
\]

`Fiscal policy → aggregate demand → growth → inflation → Fed policy`

`Fiscal policy → debt issuance → term premium → Treasury yields`

`Fiscal credibility → term premium / inflation expectations / USD`

### Inputs

Treasury debt, receipts, outlays, primary balance, interest expense, issuance, maturity, fiscal legislation/announcements, CBO projections.

---

## 4. Growth and productivity

### Nodes

Real GDP growth, potential growth, output gap, labor-force growth, productivity, capital deepening, business investment, global growth.

\[
PotentialGrowth=f(Productivity,LaborForceGrowth,CapitalDeepening)
\]

\[
GDPGrowth=f(PotentialGrowth,OutputGap,FiscalImpulse,GlobalDemand,FinancialConditions)
\]

Growth affects inflation, earnings, rents/NOI, credit risk, tax receipts, debt/GDP and USD.

### Inputs

GDP, industrial production, employment, hours, labor-force participation, productivity, unit labor costs, business investment, global GDP/PMIs, financial conditions.

---

## 5. Inflation

\[
Inflation_t=f(E[\pi_{t+1}],OutputGap,FiscalImpulse,Wages,CommodityShock,ImportPrices)
\]

### Inputs

CPI/PCE, core inflation, inflation expectations, wages, unit labor costs, oil/commodities, import prices, PPI, surveys and market breakevens.

Inflation affects nominal yields, real yields, Fed policy, valuation and gold demand.

---

## 6. Monetary policy and rates

Separate:

1. Expected policy path
2. Real risk-free rate
3. Inflation expectations
4. Term premium
5. Asset-specific spreads

\[
NominalYield=ExpectedFedPath+TermPremium
\]

\[
RealYield=NominalYield-ExpectedInflation
\]

\[
TermPremium=f(DebtSupply,FiscalCredibility,InflationRisk,QE/QT,TreasuryDemand,GlobalSavings)
\]

### Inputs

Fed funds, OIS, Treasury yields, TIPS real yields, breakevens, term-premium estimates, QE/QT, issuance, Treasury holdings.

---

## 7. S&P 500: cash flows + discount rate + risk premium

Do not model simply `Fed Rate → SPX`.

Use:

`Growth → Expected Earnings → Equity Cash Flows`

`Real Yield → Discount Rate → Equity Valuation`

`Equity Risk Premium → Equity Valuation`

A structural return equation is:

\[
r^{SPX}_t=\alpha+\beta_E\Delta ExpectedEarnings_t-\beta_R\Delta RealYield_t-\beta_{ERP}\Delta ERP_t+\beta_L\Delta Liquidity_t+\epsilon_t
\]

### Expected earnings

\[
ExpectedEarnings=f(ExpectedNominalGrowth,Margins,Wages,InterestExpense,USD,Taxes)
\]

### Important channels

- Rates ↑ → corporate interest expense ↑ → earnings ↓.
- USD ↑ can reduce foreign-earnings translation, while also lowering imported input costs.
- Inflation ↑ can raise nominal revenue but also raise wages/input costs and compress margins.

Add **Corporate Pricing Power / Margin Pressure** if needed.

### Equity Risk Premium (latent)

\[
ERP=f(FinancialStress,CreditSpreads,VIX,RiskAppetite,DefaultRisk,Valuation)
\]

### Inputs

Forward EPS, realized earnings, revenue growth, margins, wages, corporate interest expense, taxes, USD; TIPS/nominal yields, OIS and term premium; VIX, credit spreads, defaults, valuation; Fed balance sheet, liquidity and credit growth.

---

## 8. Residential real estate

Do not use one generic real-estate node.

Core structure:

`Treasury yield → Mortgage rate → Affordability → Housing demand → House prices`

with additional channels from income, employment, population, credit, rents and supply.

\[
MortgageRate=ExpectedFedPath+TreasuryTermPremium+MortgageSpread
\]

\[
MortgageSpread=f(CreditRisk,MBSLiquidity,PrepaymentRisk,FundingCosts,FinancialStress)
\]

\[
Affordability=f(HousePrice,MortgageRate,HouseholdIncome)
\]

\[
HousePrice=f(Income,MortgageRate,CreditAvailability,HousingSupply,Population,Rents,Expectations)
\]

### Mortgage lock-in

Higher rates reduce demand but can also reduce existing-home supply:

`Mortgage rate ↑ → lock-in ↑ → existing supply ↓ → prices supported`

Therefore add **Mortgage Lock-In** and use distributed lags.

### Inputs

30-year mortgage rates, Treasury yields, MBS spreads, applications/originations, home sales, house prices, rents, income, employment, permits, starts, completions, inventory, population/household formation, mortgage-rate distribution, LTVs, delinquencies.

---

## 9. Housing supply

\[
HousingSupply=f(Permits,Construction,LandConstraints,ConstructionCosts,Financing)
\]

`Rates ↑ → construction financing cost ↑ → new construction ↓ → future supply ↓`

Inputs: permits, starts, completions, construction employment/costs, land prices, mortgage and construction-loan rates, inventory.

---

## 10. Commercial real estate and REITs

Separate physical CRE from REITs.

\[
CREPrice=f(NOI,CapRate,CreditConditions,Vacancy,Supply)
\]

\[
CapRate=RiskFreeRate+RealEstateRiskPremium
\]

Thus:

`Real yield ↑ → cap rate ↑ → CRE value ↓`

but:

`Growth ↑ → occupancy/rents ↑ → NOI ↑ → CRE value ↑`

REITs combine property cash flows, leverage, equity risk premium and rates, so they need their own asset node.

### Inputs

CRE prices, cap rates, NOI, rents, vacancy, construction pipeline, loan maturities, mortgage rates, delinquencies, REIT prices/dividends/leverage, real yields, credit spreads.

---

## 11. Credit conditions

Add a shared latent/observed state:

\[
CreditConditions=f(BankLending,CorporateSpreads,MortgageSpreads,DefaultRisk,Liquidity)
\]

It affects housing, corporate investment, earnings, consumption, growth and ERP.

Inputs: bank lending standards, loan growth, corporate/mortgage spreads, delinquencies, defaults, loan-officer surveys, funding spreads.

---

## 12. Financial conditions

Add a broader latent state:

\[
FinancialConditions=f(RealYield,USD,CreditSpreads,MortgageRates,SPX,Volatility,BankLending)
\]

Dynamic loop:

`Fed → Financial Conditions → Investment/Consumption/Housing → Growth/Inflation → Fed`

Implement with lags.

---

## 13. Household wealth, collateral and leverage

\[
HouseholdWealth=f(SPX,HousePrices,BondPrices,Income)
\]

`SPX ↑ → wealth ↑ → consumption ↑ → growth ↑`

`House prices ↑ → wealth ↑ → consumption ↑`

`House prices ↑ → collateral ↑ → credit availability ↑`

Add **Effective Household Rate Exposure**:

\[
RateExposure=f(VariableDebt,Refinancing,MortgageMaturity,NewOriginations)
\]

This captures the fact that current rates do not instantly reprice all US household debt.

Inputs: household net worth, equity/house wealth, disposable income, household debt, consumer credit, mortgage debt, fixed/variable share, refinancing and maturity data.

---

## 14. Corporate earnings and investment

\[
Earnings=f(RealGrowth,NominalGrowth,Margins,Wages,InterestExpense,USD,Taxes)
\]

\[
Investment=f(ExpectedDemand,RealRate,CreditConditions,CashFlow,Uncertainty)
\]

Channels:

`Growth → Earnings → SPX`

`Rates → Interest Expense → Earnings → SPX`

`Rates → Investment → Productivity/Growth` (lagged)

Inputs: earnings, margins, wages, corporate debt, interest expense, taxes, capex, investment, surveys and credit conditions.

---

## 15. Corporate credit spreads

\[
CreditSpread=f(DefaultRisk,RiskAppetite,Liquidity,GrowthExpectations)
\]

`Credit spread ↑ → funding cost ↑ → investment ↓ → earnings ↓ → SPX ↓`

and:

`Credit spread ↑ → ERP ↑ → SPX ↓`

This separates risk-free discount-rate shocks from credit-risk-premium shocks.

---

## 16. Banking-system stress

Add a latent **Banking-System Stress** state:

`Property prices ↓ → collateral ↓ → loan losses ↑ → bank capital ↓ → credit supply ↓ → growth ↓`

and potentially:

`Bank stress → funding stress → forced asset sales → temporary gold liquidation`

Inputs: bank capital ratios, NPLs, CRE exposure, mortgage delinquencies, funding spreads, deposits, loan growth, lending standards, bank equity prices, financial-stress indexes.

---

## 17. Gold

Use:

\[
GoldReturn=f(OpportunityCost,InflationRisk,GlobalRisk,Liquidity,USD,CentralBankDemand,AsianDemand,StructuralDemand)
\]

### Opportunity-cost channel

`Real yield ↑ → gold opportunity cost ↑ → gold ↓`

But condition this on the source of the rate movement. Useful interactions:

\[
\Delta RealYield\times FiscalRisk
\]

\[
\Delta RealYield\times GlobalRisk
\]

### Risk and liquidity channels

`Risk aversion ↑ → safe-haven gold demand ↑`

but:

`Funding stress ↑ → forced liquidation ↑ → gold can initially ↓`

Inputs: gold spot/futures, TIPS real yields, USD, inflation expectations, ETF flows, COMEX positioning, central-bank purchases, Asian physical demand, VIX, credit/funding spreads, geopolitical risk, global liquidity.

---

## 18. Treasury bonds

Add Treasury bonds as an explicit asset node because they are the main alternative duration asset and connect rates to the full cross-asset system.

Approximate bond return:

\[
BondReturn\approx-Duration\times\Delta Yield+Carry
\]

Drivers: expected Fed path, term premium, inflation expectations, fiscal supply and risk sentiment.

Inputs: Treasury prices/yields, duration, OIS/Fed path, term premium, breakevens, issuance and global Treasury demand.

---

## 19. USD and external sector

\[
\Delta USD=f(USRealRate-GlobalRealRate,USGrowth-GlobalGrowth,FiscalCredibility,CurrentAccount,SafeHavenDemand)
\]

\[
CA=f(USGrowth,GlobalGrowth,RelativePrices,EnergyBalance,Productivity)
\]

USD affects gold, corporate earnings, imported inflation, commodities and global financial conditions.

Inputs: broad USD/DXY, US/global real-rate and growth differentials, current account, trade balance, capital flows, foreign Treasury holdings, global financial conditions.

---

## 20. Global risk, structural gold demand and asset allocation

### Global Risk / Risk Appetite

Latent state informed by VIX, credit spreads, volatility, geopolitical risk, global PMIs, EM stress, cross-asset correlations and funding spreads.

### Structural Gold Demand

\[
D^{Gold}_t=\rho D^{Gold}_{t-1}+\beta_{CB}CBPurchases_t+\beta_{Asia}AsianDemand_t+\beta_{Fragmentation}Fragmentation_t+\epsilon_t
\]

### Asset Allocation State

Investors allocate across cash, Treasuries, equities, credit, property and gold. A latent relative-attractiveness state can help explain cross-asset flows.

Inputs: fund flows, ETF flows, positioning, relative yields, volatility, valuation, liquidity and survey measures.

---

## 21. Highest-priority latent nodes

### Tier 1

1. Fiscal Credibility
2. Equity Risk Premium
3. Credit Conditions
4. Financial Conditions
5. Global Risk / Risk Appetite
6. Structural Gold Demand
7. Expected Earnings Growth
8. Potential / Productivity Growth

### Tier 2

9. Housing Affordability
10. Mortgage Lock-In
11. Banking-System Stress
12. Funding Liquidity
13. Corporate Pricing Power / Margin Pressure
14. Effective Household Rate Exposure
15. Asset Allocation State

### Tier 3

16. Global monetary fragmentation
17. Long-run neutral real rate
18. Foreign demand for Treasuries
19. Real-estate risk premium
20. Equity valuation regime

Do not add every latent state at once; use staged model development and regularization.

---

## 22. Key cross-asset relationships

### Rates → equities

Separate four channels:

1. **Discount rate:** `Real yield ↑ → valuation ↓`
2. **Cash flow:** `Growth ↑ → earnings ↑ → SPX ↑`
3. **Financing:** `Rates ↑ → interest expense ↑ → earnings ↓`
4. **Risk premium:** `Credit stress ↑ → ERP ↑ → SPX ↓`

### Rates → residential property

1. Demand: `Mortgage rate ↑ → affordability ↓ → demand ↓`
2. Supply: `Mortgage/construction rate ↑ → construction ↓ → future supply ↓`
3. Lock-in: `Mortgage rate ↑ → lock-in ↑ → existing supply ↓`
4. Valuation: required return ↑ → property value ↓

### Rates → CRE

`Real yield ↑ → cap rate ↑ → CRE value ↓`

while `Growth ↑ → NOI ↑ → CRE value ↑`.

### Rates → gold

`Real yield ↑ → opportunity cost ↑ → gold ↓`, conditional on the shock source.

### SPX/property → macro

Use lags:

`SPX ↑ → wealth ↑ → consumption ↑ → growth ↑`

`House prices ↑ → wealth/collateral ↑ → consumption/credit ↑ → growth ↑`

### Property → banking → macro

`Property ↓ → collateral ↓ → bank losses ↑ → credit supply ↓ → growth ↓`

### SPX ↔ gold

Do not use a fixed direct coefficient. Route through **Risk Appetite** and **Funding Liquidity**:

`SPX ↓ → risk aversion ↑ → gold demand ↑`

versus during funding crises:

`SPX ↓ → margin calls ↑ → funding stress ↑ → forced gold sales ↑ → gold ↓`

---

## 23. Bayesian parameterization

Use economically informed priors rather than hard-coded coefficients.

Examples:

\[
\beta_{RealYield,Gold}\sim Normal(\mu,\sigma)
\]

with sign-informed priors where justified.

For equities:

\[
\beta_{CashFlow}>0,\quad\beta_{DiscountRate}<0,\quad\beta_{ERP}<0
\]

For residential property:

\[
\beta_{Income}>0,\quad\beta_{MortgageRate}<0,\quad\beta_{Supply}<0
\]

For CRE:

\[
\beta_{NOI}>0,\quad\beta_{CapRate}<0
\]

For gold:

\[
\beta_{OpportunityCost}<0
\]

but allow interactions with fiscal risk, inflation risk and global risk.

Use hierarchical priors so coefficients can vary across regimes while remaining partially pooled.

---

## 24. Regime switching

\[
S_t\in\{Normal,Inflationary,Deflationary,ProductivityBoom,FiscalStress,FinancialCrisis\}
\]

with Markov transitions and regime-specific coefficients.

This is particularly important for:

- real yield → gold
- rates → equities
- rates → housing
- SPX → gold
- credit → property

---

## 25. Identification strategy

A causal graph does not establish causality. Use external/quasi-exogenous shocks where possible:

- FOMC monetary-policy surprises
- CPI/inflation surprises
- Treasury auction surprises
- fiscal announcement surprises
- legislative fiscal shocks
- productivity shocks
- oil/commodity supply shocks
- geopolitical events
- bank stress events
- housing-policy changes

Distinguish:

1. Predictive association
2. Structural relationship
3. Identified causal effect

Estimate impulse responses for identified shocks.

---

## 26. Recommended model versions

### V1 — Gold baseline

Gold returns on TIPS real yield, USD, inflation expectations, global risk, liquidity, central-bank demand, Asian demand and lagged gold returns. Use Student-t errors.

### V2 — Cross-asset layer

Add SPX, Treasury bonds, residential RE and CRE/REITs.

### V3 — Cash-flow / discount-rate decomposition

Add expected earnings, ERP, mortgage rates, affordability, cap rates, NOI, corporate interest expense and credit spreads.

### V4 — Latent macro-financial states

Add fiscal credibility, financial conditions, credit conditions, risk appetite, funding liquidity, structural gold demand and productivity/potential growth.

### V5 — Regime switching

Add inflationary, deflationary, productivity-boom, fiscal-stress, financial-crisis and normal regimes.

### V6 — Identified causal shocks

Add monetary, inflation, fiscal, productivity, risk-premium and funding shocks; estimate causal impulse responses.

---

## 27. Data architecture

Prefer monthly frequency for the main model; use higher-frequency data for identification where useful.

| Node / block | Main input data |
|---|---|
| Fiscal | Debt, receipts, spending, primary balance, interest expense, issuance, maturity |
| Growth | GDP, industrial production, employment, productivity, investment, global PMIs |
| Inflation | CPI/PCE, wages, unit labor costs, breakevens, surveys, commodities |
| Rates | Fed funds, OIS, Treasuries, TIPS, term premium, QE/QT |
| Equities | SPX, forward EPS, earnings, margins, VIX, valuations |
| Housing | House prices, rents, mortgage rates, applications, sales, inventory, permits, starts, completions |
| Mortgage structure | Fixed/variable share, mortgage-rate distribution, refinancing, maturity, LTVs |
| CRE | Prices, rents, NOI, cap rates, vacancy, REITs, maturities, delinquencies |
| Credit | Corporate/mortgage spreads, lending standards, defaults, delinquencies |
| Banking | Capital ratios, NPLs, CRE exposure, deposits, bank equities, funding spreads |
| Financial conditions | FCI, funding spreads, volatility, bank lending, liquidity |
| External | USD, current account, trade, capital flows, foreign Treasury holdings, global growth/rates |
| Gold | Spot/futures, ETF flows, COMEX positioning, central-bank purchases, Asian demand, geopolitical risk |

---

## 28. Validation

Use rolling or expanding out-of-sample validation.

Evaluate:

- RMSE / MAE
- log predictive density
- directional accuracy
- interval coverage and calibration
- crisis performance
- regime-specific performance
- causal impulse responses
- parameter stability
- sign consistency

Compare against:

1. Random walk
2. AR model
3. Gold ~ USD + real yield
4. Bayesian VAR
5. Structural Bayesian model

The structural model should not be judged only on point-forecast accuracy; scenario consistency and causal impulse responses are central outputs.

---

## 29. Scenario / intervention analysis

### Fiscal consolidation

`do(Primary Balance +2% GDP)`

Propagate:

`Fiscal consolidation → debt → term premium → inflation → Fed → real yield → USD → earnings/housing/gold`

### Productivity boom

`Productivity ↑ → potential growth ↑ → earnings ↑ → rents/NOI ↑ → real yield ↑ → SPX/property ↑ → gold opportunity cost ↑`

### Inflation shock

`Inflation ↑ → Fed ↑ → discount rate ↑ → SPX/property pressure`

Gold depends on the balance between inflation-risk demand and real-rate opportunity cost.

### Financial crisis

`Credit stress ↑ → ERP ↑ → credit spreads ↑ → SPX/property ↓ → funding stress ↑`

Gold safe-haven demand can rise, but forced liquidation can make gold fall initially.

---

## 30. Final conceptual rule

The mature model should not ask:

> **What happens to gold when interest rates rise?**

It should ask:

> **What shock caused rates to rise, what happens to cash flows, what happens to discount rates and risk premia, how do credit and financial conditions respond, and how do those channels propagate across bonds, equities, property, the USD and gold?**

The final system is a **joint Bayesian macro-financial causal model**:

\[
\boxed{Macro\ Shocks\rightarrow CashFlows,\ DiscountRates,\ RiskPremia,\ Credit,\ Liquidity\rightarrow Bonds,\ Equities,\ Property,\ USD,\ Gold\rightarrow Wealth,\ Credit,\ FinancialConditions\rightarrow Macro\ Economy}
\]

This architecture is designed to explain second- and third-order effects and to support counterfactual questions such as whether debt can fall while gold falls, whether rates can rise while equities rise, whether housing can remain strong despite higher rates, and when equities and gold can fall together.
