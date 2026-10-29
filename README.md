# ELK — Quantitative Model Intelligence

*Analyze · Validate · Interpret*

ELK is a browser-based tool for **auditing regression models in investment research**. An analyst loads a
table of data (Excel or CSV), chooses an outcome and explanatory variables, and ELK runs OLS, checks the
assumptions, compares alternative specifications and writes an auditable report. ELK tests whether
quantitative conclusions are supported by the data. **It does not make investment decisions.**

## Workflow

| Step | What happens |
|---|---|
| 1. Data & Suitability | Upload a file or load a demo, review column types and missing values, label units, create dummy variables or excess returns, choose Y / X / date, and get a suitability assessment |
| 2. Regression Lab | OLS estimates, standard errors, t, p, confidence intervals, F-test, R², ANOVA, and a plain-English reading |
| 3. Diagnostics | Breusch–Pagan, Durbin–Watson, Breusch–Godfrey, ARCH LM, Jarque–Bera, RESET, VIF and correlation, Cook's distance and studentised residuals, ADF, plus residual / Q–Q / ACF plots |
| 4. Compare & Repair | Suggested remedies (HC3, HAC/Newey–West, revising a redundant variable, sensitivity excluding flagged rows), a side-by-side A vs B view, nested F / robust Wald tests with guardrails, AIC/BIC, out-of-sample RMSE |
| 5. Insights & Export | A narrative that separates measured facts, warnings and judgment; a scenario tool with extrapolation warnings; an HTML report (print to PDF), an Excel workbook and an audit trail |

## Demo data

- **Demo A — ETF factor returns (public).** ETF prices come from Yahoo Finance (dividend-adjusted closes, via
  `yfinance`). Fama–French 3/5 factors, Momentum and the **risk-free rate** (1-month T-bill) come from the
  [Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
  The frequency can be daily, weekly, monthly or quarterly. ETF excess returns are computed as return − RF for the same period.
- **Demo B — Private credit loans (SYNTHETIC).** 300 fabricated direct-lending loans, generated in
  `core/demo_data.py` with a fixed seed. It does not represent any real borrower, lender or portfolio. It contains
  planted problems: leverage/coverage collinearity, heteroskedasticity, influential distressed loans and a binary
  default flag.

Live sources are tried first. If they can't be reached, ELK uses the snapshot in `data/snapshot/`, which a
scheduled GitHub Action (`.github/workflows/refresh-data.yml`) refreshes every weekday. Every result records
which source was used and when it was retrieved.

## Run it

**Hosted:** deploy this repository on [Streamlit Community Cloud](https://share.streamlit.io) with main file `app.py`.

**On a Windows laptop:** install Python 3.12+ from python.org (tick *Add python.exe to PATH*), then double-click
`run.bat`. On first run it creates a virtual environment, installs the requirements and opens the app in your
browser at `http://localhost:8501`. That address only exists on your own machine.

**Mac/Linux:** `python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt && streamlit run app.py`

## Correctness

`pytest tests/` checks OLS against the NIST StRD *Longley* certified values. It also checks HC3, Durbin–Watson,
Breusch–Pagan, VIF and the nested F-test against independent matrix-algebra calculations, and tests the guardrails
(binary outcomes, perfect collinearity, the dummy trap, too few rows, mismatched samples, chronological holdout).

## Privacy

The public demo runs on shared hosting. **Do not upload confidential data to it.** Before using private fund,
client or company data, run ELK locally or on approved internal infrastructure, and review storage, logging,
authentication and access with your employer.

## Project layout

```
app.py              Streamlit entry point and navigation
core/               statistics, data handling and reporting (no UI code)
ui/                 one module per workflow step
data/snapshot/      fallback copy of public data (auto-refreshed)
scripts/            snapshot refresh script
tests/              reference-value tests
```
