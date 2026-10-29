"""Demo B: a SYNTHETIC private-credit (direct lending) loan book.

Every value is fabricated by a random generator with a fixed seed. No real borrower, lender or
portfolio is represented. Problems are planted on purpose so ELK's diagnostics have something to find:
  1. leverage and interest coverage are mechanically linked → high VIF
  2. spread noise is larger for smaller borrowers          → heteroskedasticity (Breusch–Pagan)
  3. a handful of distressed loans with outsized spreads    → influential observations (Cook's distance)
  4. a binary default flag                                 → ELK explains why OLS is not appropriate for it
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DEMO_B_DICTIONARY = {
    "loan_id": "Synthetic loan identifier",
    "origination_year": "Year the loan was originated",
    "sector": "Borrower sector",
    "seniority": "Position in the capital structure (First Lien / Unitranche / Second Lien)",
    "sponsor_backed": "1 if the borrower is owned by a private-equity sponsor, else 0",
    "ebitda_musd": "Borrower last-twelve-month EBITDA, USD millions",
    "log_ebitda": "Natural log of EBITDA (USD millions) — a size measure",
    "leverage_x": "Total debt / EBITDA (turns)",
    "interest_coverage_x": "EBITDA / cash interest expense (turns)",
    "ltv_pct": "Loan-to-value, percent of enterprise value",
    "spread_bps": "Contractual spread over the base rate, basis points (outcome)",
    "default_flag": "1 if the loan defaulted within 24 months, else 0 (binary — not suitable for OLS)",
}


def make_private_credit(n: int = 300, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    year = rng.choice(np.arange(2019, 2025), size=n, p=[0.12, 0.13, 0.19, 0.2, 0.18, 0.18])
    sector = rng.choice(["Software", "Healthcare", "Industrials", "Business Services", "Consumer"], size=n,
                        p=[0.26, 0.2, 0.2, 0.2, 0.14])
    seniority = rng.choice(["First Lien", "Unitranche", "Second Lien"], size=n, p=[0.5, 0.35, 0.15])
    sponsor = (rng.random(n) < 0.72).astype(int)

    ebitda = np.exp(rng.normal(np.log(28), 0.85, n)).clip(3, 400)
    log_e = np.log(ebitda)
    lev = (4.6 + 0.35 * (seniority == "Unitranche") + 1.1 * (seniority == "Second Lien")
           + 0.25 * sponsor + 0.18 * (log_e - log_e.mean()) + 0.35 * (sector == "Software")
           + rng.normal(0, 0.8, n)).clip(1.8, 8.5)
    # all-in coupon ~ base + spread; coverage mechanically ≈ 1 / (leverage × cost of debt)
    cost = 0.095 + rng.normal(0, 0.0025, n)
    coverage = (1.0 / (lev * cost)) * np.exp(rng.normal(0, 0.02, n))
    # planted distressed / mispriced loans (influential points): small, very levered borrowers
    idx = rng.choice(np.where((ebitda < 15))[0], size=5, replace=False)
    lev[idx] = rng.uniform(7.3, 8.4, 5)
    coverage[idx] = 1.0 / (lev[idx] * 0.105)
    ltv = (22 + 6.2 * lev + rng.normal(0, 5.5, n)).clip(15, 85)

    year_fx = {2019: 0, 2020: 40, 2021: -25, 2022: 35, 2023: 70, 2024: 10}
    sector_fx = {"Software": -15, "Healthcare": 0, "Industrials": 15, "Business Services": 5, "Consumer": 35}
    sen_fx = {"First Lien": 0, "Unitranche": 70, "Second Lien": 290}
    noise_sd = 150 / np.sqrt(ebitda / 6)  # small borrowers → noisier pricing (planted heteroskedasticity)
    spread = (520 + 42 * (lev - 4.5) - 18 * (coverage - 2.3) - 38 * (log_e - np.log(28)) + 0.9 * (ltv - 50)
              - 35 * sponsor + np.vectorize(sen_fx.get)(seniority) + np.vectorize(year_fx.get)(year)
              + np.vectorize(sector_fx.get)(sector) + rng.normal(0, 1, n) * noise_sd)

    spread[idx] = spread[idx] + rng.uniform(420, 620, 5)

    spread = np.round(spread.clip(275, 1600), 0)
    logit = -3.3 + 0.55 * (lev - 4.5) - 0.9 * (coverage - 2.3) - 0.25 * (log_e - np.log(28)) + 0.6 * (seniority == "Second Lien")
    default = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    default[idx[:2]] = 1

    df = pd.DataFrame({
        "loan_id": [f"SYN-{i:04d}" for i in range(1, n + 1)],
        "origination_year": year,
        "sector": sector,
        "seniority": seniority,
        "sponsor_backed": sponsor,
        "ebitda_musd": np.round(ebitda, 1),
        "log_ebitda": np.round(np.log(np.round(ebitda, 1)), 4),
        "leverage_x": np.round(lev, 2),
        "interest_coverage_x": np.round(coverage, 2),
        "ltv_pct": np.round(ltv, 1),
        "spread_bps": spread,
        "default_flag": default,
    })
    # a few genuinely missing values, as in real loan tapes
    miss = rng.choice(n, size=6, replace=False)
    df.loc[miss[:4], "ltv_pct"] = np.nan
    df.loc[miss[4:], "interest_coverage_x"] = np.nan
    return df
