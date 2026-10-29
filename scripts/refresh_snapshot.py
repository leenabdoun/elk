"""Download fresh public data into data/snapshot/ so the app works even if a live source is unreachable.

Run by the scheduled GitHub Action (.github/workflows/refresh-data.yml), or manually:
    python scripts/refresh_snapshot.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import market_data as md  # noqa: E402

OUT = ROOT / "data" / "snapshot"
EXTRA_TICKERS = ["QQQ", "AGG", "TLT", "EFA", "EEM", "VNQ", "GLD", "MTUM", "USMV", "VLUE", "SIZE", "HYG", "LQD"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = OUT / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    failures = []
    for (name, freq), fn in md.KF_FILES.items():
        key = f"kf_{name}_{freq}"
        try:
            df = md.fetch_kf(name, freq, timeout=60)
            if freq == "daily":
                df = df[df.index >= "1990-01-01"]
            df.round(6).to_csv(OUT / f"{key}.csv")
            meta[key] = {"retrieved": md._now(), "file": fn, "rows": len(df), "last": str(df.index.max().date())}
            print("ok", key, len(df))
        except Exception as e:
            failures.append(f"{key}: {e}")
            print("FAILED", key, e)
    try:
        px = md.fetch_prices(md.DEFAULT_TICKERS + EXTRA_TICKERS, start="1993-01-01")
        px.round(4).to_csv(OUT / "etf_prices_daily.csv")
        meta["etf_prices_daily"] = {"retrieved": md._now(), "tickers": list(px.columns), "rows": len(px),
                                    "last": str(px.index.max().date())}
        print("ok prices", px.shape)
    except Exception as e:
        failures.append(f"prices: {e}")
        print("FAILED prices", e)
    try:
        s = md.fetch_fred("DTB3", timeout=60)
        s[s.index >= "1990-01-01"].to_frame().to_csv(OUT / "fred_dtb3.csv")
        meta["fred_dtb3"] = {"retrieved": md._now(), "rows": len(s), "last": str(s.index.max().date())}
        print("ok fred", len(s))
    except Exception as e:
        failures.append(f"fred: {e}")
        print("FAILED fred", e)
    meta_path.write_text(json.dumps(meta, indent=2))
    if len(failures) == len(md.KF_FILES) + 2:
        sys.exit("all downloads failed")


if __name__ == "__main__":
    main()
