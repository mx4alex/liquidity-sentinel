#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ru_liquidity_sentinel.data.cbr import CBRDataCollector
from ru_liquidity_sentinel.data.minfin import MinfinOFZCollector

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "ofz_auctions_historical.csv"


def main() -> None:
    bliq = CBRDataCollector().fetch_bliquidity().set_index("date").sort_index()
    stress = (-bliq["liquidity_balance"].diff(5)).rolling(22).mean()
    stress = (stress - stress.min()) / (stress.max() - stress.min() + 1e-9)

    dates = pd.date_range("2014-01-15", "2025-12-20", freq="2W-FRI")
    rows: list[dict[str, object]] = []
    for dt in dates:
        s = float(stress.asof(dt) if dt in stress.index else stress.reindex([dt], method="ffill").iloc[0])
        cover = float(np.clip(1.8 - 0.9 * s, 0.7, 2.5))
        offered = 200.0 + 50 * np.sin(dt.dayofyear)
        demand = offered * cover
        rows.append(
            {
                "date": dt,
                "series": "PROXY",
                "offered_bn": offered,
                "demand_bn": demand,
                "allocated_bn": min(demand, offered * 1.1),
                "weighted_yield": 8 + 10 * s,
                "cover_ratio": cover,
                "source": "bliquidity_proxy",
            }
        )

    proxy = pd.DataFrame(rows)
    live = MinfinOFZCollector().fetch_ofz_auctions(2024, pd.Timestamp.today().year)
    if "source" not in live.columns:
        live["source"] = "minfin_live"
    combined = (
        pd.concat([proxy, live], ignore_index=True)
        .sort_values("date")
        .drop_duplicates(subset=["date", "series"], keep="last")
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} rows to {OUT}")


if __name__ == "__main__":
    main()
