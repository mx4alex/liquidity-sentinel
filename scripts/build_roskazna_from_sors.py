#!/usr/bin/env python3

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ru_liquidity_sentinel.data.cbr import CBRDataCollector

OUT = Path(__file__).resolve().parents[1] / "data" / "samples" / "roskazna_deposits.csv"
LIVE_SAMPLE_ROWS = 15


def main() -> None:
    sors = CBRDataCollector().fetch_sors_attracted_funds()
    if sors.empty:
        print("SORS empty")
        return

    sors = sors.sort_values("date").drop_duplicates("date")
    delta = sors["attracted_bn"].diff()
    placement = delta.clip(lower=0).rolling(3, min_periods=1).mean()
    placement = placement.fillna(0)
    med = float(placement.replace(0, np.nan).median() or 100)
    if med > 5000:
        placement = placement / 1000.0
    placement = placement.clip(50, 2000)

    proxy = pd.DataFrame(
        {
            "date": sors["date"],
            "placement_bn": placement,
            "source": "sors_proxy",
        }
    )
    proxy = proxy[(proxy["placement_bn"] > 0) & (proxy["date"] >= "2000-01-01")]

    if OUT.exists():
        live = pd.read_csv(OUT, parse_dates=["date"])
        if "source" not in live.columns:
            live["source"] = "sample"
        recent = live[live["date"] >= "2024-01-01"]
        combined = (
            pd.concat([proxy[proxy["date"] < "2024-01-01"], recent], ignore_index=True)
            .drop_duplicates("date", keep="last")
            .sort_values("date")
        )
    else:
        combined = proxy.sort_values("date")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    combined[["date", "placement_bn"]].to_csv(OUT, index=False)
    print(f"Wrote {len(combined)} rows to {OUT}")


if __name__ == "__main__":
    main()
