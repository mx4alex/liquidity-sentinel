#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ru_liquidity_sentinel.data.minfin import MinfinOFZCollector

HISTORICAL = Path(__file__).resolve().parents[1] / "data" / "samples" / "ofz_auctions_historical.csv"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="year_from", type=int, default=2014)
    parser.add_argument("--to", dest="year_to", type=int, default=2023)
    args = parser.parse_args()

    collector = MinfinOFZCollector()
    frames: list[pd.DataFrame] = []
    for year in range(args.year_from, args.year_to + 1):
        cache = collector.cache_dir / f"ofz_{year}.parquet"
        if cache.exists():
            cache.unlink()
        df = collector._fetch_year(year)
        if not df.empty:
            if "source" not in df.columns:
                df["source"] = f"wayback_{year}"
            frames.append(df)
            print(f"{year}: {len(df)} rows")

    if not frames:
        print("No Wayback rows fetched.")
        return

    wayback = pd.concat(frames, ignore_index=True)
    wayback["date"] = pd.to_datetime(wayback["date"])

    if HISTORICAL.exists():
        base = pd.read_csv(HISTORICAL)
        base["date"] = pd.to_datetime(base["date"])
        src = base.get("source", pd.Series("bliquidity_proxy", index=base.index)).astype(str)
        years = set(wayback["date"].dt.year.unique())
        drop_proxy = (src == "bliquidity_proxy") & base["date"].dt.year.isin(years)
        base = base.loc[~drop_proxy]
        merged = (
            pd.concat([base, wayback], ignore_index=True)
            .sort_values("date")
            .drop_duplicates(subset=["date", "series"], keep="last")
        )
    else:
        merged = wayback.sort_values("date")

    HISTORICAL.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(HISTORICAL, index=False)
    print(f"Wrote {len(merged)} rows to {HISTORICAL}")


if __name__ == "__main__":
    main()
