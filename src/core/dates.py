from __future__ import annotations

import pandas as pd


def to_timestamp(value: object) -> pd.Timestamp:
    return pd.Timestamp(value).normalize()

def date_range_index(start: str | pd.Timestamp, end: str | pd.Timestamp, freq: str='D') -> pd.DatetimeIndex:
    return pd.date_range(start=to_timestamp(start), end=to_timestamp(end), freq=freq)

def business_days_between(start: pd.Timestamp, end: pd.Timestamp) -> int:
    return len(pd.bdate_range(start, end)) - 1


def today_ts() -> pd.Timestamp:
    return pd.Timestamp.today().normalize()


def truncate_index_to_today(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return df.loc[df.index <= today_ts()]
