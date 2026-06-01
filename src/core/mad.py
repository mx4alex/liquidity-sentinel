from __future__ import annotations

import numpy as np
import pandas as pd

MAD_SCALE = 1.4826

def rolling_mad(series: pd.Series, window: int, min_periods: int | None=None) -> pd.Series:
    if min_periods is None:
        min_periods = max(window // 4, 10)

    def _mad(x: np.ndarray) -> float:
        if len(x) < min_periods:
            return np.nan
        med = np.nanmedian(x)
        return float(np.nanmedian(np.abs(x - med)))
    return series.rolling(window=window, min_periods=min_periods).apply(_mad, raw=True)

def mad_zscore(series: pd.Series, window_years: int=3, min_periods: int=30, freq: str='D') -> pd.Series:
    if series.empty:
        return series.copy()
    idx = series.index
    if isinstance(idx, pd.DatetimeIndex) and len(idx) > 1:
        days = (idx[-1] - idx[0]).days / max(len(idx) - 1, 1)
        if days <= 2:
            window = window_years * 12
        elif days <= 10:
            window = window_years * 52
        else:
            window = window_years * 365
    else:
        window = window_years * 365
    window = int(window)
    min_periods = min(min_periods, window)
    roll_med = series.rolling(window=window, min_periods=min_periods).median()
    roll_mad = rolling_mad(series, window=window, min_periods=min_periods)
    denom = MAD_SCALE * roll_mad
    roll_std = series.rolling(window=window, min_periods=min_periods).std()
    denom = denom.where(denom > 0, roll_std)
    denom = denom.replace(0, np.nan)
    z = (series - roll_med) / denom
    return z.replace([np.inf, -np.inf], np.nan)

def mad_score_to_stress(z: pd.Series, clip: float=3.0) -> pd.Series:
    return z.clip(-clip, clip).fillna(0.0)
