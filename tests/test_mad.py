import numpy as np
import pandas as pd
from ru_liquidity_sentinel.core.mad import mad_score_to_stress, mad_zscore

def test_mad_zscore_spike_detected() -> None:
    idx = pd.date_range('2020-01-01', periods=400, freq='D')
    rng = np.random.default_rng(42)
    values = pd.Series(rng.normal(0, 1, 400), index=idx)
    values.iloc[-1] = 15.0
    z = mad_zscore(values, window_years=1, min_periods=30)
    assert float(z.iloc[-1]) > 2.0

def test_mad_score_to_stress_clips() -> None:
    z = pd.Series([0, 5, -5])
    s = mad_score_to_stress(z, clip=3.0)
    assert s.max() <= 3.0
    assert s.min() >= -3.0
