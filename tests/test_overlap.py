import pandas as pd
from ru_liquidity_sentinel.aggregation.overlap import apply_overlap_adjustment, compute_overlap_report

def test_tax_week_downweight_reduces_m2() -> None:
    w = {'m1': 0.25, 'm2': 0.25, 'm3': 0.25, 'm5': 0.25}
    adj = apply_overlap_adjustment(w, tax_week=True)
    assert adj['m2'] < w['m2']
    assert abs(sum(adj.values()) - 1.0) < 1e-09

def test_overlap_report() -> None:
    import numpy as np
    idx = pd.date_range('2024-01-01', periods=60, freq='D')
    rng = np.random.default_rng(0)
    features = pd.DataFrame({'m1_stress': rng.normal(0, 1, 60), 'm2_stress': rng.normal(0, 1, 60), 'tax_week_flag': [1] * 20 + [0] * 40}, index=idx)
    report = compute_overlap_report(features, ['m1_stress', 'm2_stress'])
    assert report.method == 'conditional_downweight'
