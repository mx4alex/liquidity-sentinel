import pandas as pd
from ru_liquidity_sentinel.aggregation.lsi import LSIAggregator
from ru_liquidity_sentinel.core.dates import today_ts


def test_feature_matrix_excludes_future_dates() -> None:
    today = today_ts()
    future = today + pd.Timedelta(days=30)
    m4 = pd.DataFrame(
        {'tax_week_flag': 0, 'seasonal_factor': 1.2},
        index=pd.date_range(today - pd.Timedelta(days=5), future, freq='D'),
    )
    m1 = pd.DataFrame({'m1_stress': 1.0}, index=[today - pd.Timedelta(days=1), today])
    features = LSIAggregator().build_feature_matrix(m1, pd.DataFrame(), pd.DataFrame(), m4, pd.DataFrame())
    assert features.index.max() <= today


def test_lsi_in_range() -> None:
    idx = pd.date_range('2024-01-01', periods=30, freq='D')
    features = pd.DataFrame({'m1_stress': 1.0, 'm2_stress': 2.0, 'm3_stress': 0.5, 'm5_stress': 1.5, 'tax_week_flag': 0, 'seasonal_factor': 1.0}, index=idx)
    agg = LSIAggregator()
    lsi = agg.compute_lsi(features)
    assert not lsi.empty
    assert lsi['lsi'].between(0, 100).all()
