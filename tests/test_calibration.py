from __future__ import annotations
import numpy as np
import pandas as pd
from ru_liquidity_sentinel.aggregation.calibration import build_training_mask, fit_module_weights
from ru_liquidity_sentinel.data.fns import _coalesce_tax_dates_for_year, _enp_business_calendar, _parse_nalog_calendar_js

def test_fit_module_weights_prefers_m5() -> None:
    idx = pd.date_range('2020-01-01', periods=200, freq='D')
    gt = pd.Series(np.linspace(0, 100, len(idx)), index=idx)
    features = pd.DataFrame({'m1_stress': np.random.default_rng(0).normal(0, 1, len(idx)), 'm2_stress': np.random.default_rng(1).normal(0, 1, len(idx)), 'm3_stress': np.random.default_rng(2).normal(0, 1, len(idx)), 'm5_stress': gt / 100 + np.random.default_rng(3).normal(0, 0.05, len(idx))}, index=idx)
    train = pd.Series(True, index=idx)
    w = fit_module_weights(features, gt, train)
    assert w['m5'] > w['m1']

def test_build_training_mask_excludes_episode() -> None:
    idx = pd.date_range('2020-01-01', '2022-12-31', freq='D')
    mask = build_training_mask(idx, holdout_start='2020-01-01', episodes=[{'start': '2022-02-01', 'end': '2022-03-31'}])
    assert not mask.loc['2022-02-15']
    assert mask.loc['2021-06-01']

def test_dense_tax_calendar_falls_back_to_enp() -> None:
    dense = [pd.Timestamp(2026, 1, 1) + pd.Timedelta(days=i) for i in range(40)]
    coalesced = _coalesce_tax_dates_for_year(2026, dense)
    assert len(coalesced) == len(_enp_business_calendar(2026))


def test_parse_nalog_calendar_js() -> None:
    html = '$(".calendar").NalogCalendar([{date: new Date(2024, 0, 28), data: []}])'
    dates = _parse_nalog_calendar_js(html, 2024)
    assert len(dates) == 1
    assert dates[0] == pd.Timestamp('2024-01-28')
