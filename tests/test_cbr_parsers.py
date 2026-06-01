from __future__ import annotations
from pathlib import Path
import pandas as pd
import pytest
from ru_liquidity_sentinel.data.cbr import CBRDataCollector, _parse_sors_excel
from ru_liquidity_sentinel.data.cbr_hd_base import normalize_rate_percent

def test_normalize_rate_percent() -> None:
    s = normalize_rate_percent(__import__('pandas').Series([1450, 14.5, 8.0]))
    assert s.iloc[0] == pytest.approx(14.5)
    assert s.iloc[1] == pytest.approx(14.5)

def test_parse_sors_budget_all_wide() -> None:
    path = Path(__file__).resolve().parents[1] / 'data' / 'raw' / '02_29_Budget_all.xlsx'
    if not path.exists():
        pytest.skip('cached SORS xlsx missing')
    raw = pd.read_excel(path, header=None)
    out = _parse_sors_excel(raw)
    assert len(out) > 100
    assert out['date'].min() >= pd.Timestamp('2012-01-01')
    assert out['attracted_bn'].median() > 1

@pytest.mark.integration
def test_fetch_ruonia_live() -> None:
    df = CBRDataCollector().fetch_ruonia()
    assert len(df) > 500
    assert df['ruonia'].between(0, 50).all()
