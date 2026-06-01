from __future__ import annotations
import io
import httpx
import pandas as pd
import pytest
from ru_liquidity_sentinel.data.minfin import _parse_minfin_html_table, parse_minfin_auction_xlsx

def test_parse_wayback_style_html_table() -> None:
    cols = pd.MultiIndex.from_tuples([('Дата аукциона', 'Дата аукциона', '1'), ('Код  выпуска', 'Код  выпуска', '2'), ('Объем предложения', 'млн. рублей', '6'), ('Совокупный объем спроса', 'млн. рублей', '9'), ('Коэффициент удовлетворения спроса на аукционе', '(12/11)', '14')])
    raw = pd.DataFrame([['1', '2', '6', '9', '14'], ['2022-01-12', '26238RMFS', 150000, 65280, 0.4352], ['2022-02-02', '26239RMFS', 200000, 145220, 0.7261]], columns=cols)
    df = _parse_minfin_html_table(raw)
    assert len(df) == 2
    assert df['date'].dt.year.tolist() == [2022, 2022]
    assert df['cover_ratio'].iloc[0] == pytest.approx(0.4352)

@pytest.mark.integration
def test_parse_live_2026_xlsx() -> None:
    url = 'https://minfin.gov.ru/ru/document?id_4=315131-rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_2026_godu_na_28.05.2026'
    html = httpx.get(url, timeout=30, headers={'User-Agent': 'Mozilla/5.0'}).text
    import re
    link = re.findall('href="(/common/upload/library/[^"]+\\.xlsx)"', html)[0]
    data = httpx.get('https://minfin.gov.ru' + link, timeout=60).content
    df = parse_minfin_auction_xlsx(io.BytesIO(data))
    assert len(df) > 20
    assert df['cover_ratio'].between(0, 10).all()
    assert df['offered_bn'].notna().all()
