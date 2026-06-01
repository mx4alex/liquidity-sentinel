from __future__ import annotations

import io
import re
from typing import Any

import pandas as pd

from ru_liquidity_sentinel.data.http_client import DataHttpClient
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
_RATE_SCALE = 100.0
_RATE_PERCENT_CAP = 40.0
_MLN_TO_BLN = 1000.0

def post_hd_base_table(client: DataHttpClient, base_url: str, date_from: str='01.01.2014', date_to: str | None=None, extra: dict[str, str] | None=None) -> pd.DataFrame:
    if date_to is None:
        date_to = pd.Timestamp.today().strftime('%d.%m.%Y')
    payload: dict[str, str] = {'UniDbQuery.Posted': 'True', 'UniDbQuery.From': date_from, 'UniDbQuery.To': date_to}
    if extra:
        payload.update(extra)
    text = client.post_form(base_url, payload, use_cache=True)
    tables = pd.read_html(io.StringIO(text))
    if not tables:
        return pd.DataFrame()
    return max(tables, key=len)

def get_hd_base_dynamics(client: DataHttpClient, path: str, date_from: str='01.01.2014', date_to: str | None=None) -> pd.DataFrame:
    if date_to is None:
        date_to = pd.Timestamp.today().strftime('%d.%m.%Y')
    url = f"https://www.cbr.ru{path.rstrip('/')}/dynamics/?UniDbQuery.Posted=True&UniDbQuery.From={date_from}&UniDbQuery.To={date_to}"
    html = client.get_bytes(url).decode('utf-8', errors='replace')
    tables = pd.read_html(io.StringIO(html))
    if not tables:
        return pd.DataFrame()
    return max(tables, key=len)

def normalize_rate_percent(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace('[^\\d.\\-]', '', regex=True), errors='coerce')
    return s.where(s <= _RATE_PERCENT_CAP, s / _RATE_SCALE)

def normalize_volume_mln_to_bn(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series.astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')
    return s / _MLN_TO_BLN

def parse_repo_summary_table(table: pd.DataFrame) -> dict[str, Any]:
    if table.shape[1] < 2:
        return {}
    keys = table.iloc[:, 0].astype(str).str.strip().str.lower()
    vals = table.iloc[:, 1]
    out: dict[str, Any] = {}
    for k, v in zip(keys, vals, strict=False):
        if 'спрос' in k:
            out['demand_bn'] = float(normalize_volume_mln_to_bn(pd.Series([v])).iloc[0])
        elif 'заключенных сделок' in k and 'спрос' not in k:
            out['allocated_bn'] = float(normalize_volume_mln_to_bn(pd.Series([v])).iloc[0])
        elif 'отсеч' in k:
            out['cutoff_rate'] = float(normalize_rate_percent(pd.Series([v])).iloc[0])
        elif 'средневзвеш' in k and 'отсеч' not in k:
            out['weighted_rate'] = float(normalize_rate_percent(pd.Series([v])).iloc[0])
        elif re.search('срок.*дн', k):
            out['term_days'] = float(pd.to_numeric(str(v).replace('д', '').strip(), errors='coerce'))
    return out

def find_column(df: pd.DataFrame, *substrings: str) -> str | None:
    for col in df.columns:
        cl = str(col).lower()
        if all(s in cl for s in substrings):
            return col
    for col in df.columns:
        cl = str(col).lower()
        if any(s in cl for s in substrings):
            return col
    return None
