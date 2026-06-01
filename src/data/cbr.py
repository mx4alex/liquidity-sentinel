from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
from bs4 import BeautifulSoup

from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.data.cbr_hd_base import (
    find_column,
    get_hd_base_dynamics,
    normalize_rate_percent,
    normalize_volume_mln_to_bn,
    parse_repo_summary_table,
    post_hd_base_table,
)
from ru_liquidity_sentinel.data.http_client import DataHttpClient
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
_REPO_BASE = 'https://www.cbr.ru/hd_base/repo/'
_CACHE_REPO_ENRICHED = 'repo_auctions_enriched.parquet'

class CBRDataCollector:

    def __init__(self, client: DataHttpClient | None=None) -> None:
        self.settings = get_settings()
        self.client = client or DataHttpClient()
        self.sources = self.settings.get('sources', default={})

    def fetch_required_reserves(self) -> pd.DataFrame:
        url = self.sources.get('cbr_reserves_xlsx', 'https://www.cbr.ru/vfs/hd_base/RReserves/required_reserves_table.xlsx')
        path = self.client.download(url, 'required_reserves_table.xlsx')
        raw = pd.read_excel(path, sheet_name=0, header=None)
        df = _parse_reserves_excel(raw)
        logger.info('reserves_loaded', rows=len(df))
        return df

    def fetch_ruonia(self, date_from: str='01.01.2014') -> pd.DataFrame:
        today = pd.Timestamp.today().strftime('%d.%m.%Y')
        table = get_hd_base_dynamics(self.client, '/hd_base/ruonia', date_from=date_from, date_to=today)
        date_col = find_column(table, 'дата') or table.columns[0]
        rate_col = find_column(table, 'ruonia', 'ставка') or table.columns[1]
        out = pd.DataFrame({'date': pd.to_datetime(table[date_col], dayfirst=True, errors='coerce'), 'ruonia': normalize_rate_percent(table[rate_col])})
        out = out.dropna().sort_values('date')
        logger.info('ruonia_loaded', rows=len(out))
        return out

    def fetch_keyrate(self, date_from: str='01.01.2014') -> pd.DataFrame:
        today = pd.Timestamp.today().strftime('%d.%m.%Y')
        base = self.sources.get('cbr_keyrate', 'https://www.cbr.ru/hd_base/keyrate/')
        table = post_hd_base_table(self.client, base, date_from=date_from, date_to=today)
        date_col = find_column(table, 'дата') or table.columns[0]
        rate_col = find_column(table, 'ставка') or table.columns[1]
        out = pd.DataFrame({'date': pd.to_datetime(table[date_col], dayfirst=True, errors='coerce'), 'key_rate': normalize_rate_percent(table[rate_col])})
        out = out.dropna().sort_values('date')
        logger.info('keyrate_loaded', rows=len(out))
        return out

    def fetch_repo_auctions(self, date_from: str='01.01.2010', enrich_demand: bool=True) -> pd.DataFrame:
        cache_path = self.settings.processed_dir / _CACHE_REPO_ENRICHED
        if cache_path.exists() and (not enrich_demand):
            return _normalize_repo_rates(pd.read_parquet(cache_path))
        today = pd.Timestamp.today().strftime('%d.%m.%Y')
        table = post_hd_base_table(self.client, _REPO_BASE, date_from=date_from, date_to=today)
        df = _parse_repo_bulk_table(table)
        if df.empty:
            df = _load_repo_sample()
            logger.warning('repo_bulk_empty_using_sample', rows=len(df))
        elif enrich_demand:
            df = self._enrich_repo_demand(df, cache_path)
        df = _normalize_repo_rates(df)
        logger.info('repo_loaded', rows=len(df), term7=len(df[df['term_days'] == 7]))
        return df

    def _enrich_repo_demand(self, df: pd.DataFrame, cache_path: Path) -> pd.DataFrame:
        if cache_path.exists():
            cached = pd.read_parquet(cache_path)
            if len(cached) >= len(df) * 0.9:
                return cached
        focus = df[df['term_days'] == 7].copy()
        records: list[dict[str, Any]] = []
        for n, (_, row) in enumerate(focus.iterrows()):
            d_str = pd.Timestamp(row['date']).strftime('%d.%m.%Y')
            try:
                summary = post_hd_base_table(self.client, _REPO_BASE, date_from=d_str, date_to=d_str)
                parsed = parse_repo_summary_table(summary)
                if parsed:
                    rec = row.to_dict()
                    rec.update(parsed)
                    records.append(rec)
                if n and n % 50 == 0:
                    time.sleep(0.12)
                    logger.info('repo_enrichment_progress', done=n)
            except Exception as exc:
                logger.debug('repo_day_fetch_failed', date=d_str, error=str(exc))
        if records:
            enriched = pd.DataFrame(records)
            enriched['date'] = pd.to_datetime(enriched['date'])
            if 'demand_bn' in enriched.columns and 'allocated_bn' in enriched.columns:
                enriched['cover_ratio'] = enriched['demand_bn'] / enriched['allocated_bn'].replace(0, pd.NA)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            enriched.to_parquet(cache_path)
            return enriched
        return df

    def fetch_bliquidity(self, date_from: str='01.02.2014') -> pd.DataFrame:
        base = self.sources.get('cbr_bliquidity', 'https://www.cbr.ru/hd_base/bliquidity/')
        today = pd.Timestamp.today().strftime('%d.%m.%Y')
        url = f"{base.rstrip('/')}/?UniDbQuery.Posted=True&UniDbQuery.From={date_from}&UniDbQuery.To={today}"
        html = self.client.get_bytes(url).decode('utf-8', errors='replace')
        df = _parse_cbr_html_table(html)
        out = _coerce_bliquidity_table(df)
        logger.info('bliquidity_loaded', rows=len(out))
        return out

    def fetch_sors_attracted_funds(self) -> pd.DataFrame:
        url = self.sources.get('cbr_sors', 'https://www.cbr.ru/statistics/bank_sector/sors/')
        html = self.client.get_bytes(url).decode('utf-8', errors='replace')
        xlsx_links = re.findall('href="(/vfs/statistics/[^"]+\\.xlsx)"', html, re.I)
        prioritized = sorted(xlsx_links, key=lambda u: 0 if 'budget' in u.lower() else 1 if 'funds' in u.lower() and 'borrow' in u.lower() else 2 if any(x in u.lower() for x in ('privlech', 'privl')) else 3)
        for link in prioritized[:5]:
            full_url = f'https://www.cbr.ru{link}'
            try:
                path = self.client.download(full_url, Path(link).name)
                raw = pd.read_excel(path, sheet_name=0, header=None)
                parsed = _parse_sors_excel(raw)
                if not parsed.empty:
                    logger.info('sors_loaded', rows=len(parsed), file=link)
                    return parsed
            except Exception as exc:
                logger.debug('sors_file_skip', link=link, error=str(exc))
        logger.warning('sors_not_loaded')
        return pd.DataFrame(columns=['date', 'attracted_bn'])

def _normalize_repo_rates(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    for col in ('cutoff_rate', 'weighted_rate'):
        if col in df.columns:
            df[col] = normalize_rate_percent(df[col])
    return df


def _parse_repo_bulk_table(table: pd.DataFrame) -> pd.DataFrame:
    if table.empty:
        return pd.DataFrame()
    date_col = find_column(table, 'дата') or table.columns[2]
    term_col = find_column(table, 'срок') or 'Срок, дни'
    alloc_col = find_column(table, 'заключенных') or find_column(table, 'объем')
    rate_col = find_column(table, 'средневзвеш') or find_column(table, 'ставка')
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(table[date_col], dayfirst=True, errors='coerce')
    if term_col in table.columns:
        out['term_days'] = pd.to_numeric(table[term_col].astype(str).str.extract('(\\d+)')[0], errors='coerce')
    if alloc_col and alloc_col in table.columns:
        out['allocated_bn'] = normalize_volume_mln_to_bn(table[alloc_col])
    if rate_col and rate_col in table.columns:
        out['weighted_rate'] = normalize_rate_percent(table[rate_col])
        out['cutoff_rate'] = out['weighted_rate']
    if 'cutoff_rate' in out.columns:
        out['cutoff_rate'] = normalize_rate_percent(out['cutoff_rate'])
    out = out.dropna(subset=['date'])
    if 'demand_bn' not in out.columns:
        out['demand_bn'] = out['allocated_bn'] * 1.2
    out['cover_ratio'] = out['demand_bn'] / out['allocated_bn'].replace(0, pd.NA)
    return out.drop_duplicates(subset=['date', 'term_days'], keep='last')

def _load_repo_sample() -> pd.DataFrame:
    sample = Path(__file__).resolve().parents[3] / 'data' / 'samples' / 'repo_auctions.csv'
    if sample.exists():
        df = pd.read_csv(sample, parse_dates=['date'])
        return _standardize_repo_columns(df)
    return pd.DataFrame()

def _parse_reserves_excel(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.shape[0] < 4:
        return pd.DataFrame()
    data = raw.iloc[3:].copy()
    period = pd.to_datetime(data.iloc[:, 0], errors='coerce')
    data = data[period.notna()].copy()
    out = pd.DataFrame({'period_start': pd.to_datetime(data.iloc[:, 0], errors='coerce'), 'actual_balances': pd.to_numeric(data.iloc[:, 1], errors='coerce'), 'required_averaging': pd.to_numeric(data.iloc[:, 2], errors='coerce')})
    if raw.shape[1] > 3:
        out['required_accounts'] = pd.to_numeric(data.iloc[:, 3], errors='coerce')
    if raw.shape[1] > 7:
        out['period_days'] = pd.to_numeric(data.iloc[:, 7], errors='coerce')
    out['reserve_spread'] = out['actual_balances'] - out['required_averaging']
    return out.dropna(subset=['period_start', 'actual_balances'])

def _parse_sors_excel(raw: pd.DataFrame) -> pd.DataFrame:
    wide = _parse_sors_wide_layout(raw)
    if not wide.empty:
        return wide
    for header_row in range(min(20, len(raw))):
        row = raw.iloc[header_row].astype(str).str.lower()
        if not row.str.contains('дата|месяц|период|date|month', regex=True, na=False).any():
            continue
        headers = [str(x).strip().lower() for x in raw.iloc[header_row]]
        body = raw.iloc[header_row + 1:].copy()
        body.columns = headers
        date_col = next((h for h in headers if h and any((x in h for x in ('дата', 'месяц', 'период')))), None)
        val_col = next((h for h in headers if h and any((x in h for x in ('бюджет', 'привлеч', 'остат', 'средств', 'объем', 'объём')))), None)
        if date_col and val_col:
            out = pd.DataFrame({'date': pd.to_datetime(body[date_col], dayfirst=True, errors='coerce'), 'attracted_bn': normalize_volume_mln_to_bn(body[val_col])})
            out = out.dropna()
            if len(out) > 12:
                return out.sort_values('date')
    return pd.DataFrame(columns=['date', 'attracted_bn'])

def _parse_sors_wide_layout(raw: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for header_row in range(min(8, len(raw))):
        dates = pd.to_datetime(raw.iloc[header_row, 1:], dayfirst=True, errors='coerce')
        if dates.notna().sum() < 12:
            continue
        for val_row in range(header_row + 1, len(raw)):
            label = str(raw.iloc[val_row, 0]).lower()
            if 'всего' not in label and (not label.strip().startswith('остатки')):
                continue
            vals = pd.to_numeric(raw.iloc[val_row, 1:], errors='coerce')
            for dt, val in zip(dates, vals, strict=False):
                if pd.notna(dt) and pd.notna(val):
                    records.append({'date': dt, 'attracted_bn': float(normalize_volume_mln_to_bn(pd.Series([val])).iloc[0])})
            break
        if records:
            break
    if not records:
        return pd.DataFrame(columns=['date', 'attracted_bn'])
    out = pd.DataFrame(records).drop_duplicates('date').sort_values('date')
    out = out[out['date'] >= pd.Timestamp('2000-01-01')]
    return out

def _parse_cbr_html_table(html: str) -> pd.DataFrame:
    try:
        tables = pd.read_html(io.StringIO(html))
        if tables:
            return max(tables, key=len)
    except ValueError:
        pass
    soup = BeautifulSoup(html, 'lxml')
    table = soup.find('table')
    if table is None:
        return pd.DataFrame()
    rows: list[list[str]] = []
    for tr in table.find_all('tr'):
        cells = [td.get_text(strip=True) for td in tr.find_all(['td', 'th'])]
        if cells:
            rows.append(cells)
    if not rows:
        return pd.DataFrame()
    ncols = max(len(r) for r in rows)
    normalized = [r + [''] * (ncols - len(r)) for r in rows]
    return pd.DataFrame(normalized[1:], columns=normalized[0])

def _coerce_bliquidity_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=['date', 'liquidity_balance'])
    date_col = find_column(df, 'дата') or df.columns[0]
    val_col = None
    for col in df.columns:
        if col == date_col:
            continue
        nums = pd.to_numeric(df[col].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')
        if nums.notna().sum() > len(df) * 0.3:
            val_col = col
            break
    if val_col is None:
        return pd.DataFrame(columns=['date', 'liquidity_balance'])
    out = pd.DataFrame({'date': pd.to_datetime(df[date_col], dayfirst=True, errors='coerce'), 'liquidity_balance': pd.to_numeric(df[val_col].astype(str).str.replace(',', '.').str.replace(' ', ''), errors='coerce')})
    return out.dropna().sort_values('date')

def _standardize_repo_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    for col in ('demand_bn', 'allocated_bn', 'cutoff_rate', 'weighted_rate'):
        if col in df.columns:
            if 'rate' in col:
                df[col] = normalize_rate_percent(df[col])
            elif 'bn' in col:
                df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'cover_ratio' not in df.columns and {'demand_bn', 'allocated_bn'} <= set(df.columns):
        df['cover_ratio'] = df['demand_bn'] / df['allocated_bn'].replace(0, pd.NA)
    return df.dropna(subset=['date'])
