from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.data import wayback as wayback_archive
from ru_liquidity_sentinel.data.http_client import DataHttpClient
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
MINFIN_BASE = 'https://minfin.gov.ru'
DOC_SLUG = 'rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_{year}_godu'
OFZ_CSV_COLUMNS = ['date', 'series', 'offered_bn', 'demand_bn', 'allocated_bn', 'weighted_yield', 'cover_ratio']

class MinfinOFZCollector:

    def __init__(self, client: DataHttpClient | None=None) -> None:
        self.settings = get_settings()
        self.client = client or DataHttpClient()
        self.cache_dir = self.settings.raw_dir / 'minfin_ofz'
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_csv(self, path: str | pd.PathLike) -> pd.DataFrame:
        return self._standardize(pd.read_csv(path))

    def fetch_ofz_auctions(self, year_from: int=2014, year_to: int | None=None) -> pd.DataFrame:
        if year_to is None:
            year_to = pd.Timestamp.today().year
        frames: list[pd.DataFrame] = []
        for name in ('ofz_auctions_historical.csv', 'ofz_auctions.csv'):
            sample = Path(__file__).resolve().parents[3] / 'data' / 'samples' / name
            if sample.exists():
                frames.append(self.load_from_csv(sample))
        for year in range(year_from, year_to + 1):
            try:
                df_year = self._fetch_year(year)
                if not df_year.empty:
                    frames.append(df_year)
                    logger.info('ofz_year_loaded', year=year, rows=len(df_year))
            except Exception as exc:
                logger.warning('ofz_year_failed', year=year, error=str(exc))
        if not frames:
            logger.warning('ofz_no_data')
            return pd.DataFrame(columns=OFZ_CSV_COLUMNS)
        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=['date', 'series'], keep='last')
        df = self._standardize(df)
        logger.info('ofz_loaded', rows=len(df), from_year=year_from, to_year=year_to)
        return df

    def _standardize(self, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame(columns=OFZ_CSV_COLUMNS)
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        for c in ('offered_bn', 'demand_bn', 'allocated_bn', 'weighted_yield', 'cover_ratio'):
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
        if 'cover_ratio' not in df.columns and {'demand_bn', 'offered_bn'} <= set(df.columns):
            df['cover_ratio'] = df['demand_bn'] / df['offered_bn'].replace(0, pd.NA)
        if 'series' in df.columns:
            df['series'] = df['series'].astype(str)
        return df.sort_values('date').dropna(subset=['date'])

    def _fetch_year(self, year: int) -> pd.DataFrame:
        cache_path = self.cache_dir / f'ofz_{year}.parquet'
        if cache_path.exists():
            return pd.read_parquet(cache_path)
        url = f'{MINFIN_BASE}/ru/document?id_4=315131-{DOC_SLUG.format(year=year)}'
        if year == pd.Timestamp.today().year:
            suffix = pd.Timestamp.today().strftime('_na_%d.%m.%Y')
            url = f'{url}{suffix}'
        html = self.client.get_bytes(url).decode('utf-8', errors='replace')
        xlsx_links = re.findall('href="(/common/upload/library/[^"]+\\.xlsx)"', html, re.I)
        link = next((c for c in xlsx_links if f'/library/{year}/' in c), None)
        if link is None:
            link = next((c for c in xlsx_links if f'_{year}_' in c or f'_{year}.' in c), None)
        frames: list[pd.DataFrame] = []
        if link:
            xlsx_bytes = self.client.get_bytes(f'{MINFIN_BASE}{link}')
            (self.cache_dir / f'ofz_{year}.xlsx').write_bytes(xlsx_bytes)
            frames.append(_filter_auction_year(parse_minfin_auction_xlsx(io.BytesIO(xlsx_bytes)), year))
        try:
            tables = pd.read_html(io.StringIO(html))
            if tables:
                html_df = _filter_auction_year(_parse_minfin_html_table(max(tables, key=len)), year)
                if not html_df.empty:
                    html_df['source'] = 'minfin_html'
                    frames.append(html_df)
        except ValueError:
            pass
        if not frames:
            snap = wayback_archive.find_minfin_ofz_snapshot(year)
            if snap:
                ts, orig = snap
                logger.info('wayback_snapshot_found', year=year, timestamp=ts)
                try:
                    html_wb = wayback_archive.fetch_wayback_html(ts, orig)
                    tables = pd.read_html(io.StringIO(html_wb))
                    if tables:
                        html_df = _filter_auction_year(_parse_minfin_html_table(max(tables, key=len)), year)
                        if not html_df.empty:
                            html_df['source'] = 'wayback_html'
                            frames.append(html_df)
                    for path in wayback_archive.extract_xlsx_links(html_wb, year):
                        raw = wayback_archive.fetch_wayback_xlsx(ts, path)
                        if raw:
                            wb_df = _filter_auction_year(parse_minfin_auction_xlsx(io.BytesIO(raw)), year)
                            if not wb_df.empty:
                                wb_df['source'] = 'wayback_xlsx'
                                frames.append(wb_df)
                except Exception as exc:
                    logger.warning('wayback_fetch_failed', year=year, error=str(exc))
        if not frames:
            return pd.DataFrame()
        df = pd.concat(frames, ignore_index=True).drop_duplicates(subset=['date', 'series'])
        if not df.empty:
            df.to_parquet(cache_path)
        return df

def _filter_auction_year(df: pd.DataFrame, year: int) -> pd.DataFrame:
    if df.empty or 'date' not in df.columns:
        return df
    dates = pd.to_datetime(df['date'], errors='coerce')
    return df.loc[dates.dt.year == year].copy()

def _flatten_html_columns(columns: pd.Index) -> list[str]:
    flat: list[str] = []
    for col in columns:
        if isinstance(col, tuple):
            parts = [str(p).strip() for p in col if p is not None and 'Unnamed' not in str(p)]
            flat.append(parts[0] if parts else str(col[-1]))
        else:
            flat.append(str(col).strip())
    return flat

def _parse_minfin_html_table(table: pd.DataFrame) -> pd.DataFrame:
    t = table.copy()
    t.columns = _flatten_html_columns(t.columns)
    t = t.rename(columns={c: c.strip() for c in t.columns})
    first = t.iloc[0].astype(str).str.strip()
    if first.str.fullmatch('\\d+').sum() >= max(3, len(t.columns) // 3):
        t = t.iloc[1:].reset_index(drop=True)
    colmap = {'date': _pick_col(list(t.columns), 'дата'), 'series': _pick_col(list(t.columns), 'код'), 'offered_bn': _pick_col(list(t.columns), 'предлож'), 'demand_bn': _pick_col(list(t.columns), 'спрос'), 'allocated_bn': _pick_col(list(t.columns), 'размещ'), 'weighted_yield': _pick_col(list(t.columns), 'доход', 'отсеч'), 'cover_ratio': _pick_col(list(t.columns), 'коэфф', 'удовлетвор')}
    if not colmap['date']:
        return pd.DataFrame()
    out = pd.DataFrame()
    out['date'] = pd.to_datetime(t[colmap['date']], errors='coerce')
    if colmap['series']:
        out['series'] = t[colmap['series']].astype(str)
    for key, col in (('offered_bn', colmap['offered_bn']), ('demand_bn', colmap['demand_bn']), ('allocated_bn', colmap['allocated_bn'])):
        if col:
            out[key] = pd.to_numeric(t[col], errors='coerce') / 1000.0
    if colmap['weighted_yield']:
        out['weighted_yield'] = pd.to_numeric(t[colmap['weighted_yield']], errors='coerce')
    if colmap['cover_ratio']:
        out['cover_ratio'] = pd.to_numeric(t[colmap['cover_ratio']], errors='coerce')
    fmt_col = _pick_col(list(t.columns), 'формат')
    if fmt_col and len(out) == len(t):
        out = out[t[fmt_col].astype(str).str.contains('аукцион', case=False, na=False).values]
    return out.dropna(subset=['date'])

def parse_minfin_auction_xlsx(source: io.BytesIO | Path) -> pd.DataFrame:
    raw = pd.read_excel(source, sheet_name=0, header=None)
    header_row = None
    for i in range(min(15, len(raw))):
        row = raw.iloc[i].astype(str).str.lower()
        if row.str.contains('дата', na=False).any() and row.str.contains('предлож', na=False).any():
            header_row = i
            break
    if header_row is None:
        return pd.DataFrame()
    headers = [str(x).strip().lower() for x in raw.iloc[header_row].tolist()]
    data = raw.iloc[header_row + 1:].copy()
    data.columns = headers
    col_date = _pick_col(headers, 'дата')
    col_series = _pick_col(headers, 'код', 'выпуск')
    col_offered = _pick_col(headers, 'предлож')
    col_demand = _pick_col(headers, 'спрос')
    col_allocated = _pick_col(headers, 'размещ')
    col_yield = _pick_col(headers, 'доход', 'отсеч')
    col_format = _pick_col(headers, 'формат')
    col_cover = _pick_col(headers, 'коэфф', 'удовлетвор')
    if not col_date:
        return pd.DataFrame()
    if col_format:
        fmt = data[col_format].astype(str).str.lower()
        data = data[fmt.str.contains('аукцион', na=False)]
    df = pd.DataFrame()
    df['date'] = pd.to_datetime(data[col_date], errors='coerce')
    if col_series:
        df['series'] = data[col_series].astype(str)
    if col_offered:
        df['offered_bn'] = pd.to_numeric(data[col_offered], errors='coerce') / 1000.0
    if col_demand:
        df['demand_bn'] = pd.to_numeric(data[col_demand], errors='coerce') / 1000.0
    if col_allocated:
        df['allocated_bn'] = pd.to_numeric(data[col_allocated], errors='coerce') / 1000.0
    if col_yield:
        df['weighted_yield'] = pd.to_numeric(data[col_yield], errors='coerce')
    if col_cover:
        df['cover_ratio'] = pd.to_numeric(data[col_cover], errors='coerce')
    elif {'demand_bn', 'offered_bn'} <= set(df.columns):
        df['cover_ratio'] = df['demand_bn'] / df['offered_bn'].replace(0, pd.NA)
    return df.dropna(subset=['date'])

def _norm_header(h: str) -> str:
    return str(h).casefold().replace('\xa0', ' ')

def _pick_col(headers: list[str], *substrings: str) -> str | None:
    norms = [_norm_header(h) for h in headers]
    subs = [s.casefold() for s in substrings]
    for h, nh in zip(headers, norms):
        if h and all(s in nh for s in subs):
            return h
    for h, nh in zip(headers, norms):
        if h and any(s in nh for s in subs):
            return h
    return None
