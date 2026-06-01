from __future__ import annotations

import io
import re
from pathlib import Path

import pandas as pd

from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.data.cbr_hd_base import normalize_volume_mln_to_bn
from ru_liquidity_sentinel.data.http_client import DataHttpClient
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)

class RoskaznaCollector:

    def __init__(self, client: DataHttpClient | None=None) -> None:
        self.settings = get_settings()
        self.client = client or DataHttpClient()
        self.url = self.settings.get('sources', 'roskazna_deposits', default='https://roskazna.gov.ru/finansovye-operacii/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah')
        self._fallback_urls = [self.url, 'https://roskazna.gov.ru/activity/razmeshchenie-sredstv-na-depozitah/', 'http://roskazna.gov.ru/finansovye-operacii/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta/razmeshchenie-sredstv-edinogo-kaznachejskogo-scheta-na-bankovskih-depozitah/']

    def load_from_csv(self, path: str | pd.PathLike) -> pd.DataFrame:
        df = pd.read_csv(path, parse_dates=['date'])
        return df.sort_values('date')

    def fetch_deposit_placements(self) -> pd.DataFrame:
        sample = Path(__file__).resolve().parents[3] / 'data' / 'samples' / 'roskazna_deposits.csv'
        parsed_frames: list[pd.DataFrame] = []
        for url in dict.fromkeys(self._fallback_urls):
            try:
                html = self.client.get_bytes(url).decode('utf-8', errors='replace')
                chunk = self._parse_deposits_html(html)
                if not chunk.empty:
                    parsed_frames.append(chunk)
                    logger.info('roskazna_parsed', url=url[:60], rows=len(chunk))
            except Exception as exc:
                logger.debug('roskazna_url_failed', url=url[:60], error=str(exc))
        if parsed_frames:
            parsed = pd.concat(parsed_frames, ignore_index=True).drop_duplicates('date').sort_values('date')
            if sample.exists():
                parsed = pd.concat([self.load_from_csv(sample), parsed], ignore_index=True).drop_duplicates('date', keep='last').sort_values('date')
            return parsed
        if sample.exists():
            df = self.load_from_csv(sample)
            logger.info('roskazna_sample_fallback', rows=len(df))
            return df
        sors_proxy = self._proxy_from_sors()
        if not sors_proxy.empty:
            logger.info('roskazna_sors_proxy', rows=len(sors_proxy))
            return sors_proxy
        logger.warning('roskazna_no_data')
        return pd.DataFrame(columns=['date', 'placement_bn'])

    def _proxy_from_sors(self) -> pd.DataFrame:
        try:
            from ru_liquidity_sentinel.data.cbr import CBRDataCollector
            sors = CBRDataCollector().fetch_sors_attracted_funds()
            if sors.empty:
                return pd.DataFrame()
            sors = sors.sort_values('date')
            delta = sors['attracted_bn'].diff().clip(lower=0).rolling(3, min_periods=1).mean()
            med = float(delta.replace(0, pd.NA).median() or 100)
            if med > 5000:
                delta = delta / 1000.0
            out = pd.DataFrame({'date': sors['date'], 'placement_bn': delta.clip(50, 2000)})
            return out.dropna(subset=['placement_bn']).query('placement_bn > 0')
        except Exception as exc:
            logger.debug('roskazna_sors_proxy_failed', error=str(exc))
            return pd.DataFrame()

    def _parse_deposits_html(self, html: str) -> pd.DataFrame:
        records: list[dict[str, object]] = []
        try:
            tables = pd.read_html(io.StringIO(html))
        except ValueError:
            tables = []
        for table in tables:
            date_col = None
            val_col = None
            for col in table.columns:
                cl = str(col).lower()
                if 'дата' in cl or 'date' in cl or 'период' in cl:
                    date_col = col
                if any(x in cl for x in ('размещ', 'объем', 'объём', 'сумм')):
                    val_col = col
            if date_col and val_col:
                for _, row in table.iterrows():
                    dt = pd.to_datetime(row[date_col], dayfirst=True, errors='coerce')
                    if pd.isna(dt):
                        continue
                    val = normalize_volume_mln_to_bn(pd.Series([row[val_col]])).iloc[0]
                    if pd.notna(val):
                        records.append({'date': dt, 'placement_bn': float(val)})
        if not records:
            for m in re.finditer('(\\d{2}\\.\\d{2}\\.\\d{4}).{0,80}?(\\d[\\d\\s,]+)\\s*(?:млрд|млн)', html, re.I | re.S):
                dt = pd.to_datetime(m.group(1), dayfirst=True)
                raw = m.group(2).replace(' ', '').replace(',', '.')
                val = float(raw)
                if 'млн' in m.group(0).lower():
                    val /= 1000
                records.append({'date': dt, 'placement_bn': val})
        if not records:
            return pd.DataFrame(columns=['date', 'placement_bn'])
        return pd.DataFrame(records).drop_duplicates('date').sort_values('date')
