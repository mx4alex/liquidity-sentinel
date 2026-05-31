from __future__ import annotations

import re

import pandas as pd
from bs4 import BeautifulSoup

from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.data.http_client import DataHttpClient
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
_NALOG_DATE_RE = re.compile('date:\\s*new Date\\((\\d{4}),\\s*(\\d{1,2}),\\s*(\\d{1,2})\\)', re.IGNORECASE)
_MAX_PAYMENT_DATES_PER_YEAR = 28

def _enp_business_calendar(year: int) -> list[pd.Timestamp]:
    dates: list[pd.Timestamp] = []
    for month in range(1, 13):
        for day in (25, 28):
            try:
                ts = pd.Timestamp(year=year, month=month, day=day)
            except ValueError:
                continue
            if ts.weekday() >= 5:
                ts = ts + pd.offsets.BusinessDay(1)
            dates.append(ts)
    return dates

class FNSCalendarCollector:

    def __init__(self, client: DataHttpClient | None=None) -> None:
        self.settings = get_settings()
        self.client = client or DataHttpClient()
        self.url = self.settings.get('sources', 'fns_calendar', default='https://www.nalog.gov.ru/rn77/calendar/')

    def fetch_tax_dates(self, year: int | None=None) -> pd.DataFrame:
        all_dates: list[pd.Timestamp] = []
        years = [year] if year else list(range(2014, pd.Timestamp.today().year + 2))
        for y in years:
            all_dates.extend(self._fetch_year(y))
        if not all_dates:
            logger.warning('fns_calendar_empty')
            return pd.DataFrame(columns=['date', 'event_type'])
        df = pd.DataFrame({'date': sorted(set(all_dates)), 'event_type': 'tax_payment'})
        if year:
            df = df[pd.to_datetime(df['date']).dt.year == year]
        logger.info('fns_calendar_loaded', rows=len(df), years=len(years))
        return df

    def _fetch_year(self, year: int) -> list[pd.Timestamp]:
        urls = [f'{self.url}?year={year}', f'https://www.nalog.gov.ru/rn77/calendar/?year={year}', f'http://www.nalog.gov.ru/rn77/calendar/print/?year={year}']
        for url in urls:
            try:
                html = self.client.get_bytes(url).decode('utf-8', errors='replace')
                dates = _coalesce_tax_dates_for_year(year, _parse_fns_calendar_html(html, year))
                if dates:
                    return dates
            except Exception as exc:
                logger.debug('fns_year_fetch_failed', year=year, error=str(exc))
        dates = _enp_business_calendar(year)
        logger.info('fns_calendar_enp_schedule', year=year, count=len(dates))
        return dates

def _coalesce_tax_dates_for_year(year: int, dates: list[pd.Timestamp]) -> list[pd.Timestamp]:
    unique = sorted(set(dates))
    if len(unique) > _MAX_PAYMENT_DATES_PER_YEAR:
        logger.info('fns_calendar_dense_fallback_enp', year=year, parsed=len(unique))
        return _enp_business_calendar(year)
    return unique


def _parse_fns_calendar_html(html: str, year: int) -> list[pd.Timestamp]:
    timestamps = _parse_nalog_calendar_js(html, year)
    if timestamps:
        return timestamps
    timestamps = []
    soup = BeautifulSoup(html, 'lxml')
    for table in soup.find_all('table'):
        for cell in table.find_all(['td', 'th']):
            text = cell.get_text(' ', strip=True)
            for d, m, y in re.findall('(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})', text):
                if int(y) == year and int(m) >= 1 and (int(d) >= 1):
                    timestamps.append(pd.Timestamp(year=int(y), month=int(m), day=int(d)))
    if not timestamps:
        for d, m, y in re.findall('(\\d{1,2})\\.(\\d{1,2})\\.(\\d{4})', soup.get_text(' ')):
            if int(y) == year:
                timestamps.append(pd.Timestamp(year=int(y), month=int(m), day=int(d)))
    return sorted(set(timestamps))

def _parse_nalog_calendar_js(html: str, year: int) -> list[pd.Timestamp]:
    timestamps: list[pd.Timestamp] = []
    for y_s, m_s, d_s in _NALOG_DATE_RE.findall(html):
        if int(y_s) != year:
            continue
        month = int(m_s) + 1
        day = int(d_s)
        if month < 1 or month > 12 or day < 1:
            continue
        try:
            timestamps.append(pd.Timestamp(year=year, month=month, day=day))
        except ValueError:
            continue
    return sorted(set(timestamps))
