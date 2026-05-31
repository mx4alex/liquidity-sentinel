from __future__ import annotations

import re
import time

import httpx

from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
CDX_API = 'https://web.archive.org/cdx/search/cdx'
WAYBACK_TIMEOUT = 120.0

def find_minfin_ofz_snapshot(year: int) -> tuple[str, str] | None:
    slug = f'rezultaty_provedennykh_auktsionov_po_razmeshcheniyu_gosudarstvennykh_tsennykh_bumag_v_{year}_godu'
    params = {'url': 'minfin.gov.ru/ru/document*', 'from': f'{year}0101', 'to': f'{year}1231', 'output': 'json', 'limit': 2000, 'filter': 'statuscode:200'}
    data: list | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=WAYBACK_TIMEOUT) as client:
                resp = client.get(CDX_API, params=params)
                resp.raise_for_status()
                data = resp.json()
                break
        except Exception as exc:
            logger.debug('wayback_cdx_retry', year=year, attempt=attempt + 1, error=str(exc))
            if attempt == 2:
                logger.warning('wayback_cdx_failed', year=year, error=str(exc))
                return None
            time.sleep(2 * (attempt + 1))
    if data is None:
        return None
    if not isinstance(data, list) or len(data) < 2:
        return None
    matches = [row for row in data[1:] if slug in row[2].lower()]
    if not matches:
        matches = [row for row in data[1:] if 'auktsionov' in row[2].lower() and str(year) in row[2]]
    if not matches:
        matches = _cdx_search_broad(year)
    if not matches:
        return None
    row = sorted(matches, key=lambda x: x[1])[-1]
    return (str(row[1]), str(row[2]))

def _cdx_search_broad(year: int) -> list[list[str]]:
    params = {'url': 'minfin.gov.ru/*', 'matchType': 'prefix', 'from': f'{year}0101', 'to': f'{year}1231', 'output': 'json', 'limit': 5000, 'filter': 'statuscode:200'}
    for attempt in range(2):
        try:
            with httpx.Client(timeout=WAYBACK_TIMEOUT) as client:
                resp = client.get(CDX_API, params=params)
                resp.raise_for_status()
                data = resp.json()
            if not isinstance(data, list) or len(data) < 2:
                return []
            return [row for row in data[1:] if 'auktsionov' in row[2].lower() and 'tsennykh_bumag' in row[2].lower() and (str(year) in row[2])]
        except Exception as exc:
            logger.debug('wayback_cdx_broad_retry', year=year, error=str(exc))
            time.sleep(3 * (attempt + 1))
    return []

def fetch_wayback_html(timestamp: str, original_url: str) -> str:
    url = f'https://web.archive.org/web/{timestamp}/{original_url}'
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            with httpx.Client(timeout=WAYBACK_TIMEOUT, follow_redirects=True) as client:
                resp = client.get(url)
                resp.raise_for_status()
                return resp.text
        except Exception as exc:
            last_exc = exc
            logger.debug('wayback_html_retry', attempt=attempt + 1, error=str(exc))
            time.sleep(2 * (attempt + 1))
    assert last_exc is not None
    raise last_exc

def extract_xlsx_links(html: str, year: int) -> list[str]:
    links = re.findall('href="(/common/upload/library/[^"]+\\.xlsx)"', html)
    return [ln for ln in dict.fromkeys(links) if f'/{year}/' in ln or f'_{year}_' in ln]

def fetch_wayback_xlsx(timestamp: str, path: str) -> bytes | None:
    url = f'https://web.archive.org/web/{timestamp}https://minfin.gov.ru{path}'
    try:
        with httpx.Client(timeout=WAYBACK_TIMEOUT, follow_redirects=True) as client:
            resp = client.get(url)
            if resp.status_code == 200 and len(resp.content) > 500:
                return resp.content
    except Exception as exc:
        logger.debug('wayback_xlsx_failed', path=path, error=str(exc))
    return None
