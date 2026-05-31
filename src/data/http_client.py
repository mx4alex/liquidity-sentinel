from __future__ import annotations

import hashlib
from pathlib import Path

import certifi
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from ru_liquidity_sentinel.config import get_settings


class DataHttpClient:

    def __init__(self, cache_dir: Path | None=None, timeout: float=60.0, verify: bool | str=True) -> None:
        settings = get_settings()
        self.cache_dir = cache_dir or settings.raw_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        ssl_verify = settings.get('http', 'verify_ssl', default=True)
        if verify is True and ssl_verify is not False:
            verify_param: bool | str = certifi.where()
        else:
            verify_param = verify
        self._client = httpx.Client(timeout=timeout, follow_redirects=True, verify=verify_param, headers={'User-Agent': 'RU-Liquidity-Sentinel/0.1 (+https://github.com/psb; research/educational)'})

    def _cache_path(self, url: str, suffix: str='') -> Path:
        key = hashlib.sha256(url.encode()).hexdigest()[:16]
        return self.cache_dir / f'{key}{suffix}'

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def get_bytes(self, url: str, use_cache: bool=True) -> bytes:
        path = self._cache_path(url, suffix='.bin')
        if use_cache and path.exists():
            return path.read_bytes()
        resp = self._client.get(url)
        resp.raise_for_status()
        path.write_bytes(resp.content)
        return resp.content

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=30))
    def post_form(self, url: str, data: dict[str, str], use_cache: bool=True) -> str:
        import hashlib
        key = hashlib.sha256((url + str(sorted(data.items()))).encode()).hexdigest()[:16]
        path = self.cache_dir / f'post_{key}.html'
        if use_cache and path.exists():
            return path.read_text(encoding='utf-8', errors='replace')
        resp = self._client.post(url, data=data)
        resp.raise_for_status()
        path.write_text(resp.text, encoding='utf-8')
        return resp.text

    def download(self, url: str, filename: str, use_cache: bool=True) -> Path:
        dest = self.cache_dir / filename
        if use_cache and dest.exists():
            return dest
        content = self.get_bytes(url, use_cache=False)
        dest.write_bytes(content)
        return dest

    def close(self) -> None:
        self._client.close()
