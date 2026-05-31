from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import AliasChoices, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / 'config' / 'settings.yaml'

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix='SENTINEL_', env_file='.env', extra='ignore')
    data_dir: Path = Field(default=PROJECT_ROOT / 'data')
    config_path: Path = Field(default=DEFAULT_CONFIG_PATH)
    llm_enabled: bool = False
    yandex_api_key: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            'YANDEX_API_KEY',
            'YANDEX_CLOUD_API_KEY',
            'yandex_api_key',
        ),
    )
    yandex_folder_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices(
            'YANDEX_FOLDER_ID',
            'YANDEX_CLOUD_FOLDER',
            'yandex_folder_id',
        ),
    )
    _yaml: dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        if self.config_path.exists():
            with self.config_path.open(encoding='utf-8') as f:
                object.__setattr__(self, '_yaml', yaml.safe_load(f) or {})
        yaml_llm = self._yaml.get('llm') or {}
        if not self.yandex_api_key:
            object.__setattr__(
                self,
                'yandex_api_key',
                os.environ.get('YANDEX_API_KEY') or os.environ.get('YANDEX_CLOUD_API_KEY'),
            )
        if not self.yandex_folder_id:
            fid = (
                yaml_llm.get('folder_id')
                or os.environ.get('YANDEX_FOLDER_ID')
                or os.environ.get('YANDEX_CLOUD_FOLDER')
            )
            if fid:
                object.__setattr__(self, 'yandex_folder_id', str(fid))
        if yaml_llm.get('enabled') and (not self.llm_enabled):
            object.__setattr__(self, 'llm_enabled', True)

    @property
    def llm_active(self) -> bool:
        yaml_on = bool((self._yaml.get('llm') or {}).get('enabled'))
        return bool(self.llm_enabled or yaml_on)

    @property
    def llm_ready(self) -> bool:
        return bool(self.yandex_api_key and self.yandex_folder_id)

    @property
    def llm_model(self) -> str:
        env_model = os.environ.get('YANDEX_CLOUD_MODEL') or os.environ.get('YANDEX_MODEL')
        if env_model:
            return str(env_model)
        return str(self.get('llm', 'model', default='yandexgpt-lite'))

    @property
    def raw_dir(self) -> Path:
        rel = self._yaml.get('data', {}).get('raw_dir', 'data/raw')
        return self.data_dir.parent / rel if not Path(rel).is_absolute() else Path(rel)

    @property
    def processed_dir(self) -> Path:
        rel = self._yaml.get('data', {}).get('processed_dir', 'data/processed')
        p = self.data_dir.parent / rel if not Path(rel).is_absolute() else Path(rel)
        p.mkdir(parents=True, exist_ok=True)
        return p

    def get(self, *keys: str, default: Any=None) -> Any:
        node: Any = self._yaml
        for k in keys:
            if not isinstance(node, dict):
                return default
            node = node.get(k)
            if node is None:
                return default
        return node

@lru_cache
def get_settings() -> Settings:
    return Settings()
