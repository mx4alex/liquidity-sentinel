from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from ru_liquidity_sentinel.config import get_settings


class BaseModule(ABC):
    module_id: str = 'base'

    def __init__(self) -> None:
        self.settings = get_settings()
        self.window_years = int(self.settings.get('normalization', 'mad_window_years', default=3))
        self.min_periods = int(self.settings.get('normalization', 'mad_min_periods', default=30))

    @abstractmethod
    def run(self, **data: pd.DataFrame) -> pd.DataFrame:
        pass

    def save_signals(self, df: pd.DataFrame, name: str | None=None) -> pd.DataFrame:
        path = self.settings.processed_dir / f'{name or self.module_id}_signals.parquet'
        df.to_parquet(path)
        return df
