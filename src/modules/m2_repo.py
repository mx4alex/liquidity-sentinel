from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.core.mad import mad_score_to_stress, mad_zscore
from ru_liquidity_sentinel.modules.base import BaseModule


class ModuleM2Repo(BaseModule):
    module_id = 'm2'

    def run(self, repo: pd.DataFrame, keyrate: pd.DataFrame, **_: pd.DataFrame) -> pd.DataFrame:
        focus_days = int(self.settings.get('modules', 'm2', 'focus_term_days', default=7))
        demand_thr = float(self.settings.get('modules', 'm2', 'demand_cover_threshold', default=2.0))
        df = repo.copy()
        if df.empty:
            return pd.DataFrame()
        if 'term_days' in df.columns:
            df = df[df['term_days'] == focus_days]
        df = df.sort_values('date').drop_duplicates('date', keep='last')
        kr = keyrate.set_index('date')['key_rate'].sort_index()
        df['key_rate'] = kr.reindex(pd.to_datetime(df['date']), method='ffill').values
        if 'cutoff_rate' in df.columns:
            df['rate_spread'] = df['cutoff_rate'] - df['key_rate']
        if 'cover_ratio' not in df.columns and {'demand_bn', 'allocated_bn'} <= set(df.columns):
            df['cover_ratio'] = df['demand_bn'] / df['allocated_bn'].replace(0, pd.NA)
        df = df.set_index('date')
        df['mad_score_cover'] = mad_score_to_stress(mad_zscore(df['cover_ratio'], window_years=self.window_years, min_periods=10))
        if 'rate_spread' in df.columns:
            df['mad_score_rate_spread'] = mad_score_to_stress(mad_zscore(df['rate_spread'], window_years=self.window_years, min_periods=10))
        else:
            df['mad_score_rate_spread'] = 0.0
        df['flag_demand'] = df['cover_ratio'] > demand_thr
        df['m2_stress'] = 0.6 * df['mad_score_cover'].fillna(0) + 0.4 * df['mad_score_rate_spread'].fillna(0)
        df.loc[df['flag_demand'], 'm2_stress'] *= 1.2
        df['module_id'] = self.module_id
        df.index.name = 'date'
        return df
