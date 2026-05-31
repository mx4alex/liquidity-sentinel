from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.core.mad import mad_score_to_stress, mad_zscore
from ru_liquidity_sentinel.modules.base import BaseModule


class ModuleM3OFZ(BaseModule):
    module_id = 'm3'

    def run(self, ofz: pd.DataFrame, keyrate: pd.DataFrame | None=None, **_: pd.DataFrame) -> pd.DataFrame:
        nedospros = float(self.settings.get('modules', 'm3', 'nedospros_cover_threshold', default=1.2))
        perespros = float(self.settings.get('modules', 'm3', 'perespros_cover_threshold', default=2.0))
        df = ofz.copy()
        if df.empty:
            return pd.DataFrame()
        df = df.sort_values('date').drop_duplicates('date', keep='last').set_index('date')
        if 'cover_ratio' not in df.columns:
            df['cover_ratio'] = df['demand_bn'] / df['offered_bn'].replace(0, pd.NA)
        cover_stress = -mad_zscore(df['cover_ratio'], window_years=self.window_years, min_periods=5)
        df['mad_score_cover'] = mad_score_to_stress(cover_stress)
        if 'weighted_yield' in df.columns:
            wy = pd.to_numeric(df['weighted_yield'], errors='coerce')
            benchmark = wy.shift(1).rolling(window=10, min_periods=3).median()
            if keyrate is not None and (not keyrate.empty):
                kr = keyrate.copy()
                if 'date' in kr.columns:
                    kr = kr.set_index('date')
                rate_col = 'key_rate' if 'key_rate' in kr.columns else kr.columns[-1]
                kr_s = pd.to_numeric(kr[rate_col], errors='coerce').sort_index()
                df['key_rate'] = kr_s.reindex(df.index, method='ffill')
                benchmark = benchmark.fillna(df['key_rate'])
            df['yield_benchmark'] = benchmark
            df['yield_spread'] = wy - benchmark
            df['mad_score_yield_spread'] = mad_score_to_stress(mad_zscore(df['yield_spread'], window_years=self.window_years, min_periods=5))
        else:
            df['mad_score_yield_spread'] = 0.0
        df['flag_nedospros'] = df['cover_ratio'] < nedospros
        df['flag_perespros'] = df['cover_ratio'] > perespros
        df['m3_stress'] = 0.75 * df['mad_score_cover'].fillna(0) + 0.25 * df['mad_score_yield_spread'].fillna(0)
        df.loc[df['flag_nedospros'], 'm3_stress'] *= 1.25
        df['module_id'] = self.module_id
        df.index.name = 'date'
        return df
