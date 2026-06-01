from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.core.mad import mad_score_to_stress, mad_zscore
from ru_liquidity_sentinel.modules.base import BaseModule


class ModuleM1Reserves(BaseModule):
    module_id = 'm1'

    def run(self, reserves: pd.DataFrame, ruonia: pd.DataFrame, **_: pd.DataFrame) -> pd.DataFrame:
        end_days = int(self.settings.get('modules', 'm1', 'end_of_period_days', default=5))
        r = reserves.copy()
        if r.empty or 'period_start' not in r.columns:
            return pd.DataFrame()
        r = r.dropna(subset=['period_start']).sort_values('period_start').reset_index(drop=True)
        if 'reserve_spread' not in r.columns and {'actual_balances', 'required_averaging'} <= set(r.columns):
            r['reserve_spread'] = r['actual_balances'] - r['required_averaging']
        by_len = pd.Series(pd.NaT, index=r.index, dtype='datetime64[ns]')
        if 'period_days' in r.columns and r['period_days'].notna().any():
            by_len = r['period_start'] + pd.to_timedelta(r['period_days'].fillna(30) - 1, unit='D')
        by_next = r['period_start'].shift(-1) - pd.Timedelta(days=1)
        r['period_end'] = by_next.fillna(by_len).fillna(r['period_start'] + pd.offsets.MonthEnd(0))
        r['mad_score_spread'] = mad_score_to_stress(mad_zscore(r['reserve_spread'], window_years=self.window_years, min_periods=5))
        period_cols = ['period_end', 'mad_score_spread', 'reserve_spread']
        for opt in ('actual_balances', 'required_averaging', 'required_accounts'):
            if opt in r.columns:
                period_cols.append(opt)
        period = r.set_index('period_start')[period_cols].sort_index()
        daily_idx = pd.date_range(r['period_start'].min(), pd.Timestamp.today().normalize(), freq='D')
        daily = period.reindex(daily_idx, method='ffill')
        daily.index.name = 'date'
        days_to_end = (daily['period_end'] - daily.index.to_series()).dt.days
        daily['days_to_end'] = days_to_end
        daily['flag_end_of_period'] = (days_to_end >= 0) & (days_to_end < end_days)
        ru = ruonia.copy()
        if 'date' in ru.columns:
            ru = ru.set_index('date')
        ru_s = pd.to_numeric(ru['ruonia'], errors='coerce').sort_index()
        daily['ruonia'] = ru_s.reindex(daily.index, method='ffill')
        daily['mad_score_ruonia'] = mad_score_to_stress(mad_zscore(daily['ruonia'], window_years=self.window_years, min_periods=self.min_periods))
        end_mask = daily['flag_end_of_period'].astype(bool)
        daily['flag_overfulfillment'] = end_mask & (daily['mad_score_spread'].fillna(0) > 1.0)
        daily['m1_stress'] = daily['mad_score_spread'].fillna(0) + 0.5 * daily['mad_score_ruonia'].fillna(0)
        daily.loc[daily['flag_overfulfillment'], 'm1_stress'] *= 1.15
        daily['module_id'] = self.module_id
        return daily.dropna(how='all')
