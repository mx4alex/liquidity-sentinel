from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.core.mad import mad_score_to_stress, mad_zscore
from ru_liquidity_sentinel.modules.base import BaseModule


def _monthly_key(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    return index.to_period('M').to_timestamp()

class ModuleM5Treasury(BaseModule):
    module_id = 'm5'

    def run(self, sors: pd.DataFrame | None=None, roskazna: pd.DataFrame | None=None, **_: pd.DataFrame) -> pd.DataFrame:
        drain_thr = float(self.settings.get('modules', 'm5', 'budget_drain_weekly_bn', default=400.0))
        drain_z = float(self.settings.get('modules', 'm5', 'budget_drain_zscore', default=2.0))
        legs: list[pd.DataFrame] = []
        cbr_leg = self._budget_leg(sors)
        if cbr_leg is not None:
            legs.append(cbr_leg)
        rk_leg = self._roskazna_leg(roskazna)
        if rk_leg is not None:
            legs.append(rk_leg)
        if not legs:
            return pd.DataFrame()
        monthly = pd.concat(legs, axis=1).sort_index()
        for col in ('mad_score_cbr', 'mad_score_roskazna'):
            if col not in monthly.columns:
                monthly[col] = 0.0
            monthly[col] = monthly[col].fillna(0.0)
        has_cbr = 'budget_balance_bn' in monthly.columns
        has_rk = 'placement_bn' in monthly.columns
        if has_cbr and has_rk:
            monthly['m5_stress'] = 0.6 * monthly['mad_score_cbr'] + 0.4 * monthly['mad_score_roskazna']
        elif has_cbr:
            monthly['m5_stress'] = monthly['mad_score_cbr']
        else:
            monthly['m5_stress'] = monthly['mad_score_roskazna']
        outflow_bn = -monthly.get('budget_delta', pd.Series(0.0, index=monthly.index)).fillna(0.0)
        monthly['flag_budget_drain'] = (monthly['mad_score_cbr'] >= drain_z) | (monthly['mad_score_roskazna'] >= drain_z) | (outflow_bn > drain_thr)
        monthly.loc[monthly['flag_budget_drain'], 'm5_stress'] *= 1.2
        daily_idx = pd.date_range(monthly.index.min(), pd.Timestamp.today().normalize(), freq='D')
        daily = monthly.reindex(daily_idx, method='ffill')
        daily.index.name = 'date'
        daily['module_id'] = self.module_id
        return daily.dropna(how='all')

    def _budget_leg(self, sors: pd.DataFrame | None) -> pd.DataFrame | None:
        if sors is None or sors.empty:
            return None
        s = sors.copy()
        if 'date' in s.columns:
            s = s.set_index('date')
        col = 'attracted_bn' if 'attracted_bn' in s.columns else s.columns[0]
        bal = pd.to_numeric(s[col], errors='coerce').dropna()
        if bal.empty:
            return None
        bal.index = _monthly_key(pd.DatetimeIndex(bal.index))
        bal = bal[~bal.index.duplicated(keep='last')].sort_index()
        leg = pd.DataFrame(index=bal.index)
        leg['budget_balance_bn'] = bal
        leg['budget_delta'] = bal.diff()
        leg['mad_score_cbr'] = mad_score_to_stress(mad_zscore(-leg['budget_delta'], window_years=self.window_years, min_periods=12))
        return leg

    def _roskazna_leg(self, roskazna: pd.DataFrame | None) -> pd.DataFrame | None:
        if roskazna is None or roskazna.empty:
            return None
        rk = roskazna.copy()
        if 'date' in rk.columns:
            rk = rk.set_index('date')
        pcol = 'placement_bn' if 'placement_bn' in rk.columns else rk.columns[0]
        pl = pd.to_numeric(rk[pcol], errors='coerce').dropna()
        if pl.empty:
            return None
        bank_count = None
        if 'bank_count' in rk.columns:
            bank_count = pd.to_numeric(rk['bank_count'], errors='coerce')
            bank_count.index = _monthly_key(pd.DatetimeIndex(rk.index))
        pl.index = _monthly_key(pd.DatetimeIndex(pl.index))
        pl = pl.groupby(level=0).sum().sort_index()
        leg = pd.DataFrame(index=pl.index)
        leg['placement_bn'] = pl
        leg['placement_delta'] = pl.diff()
        leg['mad_score_roskazna'] = mad_score_to_stress(mad_zscore(-leg['placement_delta'], window_years=self.window_years, min_periods=12))
        if bank_count is not None:
            leg['bank_count'] = bank_count[~bank_count.index.duplicated(keep='last')]
        return leg
