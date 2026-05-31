from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.core.dates import today_ts
from ru_liquidity_sentinel.modules.base import BaseModule

_FLAG_COLS = ("tax_week_flag", "end_of_month_flag", "end_of_quarter_flag")


class ModuleM4TaxSeasonality(BaseModule):
    module_id = "m4"

    def run(
        self,
        tax_calendar: pd.DataFrame,
        m1: pd.DataFrame | None = None,
        m2: pd.DataFrame | None = None,
        m5: pd.DataFrame | None = None,
        **_: pd.DataFrame,
    ) -> pd.DataFrame:
        before = int(self.settings.get("modules", "m4", "tax_week_before_days", default=5))
        after = int(self.settings.get("modules", "m4", "tax_week_after_days", default=5))
        sf_min = float(self.settings.get("modules", "m4", "seasonal_factor_min", default=1.0))
        sf_max = float(self.settings.get("modules", "m4", "seasonal_factor_max", default=1.4))

        tax_dates = pd.to_datetime(tax_calendar["date"]).drop_duplicates()
        idx = pd.date_range(pd.Timestamp("2014-01-01"), today_ts(), freq="D")
        cal = pd.DataFrame(index=idx)
        cal["tax_week_flag"] = 0
        for td in tax_dates:
            mask = (cal.index >= td - pd.Timedelta(days=before)) & (
                cal.index <= td + pd.Timedelta(days=after)
            )
            cal.loc[mask, "tax_week_flag"] = 1
        cal["end_of_month_flag"] = (cal.index + pd.offsets.MonthEnd(0) == cal.index).astype(int)
        cal["end_of_quarter_flag"] = (cal.index + pd.offsets.QuarterEnd(0) == cal.index).astype(int)

        combined = self._combined_stress(m1, m2, m5, cal)
        cal["seasonal_factor"] = self._seasonal_factor(combined, cal, sf_min, sf_max)
        cal["overlap_context"] = self._overlap_intensity(combined, cal)
        cal["module_id"] = self.module_id
        cal.index.name = "date"
        return cal

    def _combined_stress(
        self,
        m1: pd.DataFrame | None,
        m2: pd.DataFrame | None,
        m5: pd.DataFrame | None,
        cal: pd.DataFrame,
    ) -> pd.Series | None:
        stresses: list[pd.Series] = []
        for frame, col in ((m1, "m1_stress"), (m2, "m2_stress"), (m5, "m5_stress")):
            if frame is not None and not frame.empty and col in frame.columns:
                stresses.append(frame[col])
        if not stresses:
            return None
        combined = pd.concat(stresses, axis=1).mean(axis=1).reindex(cal.index)
        return combined.clip(lower=0.0)

    def _seasonal_factor(
        self,
        combined: pd.Series | None,
        cal: pd.DataFrame,
        sf_min: float,
        sf_max: float,
    ) -> pd.Series:
        sf = pd.Series(sf_min, index=cal.index)
        if combined is None or combined.dropna().empty:
            sf.loc[cal["tax_week_flag"] == 1] = min(sf_min + 0.25, sf_max)
            sf.loc[cal["end_of_quarter_flag"] == 1] = sf.loc[
                cal["end_of_quarter_flag"] == 1
            ].clip(lower=min(sf_min + 0.35, sf_max))
            return sf.clip(sf_min, sf_max)

        normal_mask = (cal[list(_FLAG_COLS)] == 0).all(axis=1)
        base = float(combined[normal_mask].mean())
        if not base or base < 1e-3:
            sf.loc[cal["tax_week_flag"] == 1] = min(sf_min + 0.25, sf_max)
            return sf.clip(sf_min, sf_max)

        for flag in _FLAG_COLS:
            mask = cal[flag] == 1
            if mask.any() and combined[mask].notna().any():
                ratio = float(combined[mask].mean() / base)
                factor = min(max(ratio, sf_min), sf_max)
                sf.loc[mask] = sf.loc[mask].clip(lower=factor)
        return sf.clip(sf_min, sf_max)

    def _overlap_intensity(self, combined: pd.Series | None, cal: pd.DataFrame) -> pd.Series:
        if combined is None:
            return pd.Series(0.0, index=cal.index)
        tax = cal["tax_week_flag"] == 1
        normal_mean = combined[~tax].mean()
        tax_mean = combined[tax].mean()
        ratio = (tax_mean / normal_mean - 1.0) if normal_mean and normal_mean > 0 else 0.0
        return pd.Series(float(max(ratio, 0.0)), index=cal.index)
