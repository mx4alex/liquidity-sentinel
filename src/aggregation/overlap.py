from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ru_liquidity_sentinel.config import get_settings


@dataclass
class OverlapReport:
    tax_week_mean_stress: float
    normal_mean_stress: float
    inflation_ratio: float
    method: str
    downweight_applied: float

def compute_overlap_report(features: pd.DataFrame, stress_cols: list[str]) -> OverlapReport:
    settings = get_settings()
    method = 'conditional_downweight'
    tax = features['tax_week_flag'] == 1 if 'tax_week_flag' in features.columns else pd.Series(False, index=features.index)
    combined = features[stress_cols].mean(axis=1)
    tax_mean = float(combined[tax].mean()) if tax.any() else 0.0
    normal_mean = float(combined[~tax].mean()) if (~tax).any() else float(combined.mean())
    ratio = tax_mean / normal_mean - 1.0 if normal_mean > 0 else 0.0
    downweight = float(settings.get('aggregation', 'overlap', 'tax_week_downweight', default=0.65))
    return OverlapReport(tax_week_mean_stress=tax_mean, normal_mean_stress=normal_mean, inflation_ratio=ratio, method=method, downweight_applied=downweight)

def apply_overlap_adjustment(weights: dict[str, float], tax_week: bool) -> dict[str, float]:
    settings = get_settings()
    if not tax_week:
        return weights.copy()
    down = float(settings.get('aggregation', 'overlap', 'tax_week_downweight', default=0.65))
    to_reduce = settings.get('aggregation', 'overlap', 'modules_to_downweight', default=['m1', 'm2', 'm5'])
    w = weights.copy()
    for key in to_reduce:
        if key in w:
            w[key] *= down
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w
