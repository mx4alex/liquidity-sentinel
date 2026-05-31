from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import nnls

from ru_liquidity_sentinel.config import get_settings

STRESS_FEATURE_COLS = ('m1_stress', 'm2_stress', 'm3_stress', 'm5_stress')
MODULE_IDS = ('m1', 'm2', 'm3', 'm5')

def liquidity_stress_proxy(gt: pd.DataFrame) -> pd.Series:
    df = gt.copy()
    if 'date' in df.columns:
        df = df.set_index('date')
    col = 'liquidity_balance' if 'liquidity_balance' in df.columns else df.columns[-1]
    balance = pd.to_numeric(df[col], errors='coerce')
    delta = -balance.diff(5)
    trend = balance.rolling(60, min_periods=20).mean()
    gap = (trend - balance).clip(lower=0)
    proxy = delta.fillna(0) + 0.3 * gap.fillna(0)
    proxy = (proxy - proxy.min()) / (proxy.max() - proxy.min() + 1e-09)
    return proxy * 100

def fit_logistic_params(raw_stress: pd.Series, target: pd.Series) -> tuple[float, float]:
    merged = pd.concat([raw_stress.rename('raw'), target.rename('target')], axis=1).dropna()
    if len(merged) < 100:
        return (0.55, 1.5)
    best_k, best_x0, best_corr = (0.55, 1.5, -1.0)
    for k in (0.35, 0.45, 0.55, 0.65, 0.75, 0.9):
        for x0 in (0.5, 1.0, 1.5, 2.0, 2.5):
            pred = 100 / (1 + np.exp(-k * (merged['raw'] - x0)))
            corr = stats.pearsonr(pred, merged['target'])[0]
            if corr > best_corr:
                best_corr, best_k, best_x0 = (corr, k, x0)
    return (best_k, best_x0)

def save_calibration(k: float, x0: float) -> None:
    save_calibration_bundle(k, x0)

def load_calibration() -> tuple[float, float]:
    settings = get_settings()
    path = settings.processed_dir / 'lsi_calibration.json'
    if path.exists():
        data = json.loads(path.read_text(encoding='utf-8'))
        return (float(data.get('k', 0.55)), float(data.get('x0', 1.5)))
    defaults = settings.get('aggregation', 'logistic', default={})
    return (float(defaults.get('k', 0.55)), float(defaults.get('x0', 1.5)))

def _calibration_path() -> Path:
    return get_settings().processed_dir / 'lsi_calibration.json'

def load_calibration_bundle() -> dict[str, object]:
    path = _calibration_path()
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    defaults = get_settings().get('aggregation', 'logistic', default={})
    return {'k': float(defaults.get('k', 0.55)), 'x0': float(defaults.get('x0', 1.5))}

def save_calibration_bundle(k: float, x0: float, *, module_weights: dict[str, float] | None=None) -> None:
    path = _calibration_path()
    data: dict[str, object] = {'k': k, 'x0': x0}
    if module_weights:
        data['module_weights'] = module_weights
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')

def load_module_weights() -> dict[str, float] | None:
    data = load_calibration_bundle()
    raw = data.get('module_weights')
    if not isinstance(raw, dict):
        return None
    return {str(k): float(v) for k, v in raw.items()}

def fit_module_weights(features: pd.DataFrame, target: pd.Series, train_mask: pd.Series) -> dict[str, float]:
    cols = [c for c in STRESS_FEATURE_COLS if c in features.columns]
    if not cols:
        return dict(zip(MODULE_IDS, [0.25, 0.25, 0.25, 0.25], strict=True))
    X = features.loc[train_mask, cols].clip(lower=0).fillna(0).to_numpy()
    y = target.reindex(features.index).loc[train_mask].fillna(0).to_numpy()
    if len(y) < 50 or y.std() < 1e-09:
        return {}
    w, _ = nnls(X, y)
    if w.sum() <= 0:
        return {}
    w = w / w.sum()
    mapping = dict(zip(MODULE_IDS, w, strict=True))
    return mapping

def build_training_mask(index: pd.DatetimeIndex, *, holdout_start: str, episodes: list[dict[str, str]]) -> pd.Series:
    mask = index >= pd.Timestamp(holdout_start)
    for ep in episodes:
        mask &= ~((index >= pd.Timestamp(ep['start'])) & (index <= pd.Timestamp(ep['end'])))
    return pd.Series(mask, index=index)
