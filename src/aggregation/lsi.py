from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

from ru_liquidity_sentinel.aggregation.calibration import (
    load_calibration_bundle,
    save_calibration_bundle,
)
from ru_liquidity_sentinel.aggregation.overlap import (
    apply_overlap_adjustment,
    compute_overlap_report,
)
from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.core.dates import today_ts
from ru_liquidity_sentinel.core.types import LSIStatus
from ru_liquidity_sentinel.logging import get_logger

logger = get_logger(__name__)
STRESS_COL_MAP = {'m1': 'm1_stress', 'm2': 'm2_stress', 'm3': 'm3_stress', 'm5': 'm5_stress'}

class LSIAggregator:

    def __init__(self) -> None:
        self.settings = get_settings()
        self.method = self.settings.get('aggregation', 'method', default='weighted_interpretable')
        self.green_max = float(self.settings.get('aggregation', 'lsi_green_max', default=40))
        self.yellow_max = float(self.settings.get('aggregation', 'lsi_yellow_max', default=70))
        self.base_weights: dict[str, float] = dict(self.settings.get('aggregation', 'weights', default={}))
        self._gb_model: GradientBoostingRegressor | None = None
        self._feature_names: list[str] = []
        bundle = load_calibration_bundle()
        self._logistic_k = float(bundle.get('k', 0.55))
        self._logistic_x0 = float(bundle.get('x0', 1.5))

    def build_feature_matrix(self, m1: pd.DataFrame, m2: pd.DataFrame, m3: pd.DataFrame, m4: pd.DataFrame, m5: pd.DataFrame) -> pd.DataFrame:
        frames = [f for f in (m1, m2, m3, m4, m5) if f is not None and (not f.empty)]
        if not frames:
            return pd.DataFrame()
        idx = frames[0].index
        for f in frames[1:]:
            idx = idx.union(f.index)
        idx = idx.sort_values()
        idx = idx[idx <= today_ts()]
        out = pd.DataFrame(index=idx)
        event_fill_days = int(self.settings.get('aggregation', 'event_fill_days', default=21))
        for mod, frame, col in [('m1', m1, 'm1_stress'), ('m2', m2, 'm2_stress'), ('m3', m3, 'm3_stress'), ('m5', m5, 'm5_stress')]:
            if frame is not None and (not frame.empty) and (col in frame.columns):
                series = frame[col].reindex(idx)
                if mod in ('m2', 'm3'):
                    series = series.ffill(limit=event_fill_days)
                else:
                    series = series.ffill()
                out[col] = series
        if m4 is not None and (not m4.empty):
            for c in ('tax_week_flag', 'seasonal_factor', 'end_of_month_flag', 'end_of_quarter_flag'):
                if c in m4.columns:
                    out[c] = m4[c].reindex(idx).ffill()
        if m4 is not None and (not m4.empty) and ('seasonal_factor' in m4.columns):
            out['seasonal_factor'] = out['seasonal_factor'].replace(0, 1.0)
        out = out.ffill().fillna(0)
        return out

    def compute_lsi(self, features: pd.DataFrame, ground_truth: pd.DataFrame | None=None) -> pd.DataFrame:
        if features.empty:
            return pd.DataFrame()
        stress_cols = [c for c in features.columns if c.endswith('_stress')]
        report = compute_overlap_report(features, stress_cols)
        logger.info('overlap_report', inflation_ratio=report.inflation_ratio, method=report.method)
        if self.method == 'gradient_boosting' and ground_truth is not None and (not ground_truth.empty):
            return self._compute_gb_lsi(features, ground_truth, report)
        self._calibrate_scale(features)
        return self._compute_weighted_lsi(features, report)

    def _calibrate_scale(self, features: pd.DataFrame) -> None:
        raw = self._raw_stress_series(features, apply_overlap=False)
        raw = raw[raw.notna()]
        if raw.empty:
            return
        low_pct = float(self.settings.get('aggregation', 'scale_low_pct', default=0.50))
        high_pct = float(self.settings.get('aggregation', 'scale_high_pct', default=0.97))
        low_lsi = float(self.settings.get('aggregation', 'scale_low_lsi', default=20.0))
        high_lsi = float(self.settings.get('aggregation', 'scale_high_lsi', default=80.0))
        q_lo, q_hi = float(raw.quantile(low_pct)), float(raw.quantile(high_pct))
        if q_hi <= q_lo:
            return
        logit_lo = float(np.log(low_lsi / (100 - low_lsi)))
        logit_hi = float(np.log(high_lsi / (100 - high_lsi)))
        k = (logit_hi - logit_lo) / (q_hi - q_lo)
        x0 = q_lo - logit_lo / k
        self._logistic_k, self._logistic_x0 = k, x0
        save_calibration_bundle(k, x0, module_weights=self._normalized_weights())
        logger.info('lsi_scale_calibrated', k=round(k, 4), x0=round(x0, 4),
                    q_lo=round(q_lo, 3), q_hi=round(q_hi, 3))

    def _normalized_weights(self, override: dict[str, float] | None=None) -> dict[str, float]:
        src = override if override else self.base_weights
        w = {k.replace('_repo_rate', '').replace('_context', ''): v for k, v in src.items()}
        mapping = {'m1': w.get('m1', 0.25), 'm2': w.get('m2', 0.30), 'm3': w.get('m3', 0.25), 'm5': w.get('m5', 0.20)}
        total = sum(mapping.values())
        if total <= 0:
            mapping = {'m1': 0.25, 'm2': 0.30, 'm3': 0.25, 'm5': 0.20}
            total = sum(mapping.values())
        return {k: v / total for k, v in mapping.items()}

    def _raw_stress_series(self, features: pd.DataFrame, apply_overlap: bool=True, weights: dict[str, float] | None=None) -> pd.Series:
        w_default = self._normalized_weights(weights)
        raw_values: list[float] = []
        for _date, row in features.iterrows():
            tax_week = bool(row.get('tax_week_flag', 0))
            w = apply_overlap_adjustment(w_default, tax_week) if apply_overlap else w_default
            raw_sum = sum((max(0.0, float(row.get(col, 0))) * w.get(mod, 0) for mod, col in STRESS_COL_MAP.items()))
            raw_values.append(raw_sum)
        return pd.Series(raw_values, index=features.index)

    def _compute_weighted_lsi(self, features: pd.DataFrame, report: object) -> pd.DataFrame:
        rows = []
        w_default = self._normalized_weights()
        k, x0 = (self._logistic_k, self._logistic_x0)
        pct_weight = float(self.settings.get('aggregation', 'percentile_weight', default=0.35))
        win = int(self.settings.get('aggregation', 'percentile_window_days', default=756))
        raw_hist = self._raw_stress_series(features, apply_overlap=False).to_numpy()
        for i, (date, row) in enumerate(features.iterrows()):
            tax_week = bool(row.get('tax_week_flag', 0))
            w = apply_overlap_adjustment(w_default, tax_week)
            components: dict[str, float] = {}
            raw_sum = 0.0
            for mod, col in STRESS_COL_MAP.items():
                contrib = max(0.0, float(row.get(col, 0))) * w.get(mod, 0)
                components[mod] = contrib
                raw_sum += contrib
            logistic_lsi = 100 * (1 / (1 + np.exp(-k * (raw_sum - x0))))
            if pct_weight > 0:
                window = raw_hist[max(0, i - win + 1) : i + 1]
                pct_lsi = (
                    float((window <= raw_sum).mean() * 100) if len(window) >= 30 else logistic_lsi
                )
            else:
                pct_lsi = logistic_lsi
            base_lsi = (1 - pct_weight) * logistic_lsi + pct_weight * pct_lsi
            seasonal = float(row.get('seasonal_factor', 1.0) or 1.0)
            if seasonal <= 0:
                seasonal = 1.0
            lsi = float(np.clip(base_lsi * seasonal, 0, 100))
            status = LSIStatus.from_value(lsi, self.green_max, self.yellow_max)
            rows.append({'date': date, 'lsi': lsi, 'status': status.value, 'status_ru': status.label_ru, 'seasonal_factor': seasonal, 'overlap_adjusted': tax_week, 'contrib_m1': components.get('m1', 0), 'contrib_m2': components.get('m2', 0), 'contrib_m3': components.get('m3', 0), 'contrib_m5': components.get('m5', 0), 'raw_stress_sum': raw_sum})
        result = pd.DataFrame(rows).set_index('date')
        self._attach_flags(result, features)
        return result

    def _compute_gb_lsi(self, features: pd.DataFrame, ground_truth: pd.DataFrame, report: object) -> pd.DataFrame:
        gt = ground_truth.set_index('date').sort_index()
        y_col = 'liquidity_balance' if 'liquidity_balance' in gt.columns else gt.columns[-1]
        y = -gt[y_col].diff().rolling(5).mean()
        X = features.reindex(y.index).dropna()
        y = y.reindex(X.index).dropna()
        common = X.index.intersection(y.index)
        X, y = (X.loc[common], y.loc[common])
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        self._feature_names = list(X.columns)
        model = GradientBoostingRegressor(n_estimators=100, max_depth=3, random_state=42)
        model.fit(Xs, y)
        self._gb_model = model
        try:
            import shap
            explainer = shap.Explainer(model, Xs)
            shap_values = explainer(Xs)
            shap_df = pd.DataFrame(shap_values.values, index=common, columns=X.columns)
        except Exception:
            shap_df = pd.DataFrame(0.0, index=common, columns=X.columns)
        pred = model.predict(Xs)
        lsi_series = 100 * (pred - pred.min()) / (pred.max() - pred.min() + 1e-09)
        result = pd.DataFrame({'lsi': lsi_series, 'date': common}).set_index('date')
        for col in X.columns:
            if col in shap_df.columns:
                result[f'contrib_{col}'] = shap_df[col].values
        result['status'] = result['lsi'].apply(lambda v: LSIStatus.from_value(v, self.green_max, self.yellow_max).value)
        return result

    def _attach_flags(self, result: pd.DataFrame, features: pd.DataFrame) -> None:
        flag_cols = [c for c in features.columns if c.startswith('flag_') or c.endswith('_flag')]
        if not flag_cols:
            return
        result['active_flags'] = [','.join(c for c in flag_cols if features.loc[d].get(c) in (True, 1)) for d in result.index]

    def sensitivity_analysis(self, features: pd.DataFrame, pct: float=0.2) -> pd.DataFrame:
        base = self.compute_lsi(features)
        if base.empty:
            return pd.DataFrame()
        last = base.iloc[-1]['lsi']
        rows = [{'scenario': 'base', 'lsi': last}]
        w = self._normalized_weights()
        for direction in (-1, 1):
            w_adj = {k: v * (1 + direction * pct) for k, v in w.items()}
            total = sum(w_adj.values())
            w_adj = {k: v / total for k, v in w_adj.items()}
            self.base_weights = {f'{k}': v for k, v in w_adj.items()}
            adj = self.compute_lsi(features)
            if not adj.empty:
                rows.append({'scenario': f"weights_{('plus' if direction > 0 else 'minus')}_{int(pct * 100)}pct", 'lsi': adj.iloc[-1]['lsi']})
        self.base_weights = dict(self.settings.get('aggregation', 'weights', default={}))
        return pd.DataFrame(rows)

    def save_overlap_report(self, features: pd.DataFrame, path: Path | None=None) -> Path:
        stress_cols = [c for c in features.columns if c.endswith('_stress')]
        report = compute_overlap_report(features, stress_cols)
        out = path or self.settings.processed_dir / 'overlap_report.json'
        out.write_text(json.dumps(report.__dict__, indent=2, ensure_ascii=False), encoding='utf-8')
        return out
