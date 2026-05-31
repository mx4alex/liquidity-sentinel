from __future__ import annotations

import json

import pandas as pd
from scipy import stats

from ru_liquidity_sentinel.aggregation.calibration import liquidity_stress_proxy
from ru_liquidity_sentinel.config import get_settings


class Backtester:
    def __init__(self) -> None:
        self.settings = get_settings()
        self.episodes = self.settings.get("backtest", "stress_episodes", default=[])
        self.holdout_start = self.settings.get("backtest", "holdout_start", default="2020-01-01")
        self.green_max = float(self.settings.get("aggregation", "lsi_green_max", default=40))
        self.red_thr = float(self.settings.get("aggregation", "lsi_yellow_max", default=70))

    def run(self, lsi: pd.DataFrame, ground_truth: pd.DataFrame) -> dict[str, object]:
        if lsi.empty or ground_truth.empty:
            return {"episodes": [], "overall": {}}

        gt = ground_truth.copy()
        if "date" in gt.columns:
            gt = gt.set_index("date")
        gt = gt.sort_index()
        gt["stress_proxy"] = liquidity_stress_proxy(gt)

        lsi_idx = lsi.copy()
        if "date" in lsi_idx.columns:
            lsi_idx = lsi_idx.set_index("date")
        lsi_series = lsi_idx["lsi"].dropna()

        episode_mask = pd.Series(False, index=lsi_series.index)
        episode_results: list[dict[str, object]] = []
        proxy = gt["stress_proxy"].reindex(lsi_series.index)

        for ep in self.episodes:
            mask = (lsi_series.index >= pd.Timestamp(ep["start"])) & (
                lsi_series.index <= pd.Timestamp(ep["end"])
            )
            episode_mask |= mask
            sub = lsi_series[mask]
            if sub.empty:
                continue
            ep_mean = float(sub.mean())
            mean_pct = float((lsi_series <= ep_mean).mean() * 100)
            pct_elevated = float((sub >= self.green_max).mean() * 100)
            sub_proxy = proxy[mask].dropna()
            corr = (
                float(stats.pearsonr(sub.reindex(sub_proxy.index), sub_proxy)[0])
                if len(sub_proxy) > 5
                else None
            )
            episode_results.append(
                {
                    "name": ep["name"],
                    "start": ep["start"],
                    "end": ep["end"],
                    "mean_lsi": ep_mean,
                    "max_lsi": float(sub.max()),
                    "mean_lsi_percentile": mean_pct,
                    "pct_elevated": pct_elevated,
                    "pct_red_zone": float((sub >= self.red_thr).mean() * 100),
                    "detected": bool(mean_pct >= 75 or pct_elevated >= 50),
                    "correlation_with_gt": corr,
                }
            )

        episode_lsi = lsi_series[episode_mask]
        normal_lsi = lsi_series[~episode_mask]
        auc = None
        if len(episode_lsi) > 5 and len(normal_lsi) > 5:
            u = stats.mannwhitneyu(episode_lsi, normal_lsi, alternative="greater").statistic
            auc = float(u / (len(episode_lsi) * len(normal_lsi)))

        merged = lsi_idx[["lsi"]].join(gt[["stress_proxy"]], how="inner").dropna()
        holdout = merged[merged.index >= pd.Timestamp(self.holdout_start)]
        corr_raw = (
            float(stats.pearsonr(holdout["lsi"], holdout["stress_proxy"])[0])
            if len(holdout) > 10
            else None
        )
        smoothed = holdout["stress_proxy"].rolling(21, min_periods=5).mean()
        joined = pd.concat([holdout["lsi"], smoothed], axis=1).dropna()
        corr_smoothed = (
            float(stats.pearsonr(joined.iloc[:, 0], joined.iloc[:, 1])[0])
            if len(joined) > 10
            else None
        )

        return {
            "episodes": episode_results,
            "overall": {
                "episode_discrimination_auc": auc,
                "baseline_mean_lsi": float(normal_lsi.mean()) if len(normal_lsi) else None,
                "episodes_detected": int(sum(e["detected"] for e in episode_results)),
                "episodes_total": len(episode_results),
                "holdout_correlation_smoothed_proxy": corr_smoothed,
                "holdout_correlation_raw_proxy": corr_raw,
                "holdout_start": self.holdout_start,
                "n_observations": len(holdout),
            },
        }

    def save_report(self, report: dict[str, object], path: str | None = None) -> None:
        out = self.settings.processed_dir / (path or "backtest_report.json")
        out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
