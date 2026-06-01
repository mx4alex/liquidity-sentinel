from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from ru_liquidity_sentinel.aggregation.backtest import Backtester
from ru_liquidity_sentinel.aggregation.lsi import LSIAggregator
from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.data.cbr import CBRDataCollector
from ru_liquidity_sentinel.data.fns import FNSCalendarCollector
from ru_liquidity_sentinel.data.minfin import MinfinOFZCollector
from ru_liquidity_sentinel.data.roskazna import RoskaznaCollector
from ru_liquidity_sentinel.llm.comment import LlmCommentError, generate_llm_comment, llm_comment_configured, summarize_upcoming
from ru_liquidity_sentinel.logging import configure_logging, get_logger
from ru_liquidity_sentinel.modules.m1_reserves import ModuleM1Reserves
from ru_liquidity_sentinel.modules.m2_repo import ModuleM2Repo
from ru_liquidity_sentinel.modules.m3_ofz import ModuleM3OFZ
from ru_liquidity_sentinel.modules.m4_tax import ModuleM4TaxSeasonality
from ru_liquidity_sentinel.modules.m5_treasury import ModuleM5Treasury

logger = get_logger(__name__)

@dataclass
class PipelineResult:
    m1: pd.DataFrame
    m2: pd.DataFrame
    m3: pd.DataFrame
    m4: pd.DataFrame
    m5: pd.DataFrame
    features: pd.DataFrame
    lsi: pd.DataFrame
    backtest: dict[str, object]
    sensitivity: pd.DataFrame

class Pipeline:

    def __init__(self) -> None:
        configure_logging()
        self.settings = get_settings()
        self.cbr = CBRDataCollector()
        self.fns = FNSCalendarCollector()
        self.minfin = MinfinOFZCollector()
        self.roskazna = RoskaznaCollector()

    def run(self, use_cache: bool=True) -> PipelineResult:
        logger.info('pipeline_start')
        reserves = self.cbr.fetch_required_reserves()
        ruonia = self.cbr.fetch_ruonia()
        enrich_repo = bool(self.settings.get('pipeline', 'enrich_repo_demand', default=True))
        repo = self.cbr.fetch_repo_auctions(enrich_demand=enrich_repo)
        keyrate = self.cbr.fetch_keyrate()
        ofz = self.minfin.fetch_ofz_auctions()
        tax_cal = self.fns.fetch_tax_dates()
        bliquidity = self.cbr.fetch_bliquidity()
        sors = self.cbr.fetch_sors_attracted_funds()
        roskazna = self.roskazna.fetch_deposit_placements()
        m1 = ModuleM1Reserves().run(reserves=reserves, ruonia=ruonia)
        m2 = ModuleM2Repo().run(repo=repo, keyrate=keyrate)
        m3 = ModuleM3OFZ().run(ofz=ofz, keyrate=keyrate)
        m5 = ModuleM5Treasury().run(sors=sors, roskazna=roskazna)
        m4 = ModuleM4TaxSeasonality().run(tax_calendar=tax_cal, m1=m1, m2=m2, m5=m5)
        out_dir = self.settings.processed_dir
        for name, df in [('m1', m1), ('m2', m2), ('m3', m3), ('m4', m4), ('m5', m5)]:
            if not df.empty:
                df.to_parquet(out_dir / f'{name}_signals.parquet')
        aggregator = LSIAggregator()
        features = aggregator.build_feature_matrix(m1, m2, m3, m4, m5)
        features.to_parquet(out_dir / 'features.parquet')
        aggregator.save_overlap_report(features)
        lsi = aggregator.compute_lsi(features, ground_truth=bliquidity)
        lsi.to_parquet(out_dir / 'lsi.parquet')
        if not lsi.empty and llm_comment_configured():
            tax_str, ofz_str = summarize_upcoming(tax_cal, ofz, asof=lsi.index.max())
            try:
                comment = generate_llm_comment(lsi.iloc[-1], tax_str, ofz_str, lsi_history=lsi)
                (out_dir / 'auto_comment.txt').write_text(comment, encoding='utf-8')
            except LlmCommentError as exc:
                logger.warning('auto_comment_skipped', reason=str(exc))
        elif not lsi.empty:
            logger.warning('auto_comment_skipped', reason='Yandex GPT not configured')
        sensitivity = aggregator.sensitivity_analysis(features)
        sensitivity.to_csv(out_dir / 'sensitivity.csv', index=False)
        backtester = Backtester()
        bt_report = backtester.run(lsi, bliquidity)
        backtester.save_report(bt_report)
        logger.info('pipeline_complete', lsi_rows=len(lsi))
        return PipelineResult(m1=m1, m2=m2, m3=m3, m4=m4, m5=m5, features=features, lsi=lsi, backtest=bt_report, sensitivity=sensitivity)

    def load_cached(self) -> PipelineResult | None:
        out = self.settings.processed_dir
        lsi_path = out / 'lsi.parquet'
        if not lsi_path.exists():
            return None
        return PipelineResult(m1=pd.read_parquet(out / 'm1_signals.parquet'), m2=pd.read_parquet(out / 'm2_signals.parquet'), m3=pd.read_parquet(out / 'm3_signals.parquet'), m4=pd.read_parquet(out / 'm4_signals.parquet'), m5=pd.read_parquet(out / 'm5_signals.parquet'), features=pd.read_parquet(out / 'features.parquet'), lsi=pd.read_parquet(lsi_path), backtest={}, sensitivity=pd.read_csv(out / 'sensitivity.csv'))
