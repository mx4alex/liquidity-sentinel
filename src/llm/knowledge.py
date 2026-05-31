from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ru_liquidity_sentinel.aggregation.lsi_points import lsi_points_frame
from ru_liquidity_sentinel.config import get_settings


def _lsi_monthly_docs(lsi: pd.DataFrame) -> list[dict[str, str]]:
    if lsi.empty:
        return []
    docs: list[dict[str, str]] = []
    monthly = lsi.groupby(lsi.index.to_period('M'))
    month_names = {
        1: 'январь', 2: 'февраль', 3: 'март', 4: 'апрель', 5: 'май', 6: 'июнь',
        7: 'июль', 8: 'август', 9: 'сентябрь', 10: 'октябрь', 11: 'ноябрь', 12: 'декабрь',
    }
    for period, chunk in monthly:
        if len(chunk) < 3:
            continue
        ts = period.to_timestamp()
        label = f'{month_names[ts.month]} {ts.year}'
        status_col = 'status_ru' if 'status_ru' in chunk.columns else None
        red = int((chunk['lsi'] >= 70).sum()) if 'lsi' in chunk.columns else 0
        yellow = int(((chunk['lsi'] >= 40) & (chunk['lsi'] < 70)).sum()) if 'lsi' in chunk.columns else 0
        status_top = ''
        if status_col:
            top_status = chunk[status_col].mode()
            status_top = str(top_status.iloc[0]) if len(top_status) else ''
        pts = lsi_points_frame(chunk).mean(numeric_only=True)
        lead = pts.idxmax() if not pts.empty else '—'
        text = (
            f'Сводка LSI за {label} ({period}): '
            f"средний LSI {chunk['lsi'].mean():.1f}, "
            f"макс {chunk['lsi'].max():.1f} ({chunk['lsi'].idxmax().date()}), "
            f"мин {chunk['lsi'].min():.1f}, "
            f'дней напряжение/стресс: {yellow}/{red}, '
            f'преобладающий статус {status_top}, '
            f'лидер вклада {lead} (~{float(pts.max()) if not pts.empty else 0:.1f} б.)'
        )
        docs.append({'id': f'lsi_month_{period}', 'text': text})
    return docs


def build_knowledge_documents() -> list[dict[str, str]]:
    settings = get_settings()
    proc = settings.processed_dir
    docs: list[dict[str, str]] = []
    docs.append({
        'id': 'methodology',
        'text': (
            'RU Liquidity Sentinel: LSI 0–100, зоны норма <40, напряжение 40–70, стресс ≥70. '
            'M1 резервы, M2 репо ЦБ, M3 ОФЗ, M4 налоги/сезонность, M5 казначейство. '
            'Для вопросов «что было в марте 2022» используй сводки lsi_month_YYYY-MM.'
        ),
    })
    root = Path(__file__).resolve().parents[3]
    meth_path = root / 'docs' / 'PRESENTATION.md'
    if meth_path.exists():
        docs.append({'id': 'presentation_md', 'text': meth_path.read_text(encoding='utf-8')[:4000]})

    cal_path = proc / 'lsi_calibration.json'
    if cal_path.exists():
        docs.append({'id': 'calibration', 'text': f'Параметры калибровки LSI:\n{cal_path.read_text(encoding="utf-8")}'})

    bt_path = proc / 'backtest_report.json'
    if bt_path.exists():
        bt = json.loads(bt_path.read_text(encoding='utf-8'))
        lines = ['Backtest LSI vs bliquidity ЦБ (валидация эпизодов, не ответ на «что было в месяце»):']
        for ep in bt.get('episodes', []):
            lines.append(
                f"- {ep.get('name')}: mean_lsi={ep.get('mean_lsi', 0):.1f}, "
                f"max_lsi={ep.get('max_lsi', 0):.1f}, %красная зона={ep.get('pct_red_zone', 0):.1f}%"
            )
        overall = bt.get('overall', {})
        lines.append(f"Holdout correlation: {overall.get('holdout_correlation')}")
        docs.append({'id': 'backtest', 'text': '\n'.join(lines)})

    overlap_path = proc / 'overlap_report.json'
    if overlap_path.exists():
        docs.append({'id': 'overlap', 'text': f'Отчёт двойного счёта M4:\n{overlap_path.read_text(encoding="utf-8")}'})

    lsi_path = proc / 'lsi.parquet'
    if lsi_path.exists():
        lsi = pd.read_parquet(lsi_path)
        if 'date' in lsi.columns:
            lsi = lsi.set_index('date')
        lsi.index = pd.to_datetime(lsi.index)
        lsi = lsi.sort_index()
        docs.extend(_lsi_monthly_docs(lsi))

        for episode_start, episode_label in (
            ('2014-12', 'декабрь 2014'),
            ('2022-02', 'февраль 2022'),
            ('2022-03', 'март 2022'),
            ('2023-08', 'август 2023'),
        ):
            month_start = pd.Timestamp(f'{episode_start}-01')
            month_end = month_start + pd.offsets.MonthEnd(0)
            ep = lsi.loc[month_start:month_end]
            if not ep.empty:
                docs.append({
                    'id': f'episode_{episode_start}',
                    'text': (
                        f'Эпизод {episode_label}: LSI средний {ep["lsi"].mean():.1f}, '
                        f'макс {ep["lsi"].max():.1f}, мин {ep["lsi"].min():.1f}, '
                        f'дней LSI≥70: {int((ep["lsi"] >= 70).sum())}'
                    ),
                })

        for date, row in lsi.tail(120).iterrows():
            d = date.date() if hasattr(date, 'date') else str(date)[:10]
            docs.append({
                'id': f'lsi_{d}',
                'text': (
                    f'Дата {d}: LSI={row.get("lsi", 0):.1f}, статус={row.get("status_ru", "")}, '
                    f'M2={row.get("contrib_m2", 0):.2f}, M3={row.get("contrib_m3", 0):.2f}, '
                    f'seasonal={row.get("seasonal_factor", 1):.2f}'
                ),
            })

    for mod in ('m1', 'm2', 'm3', 'm5'):
        p = proc / f'{mod}_signals.parquet'
        if not p.exists():
            continue
        df = pd.read_parquet(p)
        if df.empty:
            continue
        stress_col = f'{mod}_stress'
        if stress_col not in df.columns:
            continue
        if not isinstance(df.index, pd.DatetimeIndex) and 'date' in df.columns:
            df = df.set_index('date')
        top = df[stress_col].sort_values(ascending=False).head(10)
        lines = [f'Топ стресса {mod.upper()}:']
        for dt, val in top.items():
            d = dt.date() if hasattr(dt, 'date') else str(dt)[:10]
            lines.append(f'  {d}: stress={val:.2f}')
        docs.append({'id': f'module_{mod}', 'text': '\n'.join(lines)})

    samples_readme = root / 'data' / 'samples' / 'README.md'
    if samples_readme.exists():
        docs.append({'id': 'data_sources', 'text': samples_readme.read_text(encoding='utf-8')})
    return docs
