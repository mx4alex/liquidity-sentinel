from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.aggregation.lsi_points import lsi_points_frame
from ru_liquidity_sentinel.config import get_settings


def load_lsi() -> pd.DataFrame:
    path = get_settings().processed_dir / 'lsi.parquet'
    if not path.exists():
        return pd.DataFrame()
    lsi = pd.read_parquet(path)
    if 'date' in lsi.columns:
        lsi = lsi.set_index('date')
    lsi.index = pd.to_datetime(lsi.index)
    return lsi.sort_index()


def summarize_lsi_period(period: str) -> str | None:
    lsi = load_lsi()
    if lsi.empty:
        return None

    if len(period) == 7:
        start = pd.Timestamp(f'{period}-01')
        end = start + pd.offsets.MonthEnd(0)
        label = start.strftime('%B %Y').capitalize()
        month_ru = {
            1: 'январь', 2: 'февраль', 3: 'март', 4: 'апрель', 5: 'май', 6: 'июнь',
            7: 'июль', 8: 'август', 9: 'сентябрь', 10: 'октябрь', 11: 'ноябрь', 12: 'декабрь',
        }
        label = f'{month_ru[start.month]} {start.year}'
        chunk = lsi.loc[start:end]
    elif len(period) == 4 and period.isdigit():
        label = period
        chunk = lsi.loc[period]
    else:
        return None

    if chunk.empty:
        return f'В базе LSI нет данных за {label}.'

    mean_lsi = float(chunk['lsi'].mean())
    max_lsi = float(chunk['lsi'].max())
    min_lsi = float(chunk['lsi'].min())
    peak_day = chunk['lsi'].idxmax()
    peak_date = peak_day.strftime('%d.%m.%Y') if hasattr(peak_day, 'strftime') else str(peak_day)[:10]

    status_col = 'status_ru' if 'status_ru' in chunk.columns else 'status'
    if status_col in chunk.columns:
        counts = chunk[status_col].value_counts()
        status_txt = ', '.join(f'{k}: {int(v)} дн.' for k, v in counts.items())
    else:
        status_txt = 'н/д'

    points = lsi_points_frame(chunk)
    avg_pts = points.mean(numeric_only=True)
    top_mod = avg_pts.idxmax() if not avg_pts.empty else '—'
    top_val = float(avg_pts.max()) if not avg_pts.empty else 0.0

    flags = chunk.get('active_flags', pd.Series(dtype=str)).dropna().astype(str)
    flags = flags[flags.str.strip() != '']
    flag_sample = ', '.join(sorted({f.strip() for s in flags.head(20) for f in s.split(',') if f.strip()}))[:120]

    lines = [
        f'Ликвидность за {label} (по индексу LSI, 0–100):',
        f'средний LSI {mean_lsi:.1f}, минимум {min_lsi:.1f}, максимум {max_lsi:.1f} (пик {peak_date})',
        f'распределение статусов: {status_txt}',
        f'средний вклад в LSI баллами: лидирует {top_mod} (~{top_val:.1f} б.)',
    ]
    if flag_sample:
        lines.append(f'частые флаги: {flag_sample}')

    if label.startswith('март') and '2022' in label:
        lines.append(
            'контекст: после введения санкций в конце февраля 2022 рынок оставался волатильным; '
            'по нашему индексу март в основном в зоне «Норма», с отдельными днями напряжения '
            '(пик в начале месяца).'
        )
    elif '2022' in label and ('феврал' in label or period.endswith('-02')):
        lines.append(
            'контекст: февраль 2022 — эпизод повышенного стресса (санкции, скачок ключевой ставки); '
            'см. также backtest-эпизод «February–March 2022».'
        )

    return '\n'.join(lines)
