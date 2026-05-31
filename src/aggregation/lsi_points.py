from __future__ import annotations

import pandas as pd

FLAG_LABELS: dict[str, str] = {
    'tax_week_flag': 'налоговая неделя',
    'end_of_month_flag': 'конец месяца',
    'end_of_quarter_flag': 'конец квартала',
    'flag_end_of_period': 'конец периода усреднения резервов',
    'flag_overfulfillment': 'перебор нормы резервов',
    'flag_demand': 'переспрос на репо ЦБ',
    'flag_nedospros': 'недоспрос ОФЗ',
    'flag_perespros': 'переспрос ОФЗ',
    'flag_budget_drain': 'отток бюджетных средств',
}


def format_active_flags(flags: str | None) -> str:
    if not flags or str(flags).strip() in ('', 'нет'):
        return 'нет'
    parts = []
    for name in str(flags).split(','):
        key = name.strip()
        if key:
            parts.append(FLAG_LABELS.get(key, key))
    return ', '.join(parts) if parts else 'нет'


def lsi_points_frame(lsi: pd.DataFrame) -> pd.DataFrame:
    if 'raw_stress_sum' not in lsi.columns or 'seasonal_factor' not in lsi.columns:
        cols = [c for c in lsi.columns if c.startswith('contrib_')]
        return lsi[cols].rename(columns=lambda c: c.replace('contrib_', '').upper())
    seasonal = pd.to_numeric(lsi['seasonal_factor'], errors='coerce').replace(0, 1.0).fillna(1.0)
    lsi_val = pd.to_numeric(lsi['lsi'], errors='coerce').fillna(0.0)
    base_lsi = (lsi_val / seasonal).fillna(0.0)
    raw = pd.to_numeric(lsi['raw_stress_sum'], errors='coerce')
    raw_safe = raw.where(raw != 0)
    out = pd.DataFrame(index=lsi.index)
    for m in ('m1', 'm2', 'm3', 'm5'):
        col = f'contrib_{m}'
        if col in lsi.columns:
            share = pd.to_numeric(lsi[col], errors='coerce').divide(raw_safe)
            out[m.upper()] = (base_lsi * share).fillna(0.0).astype(float)
    out['M4'] = (lsi_val - base_lsi).clip(lower=0.0).astype(float)
    order = [c for c in ('M1', 'M2', 'M3', 'M4', 'M5') if c in out.columns]
    return out[order].astype(float)
