from __future__ import annotations

import pandas as pd

from ru_liquidity_sentinel.aggregation.lsi_points import format_active_flags, lsi_points_frame
from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.llm.yandex_gpt import complete as yandex_complete


class LlmCommentError(Exception):
    pass


def summarize_upcoming(
    tax_calendar: pd.DataFrame | None,
    ofz: pd.DataFrame | None = None,
    asof: pd.Timestamp | None = None,
    horizon_days: int = 21,
) -> tuple[str, str]:
    asof = pd.Timestamp(asof).normalize() if asof is not None else pd.Timestamp.today().normalize()
    horizon = asof + pd.Timedelta(days=horizon_days)
    tax_str = 'н/д'
    if tax_calendar is not None and (not tax_calendar.empty) and ('date' in tax_calendar.columns):
        d = pd.to_datetime(tax_calendar['date'], errors='coerce')
        upcoming = sorted(d[(d > asof) & (d <= horizon)].dt.strftime('%d.%m.%Y').unique())
        if upcoming:
            tax_str = ', '.join(upcoming[:4])
    next_wed = asof + pd.Timedelta(days=(2 - asof.weekday()) % 7 or 7)
    ofz_days = [
        (next_wed + pd.Timedelta(weeks=i)).strftime('%d.%m.%Y')
        for i in range(3)
        if next_wed + pd.Timedelta(weeks=i) <= horizon
    ]
    ofz_str = ', '.join(ofz_days) + ', ожид. среда' if ofz_days else 'н/д'
    return tax_str, ofz_str


def _points_for_row(row: pd.Series, lsi_frame: pd.DataFrame) -> dict[str, float]:
    if row.name in lsi_frame.index:
        pt = lsi_frame.loc[row.name]
    else:
        pt = lsi_frame.iloc[-1]
    return {m: float(pt.get(m, 0.0)) for m in pt.index}


def llm_comment_configured() -> bool:
    settings = get_settings()
    return bool(settings.llm_active and settings.llm_ready)


def generate_llm_comment(
    row: pd.Series,
    upcoming_tax: str = 'н/д',
    upcoming_ofz: str = 'н/д',
    lsi_history: pd.DataFrame | None = None,
) -> str:
    settings = get_settings()
    if not settings.llm_active or not settings.llm_ready:
        raise LlmCommentError(
            'Для аналитического комментария нужен Yandex GPT: '
            'задайте YANDEX_API_KEY, YANDEX_FOLDER_ID и включите llm.enabled в config/settings.yaml '
            '(или SENTINEL_LLM_ENABLED=1).'
        )

    if lsi_history is not None and not lsi_history.empty:
        points = _points_for_row(row, lsi_points_frame(lsi_history))
    else:
        one = pd.DataFrame([row])
        if row.name is not None:
            one.index = [row.name]
        points = _points_for_row(row, lsi_points_frame(one))

    pts_text = ', '.join(f'{k}={v:.1f} б.' for k, v in points.items())
    prompt = (
        'Ты — аналитик казначейства ПСБ, RU Liquidity Sentinel.\n'
        f"Текущий LSI: {float(row.get('lsi', 0)):.1f}, статус {row.get('status_ru', '')}.\n"
        f'Вклад в LSI баллами (сумма = итоговому LSI): {pts_text}.\n'
        f'M4 сезонный множитель: {float(row.get("seasonal_factor", 1.0) or 1.0):.2f}.\n'
        f"Активные флаги: {format_active_flags(row.get('active_flags'))}.\n"
        f'Ближайшие события: налоги {upcoming_tax}; аукционы ОФЗ {upcoming_ofz}.\n\n'
        'Напиши аналитический комментарий на русском в 3–5 предложений: '
        'что происходило, кто давил на ликвидность, чего ожидать. '
        'Не перечисляй сырые имена полей и флагов — только смысл для казначейства.'
    )
    try:
        text = yandex_complete(
            [{'role': 'user', 'text': prompt}],
            api_key=settings.yandex_api_key or '',
            folder_id=settings.yandex_folder_id or '',
            model=settings.llm_model,
            temperature=float(settings.get('llm', 'temperature', default=0.3)),
            max_tokens=int(settings.get('llm', 'max_tokens', default=1500)),
            api_url=str(
                settings.get(
                    'llm',
                    'api_url',
                    default='https://llm.api.cloud.yandex.net/foundationModels/v1/completion',
                ),
            ),
        ).strip()
    except Exception as exc:
        raise LlmCommentError(f'Ошибка Yandex GPT: {exc}') from exc

    if not text:
        raise LlmCommentError('Yandex GPT вернул пустой ответ.')
    return text
