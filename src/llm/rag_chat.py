from __future__ import annotations

import re
from dataclasses import dataclass, field

from ru_liquidity_sentinel.config import get_settings
from ru_liquidity_sentinel.llm.knowledge import build_knowledge_documents
from ru_liquidity_sentinel.llm.lsi_context import summarize_lsi_period
from ru_liquidity_sentinel.llm.retriever import build_retriever
from ru_liquidity_sentinel.llm.yandex_gpt import complete as yandex_complete


@dataclass
class ChatSession:
    messages: list[dict[str, str]] = field(default_factory=list)


class SentinelRAG:

    def __init__(self) -> None:
        self.settings = get_settings()
        self._docs = build_knowledge_documents()
        self._retriever = build_retriever(self._docs)
        self._backend = type(self._retriever).__name__

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def documents(self) -> list[dict[str, str]]:
        return self._docs

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        k = top_k or int(self.settings.get('llm', 'rag_top_k', default=8))
        period = _extract_period_hint(query.lower())
        if period:
            boosted = [
                d['text']
                for d in self._docs
                if period in d['id'] or period in d['text'] or f'lsi_month_{period}' == d['id']
            ]
            if boosted:
                hits = self._retriever.retrieve(query, top_k=k)
                rest = [text for _, text in hits if text not in boosted]
                return (boosted[:3] + rest)[:k]
        hits = self._retriever.retrieve(query, top_k=k)
        if hits:
            return [text for _, text in hits]
        return [d['text'] for d in self._docs[:k]]

    def ask(self, query: str, session: ChatSession | None = None) -> str:
        session = session or ChatSession()
        period = _extract_period_hint(query.lower())
        period_block = summarize_lsi_period(period) if period else None

        context = '\n---\n'.join(self.retrieve(query))
        if period_block:
            context = period_block + '\n---\n' + context

        settings = get_settings()
        if settings.llm_active and settings.llm_ready:
            try:
                history = [
                    {
                        'role': 'system',
                        'text': (
                            'Ты — аналитик казначейства ПСБ, RU Liquidity Sentinel. '
                            'Отвечай на русском кратко (4–8 предложений). '
                            'На вопросы о конкретном месяце/годе опирайся на сводку LSI за период, '
                            'не пересказывай методологию и backtest, если об этом не спрашивали. '
                            'Укажи средний/макс LSI, статус, главный канал давления (M2/M3/…).'
                        ),
                    },
                ]
                for msg in session.messages[-6:]:
                    history.append(msg)
                history.append(
                    {
                        'role': 'user',
                        'text': f'Данные системы:\n{context}\n\nВопрос: {query}',
                    },
                )
                answer = yandex_complete(
                    history,
                    api_key=settings.yandex_api_key or '',
                    folder_id=settings.yandex_folder_id or '',
                    model=settings.llm_model,
                    temperature=float(settings.get('llm', 'temperature', default=0.2)),
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
                answer = f'Ошибка LLM: {exc}\n\n{self._rule_based_answer(query, context, period_block)}'
        else:
            answer = self._rule_based_answer(query, context, period_block)

        session.messages.append({'role': 'user', 'content': query})
        session.messages.append({'role': 'assistant', 'content': answer})
        return answer

    def _rule_based_answer(
        self,
        query: str,
        context: str,
        period_block: str | None = None,
    ) -> str:
        q = query.lower()
        if period_block:
            return period_block

        if _is_backtest_question(q):
            bt = [line for line in context.splitlines() if 'mean_lsi' in line or 'Backtest' in line]
            if bt:
                return 'Результаты backtest (валидация на эпизодах):\n' + '\n'.join(bt[:8])

        if 'вес' in q or 'калибр' in q:
            cal = [line for line in context.splitlines() if 'm1' in line.lower() or 'калибр' in line.lower()]
            if cal:
                return 'Калибровка:\n' + '\n'.join(cal[:10])

        period = _extract_period_hint(q)
        if period:
            rows = [line for line in context.splitlines() if period in line or f'lsi_month_{period}' in line]
            if rows:
                return f'По данным LSI за {period}:\n' + '\n'.join(rows[:8])

        if 'стресс' in q or 'lsi' in q or 'ликвидност' in q:
            lsi_lines = [ln for ln in context.splitlines() if 'LSI' in ln or 'lsi' in ln.lower()][:10]
            if lsi_lines:
                return 'Релевантные фрагменты:\n' + '\n'.join(lsi_lines)

        return (
            'Не удалось однозначно ответить. Уточните период (например, «март 2022») '
            'или спросите про backtest / калибровку.\n\n'
            f'Контекст ({self._backend}):\n{context[:1500]}'
        )


def _is_backtest_question(q: str) -> bool:
    keys = ('backtest', 'бэктест', 'валидац', 'auc', 'out-of-sample', 'out of sample', 'holdout')
    return any(k in q for k in keys) or ('эпизод' in q and 'что было' not in q and 'ликвидност' not in q)


def _extract_period_hint(q: str) -> str | None:
    m = re.search(r'(20\d{2})[-/](0[1-9]|1[0-2])', q)
    if m:
        return f'{m.group(1)}-{m.group(2)}'
    months = {
        'январ': '01', 'феврал': '02', 'март': '03', 'апрел': '04',
        'май': '05', 'июн': '06', 'июл': '07', 'август': '08',
        'сентябр': '09', 'октябр': '10', 'ноябр': '11', 'декабр': '12',
    }
    year_match = re.search(r'20\d{2}', q)
    year = year_match.group(0) if year_match else None
    for stem, mm in months.items():
        if stem in q:
            if not year:
                return None
            return f'{year}-{mm}'
    if year_match and not any(stem in q for stem in months):
        return year_match.group(0)
    return None
