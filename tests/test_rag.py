from __future__ import annotations
from ru_liquidity_sentinel.llm.knowledge import build_knowledge_documents
from ru_liquidity_sentinel.llm.rag_chat import SentinelRAG, _extract_period_hint
from ru_liquidity_sentinel.llm.retriever import TfidfRetriever

def test_tfidf_retriever_finds_lsi() -> None:
    docs = [{'id': 'a', 'text': 'LSI 72 стресс ликвидности март 2022 репо ЦБ'}, {'id': 'b', 'text': 'спокойный день LSI 25 норма'}]
    retriever = TfidfRetriever(docs)
    hits = retriever.retrieve('март 2022 стресс', top_k=1)
    assert hits
    assert '2022' in hits[0][1] or '72' in hits[0][1]

def test_period_hint() -> None:
    assert _extract_period_hint('что было в марте 2022') == '2022-03'
    assert _extract_period_hint('август 2023') == '2023-08'

def test_knowledge_builder_runs() -> None:
    docs = build_knowledge_documents()
    assert any((d['id'] == 'methodology' for d in docs))

def test_sentinel_rag_rule_answer() -> None:
    rag = SentinelRAG()
    answer = rag._rule_based_answer('backtest 2022', 'Backtest LSI vs bliquidity\n- February–March 2022: mean_lsi=40', None)
    assert '2022' in answer or 'backtest' in answer.lower()


def test_march_2022_not_backtest_dump() -> None:
    rag = SentinelRAG()
    answer = rag.ask('Что было с ликвидностью в марте 2022?')
    assert 'mean_lsi' not in answer or 'март' in answer.lower()
    assert 'средний LSI' in answer or 'LSI' in answer
