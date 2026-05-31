from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class BaseRetriever(ABC):

    @abstractmethod
    def retrieve(self, query: str, top_k: int=8) -> list[tuple[float, str]]:
        pass

class TfidfRetriever(BaseRetriever):

    def __init__(self, documents: list[dict[str, str]]) -> None:
        self._ids = [d['id'] for d in documents]
        self._texts = [d['text'] for d in documents]
        if not self._texts:
            self._vectorizer = None
            self._matrix = None
            return
        self._vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), strip_accents='unicode')
        self._matrix = self._vectorizer.fit_transform(self._texts)

    def retrieve(self, query: str, top_k: int=8) -> list[tuple[float, str]]:
        if not self._texts or self._matrix is None or self._vectorizer is None:
            return []
        qv = self._vectorizer.transform([query])
        scores = cosine_similarity(qv, self._matrix).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self._texts[i]) for i in order if scores[i] > 0.01]

def build_retriever(documents: list[dict[str, str]]) -> BaseRetriever:
    import os
    if os.environ.get('SENTINEL_USE_EMBEDDINGS', '').lower() in ('1', 'true', 'yes'):
        try:
            return EmbeddingRetriever(documents)
        except Exception:
            pass
    return TfidfRetriever(documents)

class EmbeddingRetriever(BaseRetriever):

    def __init__(self, documents: list[dict[str, str]]) -> None:
        from sentence_transformers import SentenceTransformer
        self._texts = [d['text'] for d in documents]
        if not self._texts:
            self._embeddings = None
            self._model = None
            return
        self._model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self._embeddings = self._model.encode(self._texts, normalize_embeddings=True)

    def retrieve(self, query: str, top_k: int=8) -> list[tuple[float, str]]:
        if self._embeddings is None or self._model is None:
            return []
        q = self._model.encode([query], normalize_embeddings=True)
        scores = (self._embeddings @ q.T).ravel()
        order = np.argsort(scores)[::-1][:top_k]
        return [(float(scores[i]), self._texts[i]) for i in order if scores[i] > 0.05]
