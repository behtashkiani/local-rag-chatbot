"""
Retrieval engine: a dependency-light TF-IDF vector index with cosine
similarity ranking, implemented directly on top of NumPy.

Why TF-IDF (and not an embedding model)?
----------------------------------------
The brief calls for a portable retriever with no external API dependency.
TF-IDF + cosine similarity is a strong, transparent lexical baseline that is
fully deterministic and runs anywhere NumPy runs. To recover some of the recall
that pure lexical matching loses, the tokenizer emits both **unigrams and
bigrams**, so multi-word concepts ("return policy", "restocking fee",
"store hours") are matched as phrases, not just bags of single words.

Scaling note
------------
For a small/medium knowledge base the term-document matrix is dense and tiny,
so a dense NumPy matrix is the simplest correct choice. For very large corpora
you would swap the dense matrix for ``scipy.sparse`` (or a dedicated vector
store) behind this same ``retrieve`` interface.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

import numpy as np

from .document_processor import Chunk

# A compact English stopword list. Removing these focuses the vector space on
# content-bearing terms and prevents ubiquitous words from dominating scores.
_STOPWORDS: frozenset[str] = frozenset(
    """
    a an and are as at be but by for from has have if in into is it its of on
    or our that the their then there these they this to was were what when
    where which who will with you your i we us can do does
    """.split()
)

_TOKEN = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase, extract alphanumeric tokens, drop stopwords."""
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOPWORDS]


def _terms(text: str) -> list[str]:
    """Unigrams plus adjacent bigrams from the cleaned token stream."""
    tokens = _tokenize(text)
    if not tokens:
        return []
    bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
    return tokens + bigrams


@dataclass
class RetrievalResult:
    """A retrieved chunk paired with its similarity score."""

    chunk: Chunk
    score: float


class TfidfRetriever:
    """An in-memory TF-IDF retriever over a fixed set of chunks.

    The index is built once at construction time. ``retrieve`` is read-only and
    safe to call concurrently.
    """

    def __init__(self, chunks: list[Chunk]) -> None:
        if not chunks:
            raise ValueError("TfidfRetriever requires at least one chunk.")
        self._chunks: list[Chunk] = list(chunks)
        self._vocab: dict[str, int] = {}
        self._idf: np.ndarray = np.empty(0)
        # L2-normalized document matrix, shape (n_chunks, vocab_size).
        self._matrix: np.ndarray = np.empty((0, 0))
        self._build_index()

    # -- public API --------------------------------------------------------
    @property
    def num_chunks(self) -> int:
        return len(self._chunks)

    @property
    def vocab_size(self) -> int:
        return len(self._vocab)

    def retrieve(
        self, query: str, top_k: int = 3, min_score: float = 0.0
    ) -> list[RetrievalResult]:
        """Return up to ``top_k`` chunks most similar to ``query``.

        Parameters
        ----------
        query:
            The user's text. Empty/whitespace queries yield an empty list.
        top_k:
            Maximum number of results to return.
        min_score:
            Minimum cosine similarity; results below this are discarded so the
            caller never receives irrelevant noise.
        """
        if not query or not query.strip() or top_k < 1:
            return []

        q_vec = self._vectorize(query)
        norm = float(np.linalg.norm(q_vec))
        if norm == 0.0:
            # No query term overlaps the vocabulary -> nothing to retrieve.
            return []
        q_vec = q_vec / norm

        # Both sides are L2-normalized, so the dot product is cosine similarity.
        scores = self._matrix @ q_vec  # shape (n_chunks,)

        # Take the top_k by score efficiently, then sort just those.
        k = min(top_k, scores.shape[0])
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]

        results: list[RetrievalResult] = []
        for i in top_idx:
            score = float(scores[i])
            if score <= min_score:
                continue
            results.append(RetrievalResult(chunk=self._chunks[int(i)], score=score))
        return results

    # -- internals ---------------------------------------------------------
    def _build_index(self) -> None:
        tokenized_docs: list[list[str]] = [_terms(c.text) for c in self._chunks]

        # Build vocabulary and document frequencies in one pass.
        df: dict[str, int] = {}
        for terms in tokenized_docs:
            for term in set(terms):
                df[term] = df.get(term, 0) + 1
        self._vocab = {term: idx for idx, term in enumerate(sorted(df))}

        n_docs = len(self._chunks)
        vocab_size = len(self._vocab)

        # Smoothed IDF: ln((1 + N) / (1 + df)) + 1  (always positive).
        idf = np.zeros(vocab_size, dtype=np.float64)
        for term, idx in self._vocab.items():
            idf[idx] = math.log((1 + n_docs) / (1 + df[term])) + 1.0
        self._idf = idf

        # Build the TF-IDF matrix with sublinear TF scaling, then L2-normalize.
        matrix = np.zeros((n_docs, vocab_size), dtype=np.float64)
        for row, terms in enumerate(tokenized_docs):
            counts: dict[int, int] = {}
            for term in terms:
                col = self._vocab.get(term)
                if col is not None:
                    counts[col] = counts.get(col, 0) + 1
            for col, count in counts.items():
                tf = 1.0 + math.log(count)  # sublinear term frequency
                matrix[row, col] = tf * idf[col]

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0  # guard empty rows against divide-by-zero
        self._matrix = matrix / norms

    def _vectorize(self, text: str) -> np.ndarray:
        """Project arbitrary text into the (un-normalized) TF-IDF space."""
        vec = np.zeros(len(self._vocab), dtype=np.float64)
        counts: dict[int, int] = {}
        for term in _terms(text):
            col = self._vocab.get(term)
            if col is not None:
                counts[col] = counts.get(col, 0) + 1
        for col, count in counts.items():
            vec[col] = (1.0 + math.log(count)) * self._idf[col]
        return vec


def build_retriever(chunks: list[Chunk]) -> TfidfRetriever:
    """Factory that constructs a ready-to-query retriever from chunks."""
    return TfidfRetriever(chunks)
