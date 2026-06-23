"""
src/search_index/embedder.py
--------------------------------------
TF-IDF–based embedding function used as the vector store backend.

Implements the same interface as ChromaDB EmbeddingFunction so it can
be swapped for sentence-transformers or OpenAI embeddings without
changing any call sites.
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from .tokenizer import tokenize, build_tfidf_matrix


class TFIDFEmbedder:
    """
    Embedding function backed by TF-IDF rather than a neural model.

    Interface matches ChromaDB's EmbeddingFunction:
        __call__(input: list[str]) → list[list[float]]

    To swap in sentence-transformers or OpenAI, replace this class
    with a wrapper that implements the same __call__ / save / load
    contract without changing HybridIndex or VectorIndex.
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, corpus: list[str]) -> "TFIDFEmbedder":
        """Build vocabulary and IDF weights from a text corpus."""
        tokenized  = [tokenize(text) for text in corpus]
        all_tokens = {t for doc in tokenized for t in doc}
        self._vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        _, self._idf = build_tfidf_matrix(tokenized, self._vocab)
        self._fitted = True
        return self

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed a list of strings into L2-normalised TF-IDF float vectors."""
        if not self._fitted:
            raise RuntimeError(
                "TFIDFEmbedder has not been fitted yet. Call fit() first."
            )
        result = []
        for text in input:
            tokens = tokenize(text)
            vec    = np.zeros(len(self._vocab), dtype=np.float32)
            for token in tokens:
                if token in self._vocab:
                    vec[self._vocab[token]] += 1
            if self._idf is not None:
                vec = vec * self._idf
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm
            result.append(vec.tolist())
        return result

    def save(self, path: Path) -> None:
        """Persist IDF weights and vocabulary to disk."""
        np.save(str(path / "embedder_idf.npy"), self._idf)
        with open(path / "embedder_vocab.json", "w") as f:
            json.dump(self._vocab, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbedder":
        """Load a previously saved embedder from disk."""
        obj         = cls()
        obj._idf    = np.load(str(path / "embedder_idf.npy"))
        with open(path / "embedder_vocab.json") as f:
            obj._vocab = json.load(f)
        obj._fitted = True
        return obj
