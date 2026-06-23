"""
src/indexing/search_index/tokenizer.py
---------------------------------------
Text tokenisation and TF-IDF matrix construction utilities.

Both HybridIndex and VectorIndex share these functions, so they live
here instead of inside either index class, eliminating the previous
cross-module import from vector_index → hybrid_index.
"""

import re

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Tokeniser
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenise text for BM25 and TF-IDF.

    Handles camelCase, kebab-case, dotted paths, and normal words.
    Returns lowercase tokens of at least 2 characters.
    """
    text = text.lower()
    # Split camelCase: pukHeader → puk header
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric (hyphens, dots, underscores, spaces)
    tokens = re.split(r"[^a-z0-9]+", text)
    return [t for t in tokens if len(t) >= 2]


# ─────────────────────────────────────────────────────────────────────────────
# TF-IDF matrix builder
# ─────────────────────────────────────────────────────────────────────────────

def build_tfidf_matrix(
    corpus: list[list[str]],
    vocab: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build an L2-normalised TF-IDF matrix (n_docs × vocab_size).

    Used for vector similarity without an external embedding API.
    Swap-ready: replace this function to plug in any other embeddings.

    Args:
        corpus: Tokenised documents — list of token lists.
        vocab:  Mapping of token → column index in the output matrix.

    Returns:
        (tfidf_matrix, idf_vector) — both as float32 numpy arrays.
    """
    n_docs     = len(corpus)
    vocab_size = len(vocab)

    # Term frequency (normalised by document length)
    tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for doc_idx, tokens in enumerate(corpus):
        for token in tokens:
            if token in vocab:
                tf[doc_idx, vocab[token]] += 1
        total = tf[doc_idx].sum()
        if total > 0:
            tf[doc_idx] /= total

    # Inverse document frequency
    doc_freq = np.zeros(vocab_size, dtype=np.float32)
    for tokens in corpus:
        for token in set(tokens):
            if token in vocab:
                doc_freq[vocab[token]] += 1
    idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1.0

    # TF-IDF with L2 row normalisation
    tfidf = tf * idf
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tfidf / norms, idf
