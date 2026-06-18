# src/hybrid_index.py
"""
Hybrid Index for eUICC Knowledge Objects.

Three complementary indexes:
  1. BM25        — exact keyword / term-frequency search (rank_bm25)
  2. Vector DB   — semantic similarity via TF-IDF vectors (numpy, no API needed)
                   Swap-ready for pgvector / Pinecone by replacing _embed()
  3. Graph       — relationship traversal via adjacency dict

All three are built from knowledge_objects.json + relationships.json
and persisted to disk as a single index bundle (.json + .npy).

Query API:
  index.search(query, top_k=5)          → hybrid ranked results
  index.search_bm25(query, top_k=10)    → BM25 only
  index.search_vector(query, top_k=10)  → vector only
  index.graph_neighbors(ko_id, rel_type) → adjacent nodes
  index.lookup_path(error_path)         → direct path lookup
"""

import os

os.environ["ANONYMIZED_TELEMETRY"] = "False"

import json
import re
import math
from pathlib import Path

import numpy as np
import chromadb
import pickle

from dataclasses import dataclass, field
from typing import Optional

from rank_bm25 import BM25Okapi


# ─────────────────────────────────────────────────────────────────────────────
# Result model
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    ko_id: str
    ko_type: str
    primary_label: str
    score: float
    bm25_score: float
    vector_score: float
    metadata: dict
    text_content: str
    relationships: list = field(default_factory=list)

    def to_dict(self):
        return {
            "ko_id":         self.ko_id,
            "ko_type":       self.ko_type,
            "primary_label": self.primary_label,
            "score":         round(self.score, 4),
            "bm25_score":    round(self.bm25_score, 4),
            "vector_score":  round(self.vector_score, 4),
            "metadata":      self.metadata,
            "text_preview":  self.text_content[:300],
        }


# ─────────────────────────────────────────────────────────────────────────────
# Text utilities
# ─────────────────────────────────────────────────────────────────────────────

def tokenize(text: str) -> list[str]:
    """
    Tokenize text for BM25.
    Handles camelCase, kebab-case, dotted paths, and normal words.
    """
    text = text.lower()
    # Split camelCase: pukHeader → puk header
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    # Split on non-alphanumeric (hyphens, dots, underscores, spaces)
    tokens = re.split(r"[^a-z0-9]+", text)
    # Filter short tokens
    return [t for t in tokens if len(t) >= 2]


def build_tfidf_matrix(corpus: list[list[str]], vocab: dict) -> np.ndarray:
    """
    Build TF-IDF matrix (n_docs × vocab_size).
    Used for vector similarity without an external embedding API.
    """
    n_docs = len(corpus)
    vocab_size = len(vocab)

    # TF matrix
    tf = np.zeros((n_docs, vocab_size), dtype=np.float32)
    for doc_idx, tokens in enumerate(corpus):
        for token in tokens:
            if token in vocab:
                tf[doc_idx, vocab[token]] += 1
        # Normalize by doc length
        total = tf[doc_idx].sum()
        if total > 0:
            tf[doc_idx] /= total

    # IDF
    doc_freq = np.zeros(vocab_size, dtype=np.float32)
    for tokens in corpus:
        for token in set(tokens):
            if token in vocab:
                doc_freq[vocab[token]] += 1
    idf = np.log((n_docs + 1) / (doc_freq + 1)) + 1.0

    # TF-IDF
    tfidf = tf * idf
    # L2 normalize each row
    norms = np.linalg.norm(tfidf, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return tfidf / norms, idf


# ─────────────────────────────────────────────────────────────────────────────
# Hybrid Index
# ─────────────────────────────────────────────────────────────────────────────

class HybridIndex:
    """
    Hybrid BM25 + TF-IDF Vector + Graph index for eUICC knowledge objects.
    """

    def __init__(self):
        self._kos: list[dict] = []          # ordered list of KO dicts
        self._ko_id_map: dict[str, int] = {} # ko_id → index position

        # BM25
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []

        # Vector
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None

        # Graph: ko_id → list of {rel_type, target_id, source_id, properties}
        self._graph: dict[str, list[dict]] = {}

        # Path lookup: normalized_path → ko_id (for exact error path matching)
        self._path_index: dict[str, str] = {}

        # Label/metadata indexes for fast filter
        self._type_index: dict[str, list[int]] = {}   # ko_type → [positions]
        self._section_index: dict[str, list[int]] = {} # section_id → [positions]

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        knowledge_objects: list[dict],
        relationships: list[dict],
    ) -> "HybridIndex":
        """Build all three indexes from KO + relationship data."""

        print("  [1/4] Tokenizing corpus...")
        self._kos = knowledge_objects
        self._ko_id_map = {ko["ko_id"]: i for i, ko in enumerate(knowledge_objects)}
        self._tokenized_corpus = [
            tokenize(ko["text_content"] + " " + ko["primary_label"])
            for ko in knowledge_objects
        ]

        print("  [2/4] Building BM25 index...")
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        print("  [3/4] Building TF-IDF vector index...")
        # Build vocabulary from corpus
        all_tokens = set(t for doc in self._tokenized_corpus for t in doc)
        self._vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        self._tfidf_matrix, self._idf = build_tfidf_matrix(
            self._tokenized_corpus, self._vocab
        )

        print("  [4/4] Building graph + lookup indexes...")
        self._build_graph(relationships)
        self._build_path_index()
        self._build_filter_indexes()

        return self

    def _build_graph(self, relationships: list[dict]):
        """Build adjacency list from relationships."""
        for rel in relationships:
            src = rel["source_id"]
            tgt = rel["target_id"]
            edge = {
                "rel_id":    rel["rel_id"],
                "rel_type":  rel["rel_type"],
                "target_id": tgt,
                "source_id": src,
                "properties": rel.get("properties", {}),
            }
            if src not in self._graph:
                self._graph[src] = []
            self._graph[src].append(edge)

            # Also reverse edge for bidirectional traversal
            rev_edge = {**edge, "direction": "incoming"}
            if tgt not in self._graph:
                self._graph[tgt] = []
            self._graph[tgt].append({**edge, "target_id": src, "source_id": tgt,
                                      "direction": "incoming"})

    def _build_path_index(self):
        """Index ValidationRule KOs by normalized_path for O(1) error lookup."""
        for ko in self._kos:
            if ko["ko_type"] == "ValidationRule":
                path = ko["metadata"].get("normalized_path", "")
                if path:
                    self._path_index[path] = ko["ko_id"]
                    # Also index variants: strip dots, underscores
                    flat = path.replace(".", "_").replace("-", "_")
                    self._path_index[flat] = ko["ko_id"]

    def _build_filter_indexes(self):
        """Build type and section filter indexes."""
        for i, ko in enumerate(self._kos):
            ko_type = ko["ko_type"]
            if ko_type not in self._type_index:
                self._type_index[ko_type] = []
            self._type_index[ko_type].append(i)

            sec_id = ko["metadata"].get("section_id", "")
            if sec_id:
                if sec_id not in self._section_index:
                    self._section_index[sec_id] = []
                self._section_index[sec_id].append(i)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: error path → direct lookup
    # ─────────────────────────────────────────────────────────────────────────

    def lookup_path(self, error_path: str) -> Optional[dict]:
        """
        Direct lookup by error path from validator output.
        Handles formats like:
          "ProfileElement[3].pukCodes.puk_Header.Identification"
          "Profile Package Rule Set.pukCodes.puk_Header.Identification"
        Tries progressively shorter suffix paths for partial matches.
        Returns the KO dict or None.
        """
        normalized = self._normalize_error_path(error_path)

        # 1. Exact match
        ko_id = self._path_index.get(normalized)
        if ko_id:
            return self._kos[self._ko_id_map[ko_id]] if ko_id in self._ko_id_map else None

        # 2. Suffix reduction — drop leading parts one at a time
        parts = normalized.split(".")
        for start in range(1, len(parts)):
            candidate = ".".join(parts[start:])
            ko_id = self._path_index.get(candidate)
            if ko_id and ko_id in self._ko_id_map:
                return self._kos[self._ko_id_map[ko_id]]

        # 3. Prefix reduction — try without last part (e.g. strip .Identification)
        for end in range(len(parts) - 1, 0, -1):
            candidate = ".".join(parts[:end])
            ko_id = self._path_index.get(candidate)
            if ko_id and ko_id in self._ko_id_map:
                return self._kos[self._ko_id_map[ko_id]]

        return None

    def _normalize_error_path(self, path: str) -> str:
        """
        Normalize validator error path to match our index format.
        Examples:
          "ProfileElement[3].pukCodes.puk_Header.Identification"
          → "pukcodes.puk_header"  (match on prefix)
          "Profile Package Rule Set.pukCodes.puk_Header.Identification"
          → "pukcodes.puk_header"
        """
        # Remove array indices
        path = re.sub(r"\[\d+\]", "", path)
        # Remove known prefix patterns
        path = re.sub(r"^ProfileElement\.", "", path, flags=re.I)
        path = re.sub(r"^Profile\s+Package\s+Rule\s+Set\.", "", path, flags=re.I)
        path = re.sub(r"^PE-[A-Za-z0-9]+\.", "", path)
        # Split on dots only (preserve underscores within each part)
        parts = path.split(".")
        return ".".join(p.lower() for p in parts if p)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: BM25 search
    # ─────────────────────────────────────────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """BM25 keyword search with optional type filter."""
        q_tokens = tokenize(query)
        raw_scores = self._bm25.get_scores(q_tokens)

        # Apply type filter
        candidate_indices = self._type_index.get(ko_type, range(len(self._kos))) \
            if ko_type else range(len(self._kos))

        results = []
        for i in candidate_indices:
            score = float(raw_scores[i])
            if score > 0:
                ko = self._kos[i]
                results.append(SearchResult(
                    ko_id=ko["ko_id"],
                    ko_type=ko["ko_type"],
                    primary_label=ko["primary_label"],
                    score=score,
                    bm25_score=score,
                    vector_score=0.0,
                    metadata=ko["metadata"],
                    text_content=ko["text_content"],
                    relationships=ko.get("relationships", []),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Vector search
    # ─────────────────────────────────────────────────────────────────────────

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """TF-IDF vector search with cosine similarity."""
        q_tokens = tokenize(query)

        # Build query vector
        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        for token in q_tokens:
            if token in self._vocab:
                q_vec[self._vocab[token]] += 1
        # Apply IDF
        if q_vec.sum() > 0:
            q_vec = q_vec * self._idf
            norm = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm

        # Cosine similarity (matrix × query vector)
        scores = self._tfidf_matrix @ q_vec  # shape: (n_docs,)

        # Apply type filter
        candidate_indices = self._type_index.get(ko_type, range(len(self._kos))) \
            if ko_type else range(len(self._kos))

        results = []
        for i in candidate_indices:
            score = float(scores[i])
            if score > 0.01:
                ko = self._kos[i]
                results.append(SearchResult(
                    ko_id=ko["ko_id"],
                    ko_type=ko["ko_type"],
                    primary_label=ko["primary_label"],
                    score=score,
                    bm25_score=0.0,
                    vector_score=score,
                    metadata=ko["metadata"],
                    text_content=ko["text_content"],
                    relationships=ko.get("relationships", []),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Hybrid search (BM25 + Vector, RRF fusion)
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        ko_type: Optional[str] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[SearchResult]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF).
        Combines BM25 and vector rankings into a single score.
        """
        k_rrf = 60  # RRF constant

        bm25_results  = self.search_bm25(query, top_k=top_k * 3, ko_type=ko_type)
        vector_results = self.search_vector(query, top_k=top_k * 3, ko_type=ko_type)

        # Build rank maps
        bm25_rank  = {r.ko_id: rank for rank, r in enumerate(bm25_results)}
        vector_rank = {r.ko_id: rank for rank, r in enumerate(vector_results)}

        # Raw score maps for reporting
        bm25_scores   = {r.ko_id: r.bm25_score for r in bm25_results}
        vector_scores = {r.ko_id: r.vector_score for r in vector_results}

        # Union of all candidates
        all_ids = set(bm25_rank) | set(vector_rank)

        # RRF score
        rrf_scores = {}
        for ko_id in all_ids:
            b_rank = bm25_rank.get(ko_id, len(self._kos))
            v_rank = vector_rank.get(ko_id, len(self._kos))
            rrf_scores[ko_id] = (
                bm25_weight  / (k_rrf + b_rank) +
                vector_weight / (k_rrf + v_rank)
            )

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for ko_id in sorted_ids:
            if ko_id not in self._ko_id_map:
                continue
            ko = self._kos[self._ko_id_map[ko_id]]
            results.append(SearchResult(
                ko_id=ko_id,
                ko_type=ko["ko_type"],
                primary_label=ko["primary_label"],
                score=rrf_scores[ko_id],
                bm25_score=bm25_scores.get(ko_id, 0.0),
                vector_score=vector_scores.get(ko_id, 0.0),
                metadata=ko["metadata"],
                text_content=ko["text_content"],
                relationships=ko.get("relationships", []),
            ))

        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Graph traversal
    # ─────────────────────────────────────────────────────────────────────────

    def graph_neighbors(
        self,
        ko_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",  # "out" | "in" | "both"
    ) -> list[dict]:
        """
        Return adjacent nodes in the knowledge graph.
        Optionally filter by relationship type and direction.
        """
        edges = self._graph.get(ko_id, [])
        result = []
        for edge in edges:
            # Direction filter
            edge_dir = edge.get("direction", "outgoing")
            if direction == "out" and edge_dir == "incoming":
                continue
            if direction == "in" and edge_dir != "incoming":
                continue
            # Type filter
            if rel_type and edge["rel_type"] != rel_type:
                continue
            result.append(edge)
        return result

    def graph_expand(
        self,
        ko_id: str,
        rel_types: Optional[list[str]] = None,
        depth: int = 2,
    ) -> dict[str, list[dict]]:
        """
        Multi-hop graph expansion from a starting node.
        Returns {ko_id: [edges]} for all reachable nodes within `depth` hops.
        """
        visited = {}
        frontier = {ko_id}

        for _ in range(depth):
            next_frontier = set()
            for node_id in frontier:
                if node_id in visited:
                    continue
                edges = self.graph_neighbors(node_id, direction="out")
                if rel_types:
                    edges = [e for e in edges if e["rel_type"] in rel_types]
                visited[node_id] = edges
                for edge in edges:
                    tgt = edge["target_id"]
                    if tgt not in visited:
                        next_frontier.add(tgt)
            frontier = next_frontier

        return visited

    # ─────────────────────────────────────────────────────────────────────────
    # Persist / Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path):
        """Persist index to disk."""
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        # Save everything except numpy matrix as JSON
        meta = {
            "kos":          self._kos,
            "ko_id_map":    self._ko_id_map,
            "vocab":        self._vocab,
            "graph":        self._graph,
            "path_index":   self._path_index,
            "type_index":   self._type_index,
            "section_index": self._section_index,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(out / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        # Save numpy arrays
        np.save(str(out / "tfidf_matrix.npy"), self._tfidf_matrix)
        np.save(str(out / "idf.npy"), self._idf)

        print(f"  Index saved to {out}/")
        print(f"    index_meta.json : {len(self._kos)} KOs, {len(self._graph)} graph nodes")
        print(f"    tfidf_matrix.npy: {self._tfidf_matrix.shape}")

    @classmethod
    def load(cls, index_dir: str | Path) -> "HybridIndex":
        """Load persisted index from disk."""
        idx_dir = Path(index_dir)
        index = cls()

        with open(idx_dir / "index_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        index._kos               = meta["kos"]
        index._ko_id_map         = meta["ko_id_map"]
        index._vocab             = meta["vocab"]
        index._graph             = meta["graph"]
        index._path_index        = meta["path_index"]
        index._type_index        = meta["type_index"]
        index._section_index     = meta["section_index"]
        index._tokenized_corpus  = meta["tokenized_corpus"]
        index._tfidf_matrix      = np.load(str(idx_dir / "tfidf_matrix.npy"))
        index._idf               = np.load(str(idx_dir / "idf.npy"))

        # Rebuild BM25 (not serializable)
        index._bm25 = BM25Okapi(index._tokenized_corpus)

        print(f"  Index loaded from {idx_dir}/")
        print(f"    {len(index._kos)} KOs | vocab size: {len(index._vocab)}")
        return index

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self):
        print(f"  Knowledge objects : {len(self._kos)}")
        print(f"  Vocabulary size   : {len(self._vocab)}")
        print(f"  TF-IDF matrix     : {self._tfidf_matrix.shape}")
        print(f"  Graph nodes       : {len(self._graph)}")
        print(f"  Path index entries: {len(self._path_index)}")
        from collections import Counter
        type_counts = Counter(ko["ko_type"] for ko in self._kos)
        for ko_type, count in type_counts.most_common():
            print(f"    {ko_type:<20} {count:>5}")