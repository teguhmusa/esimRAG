"""
src/indexing/search_index/hybrid_index.py
------------------------------------------
Hybrid BM25 + TF-IDF Vector + Graph index for eUICC Knowledge Objects.

Three complementary indexes:
  1. BM25       — exact keyword / term-frequency search (rank_bm25)
  2. TF-IDF     — sparse vector similarity (numpy, no external API needed)
  3. Graph      — relationship traversal via adjacency dict

All three are built from knowledge_objects and relationships data
and persisted to disk as a single bundle (index_meta.json + .npy files).

Query API:
  index.search(query, top_k=5)           → hybrid ranked results (RRF)
  index.search_bm25(query, top_k=10)     → BM25 only
  index.search_vector(query, top_k=10)   → TF-IDF vector only
  index.graph_neighbors(ko_id, rel_type) → adjacent graph nodes
  index.lookup_path(error_path)          → direct path lookup
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np
from rank_bm25 import BM25Okapi

from .models import SearchResult
from .tokenizer import tokenize, build_tfidf_matrix


class HybridIndex:
    """
    Hybrid BM25 + TF-IDF Vector + Graph index for eUICC Knowledge Objects.
    """

    def __init__(self):
        self._kos: list[dict] = []
        self._ko_id_map: dict[str, int] = {}

        # BM25
        self._bm25: Optional[BM25Okapi] = None
        self._tokenized_corpus: list[list[str]] = []

        # TF-IDF vector
        self._tfidf_matrix: Optional[np.ndarray] = None
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None

        # Graph: ko_id → list of edge dicts
        self._graph: dict[str, list[dict]] = {}

        # Path lookup: normalized_path → ko_id
        self._path_index: dict[str, str] = {}

        # Filter indexes
        self._type_index: dict[str, list[int]] = {}
        self._section_index: dict[str, list[int]] = {}

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
        self._kos       = knowledge_objects
        self._ko_id_map = {ko["ko_id"]: i for i, ko in enumerate(knowledge_objects)}
        self._tokenized_corpus = [
            tokenize(ko["text_content"] + " " + ko["primary_label"])
            for ko in knowledge_objects
        ]

        print("  [2/4] Building BM25 index...")
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        print("  [3/4] Building TF-IDF vector index...")
        all_tokens  = {t for doc in self._tokenized_corpus for t in doc}
        self._vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        self._tfidf_matrix, self._idf = build_tfidf_matrix(
            self._tokenized_corpus, self._vocab
        )

        print("  [4/4] Building graph + lookup indexes...")
        self._build_graph(relationships)
        self._build_path_index()
        self._build_filter_indexes()

        return self

    def _build_graph(self, relationships: list[dict]) -> None:
        for rel in relationships:
            src  = rel["source_id"]
            tgt  = rel["target_id"]
            edge = {
                "rel_id":     rel["rel_id"],
                "rel_type":   rel["rel_type"],
                "target_id":  tgt,
                "source_id":  src,
                "properties": rel.get("properties", {}),
            }
            self._graph.setdefault(src, []).append(edge)
            self._graph.setdefault(tgt, []).append(
                {**edge, "target_id": src, "source_id": tgt, "direction": "incoming"}
            )

    def _build_path_index(self) -> None:
        for ko in self._kos:
            if ko["ko_type"] == "ValidationRule":
                path = ko["metadata"].get("normalized_path", "")
                if path:
                    self._path_index[path] = ko["ko_id"]
                    flat = path.replace(".", "_").replace("-", "_")
                    self._path_index[flat] = ko["ko_id"]

    def _build_filter_indexes(self) -> None:
        for i, ko in enumerate(self._kos):
            self._type_index.setdefault(ko["ko_type"], []).append(i)
            sec_id = ko["metadata"].get("section_id", "")
            if sec_id:
                self._section_index.setdefault(sec_id, []).append(i)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: error path → direct lookup
    # ─────────────────────────────────────────────────────────────────────────

    def lookup_path(self, error_path: str) -> Optional[dict]:
        """
        Direct O(1) lookup by validator error path.

        Tries progressively shorter suffix / prefix paths for partial matches.
        Accepts formats such as:
          "ProfileElement[3].pukCodes.puk_Header.Identification"
          "Profile Package Rule Set.pukCodes.puk_Header.Identification"
        """
        normalized = self._normalize_error_path(error_path)

        ko_id = self._path_index.get(normalized)
        if ko_id:
            return self._kos[self._ko_id_map[ko_id]] if ko_id in self._ko_id_map else None

        parts = normalized.split(".")
        for start in range(1, len(parts)):
            ko_id = self._path_index.get(".".join(parts[start:]))
            if ko_id and ko_id in self._ko_id_map:
                return self._kos[self._ko_id_map[ko_id]]

        for end in range(len(parts) - 1, 0, -1):
            ko_id = self._path_index.get(".".join(parts[:end]))
            if ko_id and ko_id in self._ko_id_map:
                return self._kos[self._ko_id_map[ko_id]]

        return None

    def _normalize_error_path(self, path: str) -> str:
        import re
        path = re.sub(r"\[\d+\]", "", path)
        path = re.sub(r"^ProfileElement\.", "", path, flags=re.I)
        path = re.sub(r"^Profile\s+Package\s+Rule\s+Set\.", "", path, flags=re.I)
        path = re.sub(r"^PE-[A-Za-z0-9]+\.", "", path)
        return ".".join(p.lower() for p in path.split(".") if p)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: BM25
    # ─────────────────────────────────────────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """BM25 keyword search with optional type filter."""
        q_tokens   = tokenize(query)
        raw_scores = self._bm25.get_scores(q_tokens)

        candidate_indices = (
            self._type_index.get(ko_type, range(len(self._kos)))
            if ko_type else range(len(self._kos))
        )

        results = []
        for i in candidate_indices:
            score = float(raw_scores[i])
            if score > 0:
                ko = self._kos[i]
                results.append(SearchResult(
                    ko_id=ko["ko_id"], ko_type=ko["ko_type"],
                    primary_label=ko["primary_label"],
                    score=score, bm25_score=score, vector_score=0.0,
                    metadata=ko["metadata"], text_content=ko["text_content"],
                    relationships=ko.get("relationships", []),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Query: TF-IDF vector
    # ─────────────────────────────────────────────────────────────────────────

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """TF-IDF vector search with cosine similarity."""
        q_tokens = tokenize(query)

        q_vec = np.zeros(len(self._vocab), dtype=np.float32)
        for token in q_tokens:
            if token in self._vocab:
                q_vec[self._vocab[token]] += 1
        if q_vec.sum() > 0:
            q_vec = q_vec * self._idf
            norm  = np.linalg.norm(q_vec)
            if norm > 0:
                q_vec /= norm

        scores = self._tfidf_matrix @ q_vec

        candidate_indices = (
            self._type_index.get(ko_type, range(len(self._kos)))
            if ko_type else range(len(self._kos))
        )

        results = []
        for i in candidate_indices:
            score = float(scores[i])
            if score > 0.01:
                ko = self._kos[i]
                results.append(SearchResult(
                    ko_id=ko["ko_id"], ko_type=ko["ko_type"],
                    primary_label=ko["primary_label"],
                    score=score, bm25_score=0.0, vector_score=score,
                    metadata=ko["metadata"], text_content=ko["text_content"],
                    relationships=ko.get("relationships", []),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Hybrid (BM25 + Vector, RRF fusion)
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
        k_rrf = 60

        bm25_results   = self.search_bm25(query,   top_k=top_k * 3, ko_type=ko_type)
        vector_results = self.search_vector(query, top_k=top_k * 3, ko_type=ko_type)

        bm25_rank    = {r.ko_id: rank for rank, r in enumerate(bm25_results)}
        vector_rank  = {r.ko_id: rank for rank, r in enumerate(vector_results)}
        bm25_scores  = {r.ko_id: r.bm25_score   for r in bm25_results}
        vector_scores = {r.ko_id: r.vector_score for r in vector_results}

        all_ids = set(bm25_rank) | set(vector_rank)
        rrf_scores = {
            ko_id: (
                bm25_weight   / (k_rrf + bm25_rank.get(ko_id,   len(self._kos))) +
                vector_weight / (k_rrf + vector_rank.get(ko_id, len(self._kos)))
            )
            for ko_id in all_ids
        }

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        results = []
        for ko_id in sorted_ids:
            if ko_id not in self._ko_id_map:
                continue
            ko = self._kos[self._ko_id_map[ko_id]]
            results.append(SearchResult(
                ko_id=ko_id, ko_type=ko["ko_type"],
                primary_label=ko["primary_label"],
                score=rrf_scores[ko_id],
                bm25_score=bm25_scores.get(ko_id, 0.0),
                vector_score=vector_scores.get(ko_id, 0.0),
                metadata=ko["metadata"], text_content=ko["text_content"],
                relationships=ko.get("relationships", []),
            ))
        return results

    # ─────────────────────────────────────────────────────────────────────────
    # Graph traversal
    # ─────────────────────────────────────────────────────────────────────────

    def graph_neighbors(
        self,
        ko_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
    ) -> list[dict]:
        """
        Return adjacent nodes in the knowledge graph.
        direction: "out" | "in" | "both"
        """
        edges  = self._graph.get(ko_id, [])
        result = []
        for edge in edges:
            edge_dir = edge.get("direction", "outgoing")
            if direction == "out" and edge_dir == "incoming":
                continue
            if direction == "in" and edge_dir != "incoming":
                continue
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
        """Multi-hop graph expansion from a starting node."""
        visited:  dict[str, list[dict]] = {}
        frontier: set[str] = {ko_id}

        for _ in range(depth):
            next_frontier: set[str] = set()
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

    def save(self, output_dir: "str | Path") -> None:
        """Persist index to disk."""
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        meta = {
            "kos":               self._kos,
            "ko_id_map":         self._ko_id_map,
            "vocab":             self._vocab,
            "graph":             self._graph,
            "path_index":        self._path_index,
            "type_index":        self._type_index,
            "section_index":     self._section_index,
            "tokenized_corpus":  self._tokenized_corpus,
        }
        with open(out / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        np.save(str(out / "tfidf_matrix.npy"), self._tfidf_matrix)
        np.save(str(out / "idf.npy"),           self._idf)

        print(f"  Index saved to {out}/")
        print(f"    index_meta.json : {len(self._kos)} KOs, {len(self._graph)} graph nodes")
        print(f"    tfidf_matrix.npy: {self._tfidf_matrix.shape}")

    @classmethod
    def load(cls, index_dir: "str | Path") -> "HybridIndex":
        """Load persisted index from disk."""
        idx_dir = Path(index_dir)
        index   = cls()

        with open(idx_dir / "index_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        index._kos              = meta["kos"]
        index._ko_id_map        = meta["ko_id_map"]
        index._vocab            = meta["vocab"]
        index._graph            = meta["graph"]
        index._path_index       = meta["path_index"]
        index._type_index       = meta["type_index"]
        index._section_index    = meta["section_index"]
        index._tokenized_corpus = meta["tokenized_corpus"]
        index._tfidf_matrix     = np.load(str(idx_dir / "tfidf_matrix.npy"))
        index._idf              = np.load(str(idx_dir / "idf.npy"))

        index._bm25 = BM25Okapi(index._tokenized_corpus)

        print(f"  Index loaded from {idx_dir}/")
        print(f"    {len(index._kos)} KOs | vocab size: {len(index._vocab)}")
        return index

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> None:
        from collections import Counter
        print(f"  Knowledge objects : {len(self._kos)}")
        print(f"  Vocabulary size   : {len(self._vocab)}")
        print(f"  TF-IDF matrix     : {self._tfidf_matrix.shape}")
        print(f"  Graph nodes       : {len(self._graph)}")
        print(f"  Path index entries: {len(self._path_index)}")
        for ko_type, count in Counter(ko["ko_type"] for ko in self._kos).most_common():
            print(f"    {ko_type:<20} {count:>5}")
