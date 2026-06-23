"""
src/indexing/search_index/vector_index.py
------------------------------------------
ChromaDB-backed Vector Index for eUICC Knowledge Objects.

Drop-in replacement for HybridIndex that swaps the numpy TF-IDF matrix
for a persistent ChromaDB HNSW vector store.  BM25, graph, path lookup,
and RRF fusion logic are identical to HybridIndex.

What changed vs HybridIndex:
  BEFORE (HybridIndex)              AFTER (VectorIndex)
  TF-IDF sparse matrix (numpy)   →  Dense embeddings in ChromaDB
  Manual cosine similarity        →  ChromaDB HNSW index (automatic)
  4 MB .npy + 4 MB .json         →  One chroma_db/ folder (persistent)
  No metadata filtering           →  where={"ko_type": "ValidationRule"}
  Full matrix loaded into RAM     →  ChromaDB lazy-load from disk

What stayed the same (public API is identical):
  ✅ BM25 index (rank_bm25)
  ✅ Graph adjacency dict
  ✅ Path lookup (O(1) dict)
  ✅ RRF fusion logic
  ✅ search() / search_bm25() / lookup_path() / graph_neighbors() signatures

Embedding strategy:
  TFIDFEmbedder is used to produce vectors, making this environment-
  compatible without network access.  Swap TFIDFEmbedder for any
  sentence-transformer or API-backed embedder by replacing the
  embedder.fit() / embedder() calls in build() and upsert().
"""

import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

from .models import SearchResult
from .tokenizer import tokenize
from .embedder import TFIDFEmbedder
from ._metadata import flatten_metadata


class VectorIndex:
    """
    Hybrid index with ChromaDB as the vector store.

    Architecture:
      Query
        ├─► BM25 (rank_bm25, in-memory)         → keyword scores
        ├─► ChromaDB (persistent vector store)   → semantic scores
        └─► RRF Fusion                           → ranked results

      Direct lookups:
        lookup_path()     → Path index (dict, O(1))
        graph_neighbors() → Graph adjacency dict
    """

    COLLECTION_NAME = "euicc_knowledge"

    def __init__(self):
        self._kos: list[dict] = []
        self._ko_id_map: dict[str, int] = {}
        self._tokenized_corpus: list[list[str]] = []

        self._bm25: Optional[BM25Okapi] = None
        self._chroma_client = None
        self._collection    = None
        self._embedder: Optional[TFIDFEmbedder] = None

        self._graph: dict[str, list[dict]]    = {}
        self._path_index: dict[str, str]      = {}
        self._type_index: dict[str, list[int]]    = {}
        self._section_index: dict[str, list[int]] = {}

    # ─────────────────────────────────────────────────────────────────────────
    # Build
    # ─────────────────────────────────────────────────────────────────────────

    def build(
        self,
        knowledge_objects: list[dict],
        relationships: list[dict],
        persist_dir: Optional[str] = None,
    ) -> "VectorIndex":
        """
        Build all indexes.

        Args:
            persist_dir: Folder for ChromaDB persistent storage.
                         None = in-memory (lost when process ends).
        """
        self._kos       = knowledge_objects
        self._ko_id_map = {ko["ko_id"]: i for i, ko in enumerate(knowledge_objects)}
        self._tokenized_corpus = [
            tokenize(ko["text_content"] + " " + ko["primary_label"])
            for ko in knowledge_objects
        ]

        print("  [1/4] Building BM25 index...")
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        print("  [2/4] Fitting embedder...")
        corpus_texts = [
            ko["text_content"] + " " + ko["primary_label"]
            for ko in knowledge_objects
        ]
        self._embedder = TFIDFEmbedder()
        self._embedder.fit(corpus_texts)

        print("  [3/4] Building ChromaDB vector store...")
        self._init_chroma(persist_dir)
        self._populate_chroma(knowledge_objects, corpus_texts)

        print("  [4/4] Building graph + lookup indexes...")
        self._build_graph(relationships)
        self._build_path_index()
        self._build_filter_indexes()

        return self

    def _init_chroma(self, persist_dir: Optional[str]) -> None:
        if persist_dir:
            self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._chroma_client = chromadb.Client()

        try:
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        self._collection = self._chroma_client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _populate_chroma(
        self,
        knowledge_objects: list[dict],
        corpus_texts: list[str],
    ) -> None:
        BATCH_SIZE = 100
        total      = len(knowledge_objects)

        for batch_start in range(0, total, BATCH_SIZE):
            batch_kos   = knowledge_objects[batch_start:batch_start + BATCH_SIZE]
            batch_texts = corpus_texts[batch_start:batch_start + BATCH_SIZE]

            self._collection.add(
                ids        = [ko["ko_id"] for ko in batch_kos],
                documents  = batch_texts,
                embeddings = self._embedder(batch_texts),
                metadatas  = [
                    flatten_metadata({
                        "ko_type":       ko["ko_type"],
                        "primary_label": ko["primary_label"],
                        **ko["metadata"],
                    })
                    for ko in batch_kos
                ],
            )

        print(f"    Indexed {total} documents into ChromaDB")

    # ─────────────────────────────────────────────────────────────────────────
    # Graph + lookup indexes (same logic as HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_graph(self, relationships: list[dict]) -> None:
        for rel in relationships:
            src, tgt = rel["source_id"], rel["target_id"]
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
    # Query: path lookup (same as HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def lookup_path(self, error_path: str) -> Optional[dict]:
        normalized = self._normalize_error_path(error_path)

        ko_id = self._path_index.get(normalized)
        if not ko_id:
            parts = normalized.split(".")
            for start in range(1, len(parts)):
                ko_id = self._path_index.get(".".join(parts[start:]))
                if ko_id:
                    break
        if not ko_id:
            parts = normalized.split(".")
            for end in range(len(parts) - 1, 0, -1):
                ko_id = self._path_index.get(".".join(parts[:end]))
                if ko_id:
                    break
        if ko_id and ko_id in self._ko_id_map:
            return self._kos[self._ko_id_map[ko_id]]
        return None

    def _normalize_error_path(self, path: str) -> str:
        path = re.sub(r"\[\d+\]", "", path)
        path = re.sub(r"^ProfileElement\.", "", path, flags=re.I)
        path = re.sub(r"^Profile\s+Package\s+Rule\s+Set\.", "", path, flags=re.I)
        path = re.sub(r"^PE-[A-Za-z0-9]+\.", "", path)
        return ".".join(p.lower() for p in path.split(".") if p)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: BM25 (same as HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def search_bm25(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
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
    # Query: ChromaDB vector search
    # ─────────────────────────────────────────────────────────────────────────

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search via ChromaDB HNSW index.

        Supports metadata filtering (where={"ko_type": ...}) which is
        not available in HybridIndex's numpy brute-force approach.
        """
        query_embedding = self._embedder([query])[0]
        where_filter    = {"ko_type": ko_type} if ko_type else None

        chroma_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where_filter,
            include=["distances", "metadatas", "documents"],
        )

        results   = []
        ids       = chroma_results["ids"][0]
        distances = chroma_results["distances"][0]

        for ko_id, dist in zip(ids, distances):
            similarity = 1.0 - (dist / 2.0)
            if ko_id not in self._ko_id_map:
                continue
            ko = self._kos[self._ko_id_map[ko_id]]
            results.append(SearchResult(
                ko_id=ko_id, ko_type=ko["ko_type"],
                primary_label=ko["primary_label"],
                score=similarity, bm25_score=0.0, vector_score=similarity,
                metadata=ko["metadata"], text_content=ko["text_content"],
                relationships=ko.get("relationships", []),
            ))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Hybrid RRF (same fusion logic as HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 5,
        ko_type: Optional[str] = None,
        bm25_weight: float = 0.5,
        vector_weight: float = 0.5,
    ) -> list[SearchResult]:
        k_rrf = 60
        bm25_results   = self.search_bm25(query,   top_k=top_k * 3, ko_type=ko_type)
        vector_results = self.search_vector(query, top_k=top_k * 3, ko_type=ko_type)

        bm25_rank     = {r.ko_id: rank for rank, r in enumerate(bm25_results)}
        vector_rank   = {r.ko_id: rank for rank, r in enumerate(vector_results)}
        bm25_scores   = {r.ko_id: r.bm25_score   for r in bm25_results}
        vector_scores = {r.ko_id: r.vector_score  for r in vector_results}

        all_ids = set(bm25_rank) | set(vector_rank)
        rrf_scores = {
            ko_id: (
                bm25_weight   / (k_rrf + bm25_rank.get(ko_id,   len(self._kos))) +
                vector_weight / (k_rrf + vector_rank.get(ko_id, len(self._kos)))
            )
            for ko_id in all_ids
        }

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]
        results    = []
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
    # Graph traversal (same as HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def graph_neighbors(
        self,
        ko_id: str,
        rel_type: Optional[str] = None,
        direction: str = "both",
    ) -> list[dict]:
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
    # Incremental upsert (VectorIndex-only feature)
    # ─────────────────────────────────────────────────────────────────────────

    def upsert(self, knowledge_object: dict) -> None:
        """
        Add or update a single KO without rebuilding the entire index.
        This capability is not available in HybridIndex.
        """
        ko   = knowledge_object
        text = ko["text_content"] + " " + ko["primary_label"]
        self._collection.upsert(
            ids        = [ko["ko_id"]],
            documents  = [text],
            embeddings = [self._embedder([text])[0]],
            metadatas  = [flatten_metadata({
                "ko_type":       ko["ko_type"],
                "primary_label": ko["primary_label"],
                **ko["metadata"],
            })],
        )
        if ko["ko_id"] not in self._ko_id_map:
            self._ko_id_map[ko["ko_id"]] = len(self._kos)
            self._kos.append(ko)
            self._tokenized_corpus.append(tokenize(text))
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    # ─────────────────────────────────────────────────────────────────────────
    # Section-filtered vector search (VectorIndex-only feature)
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_section(
        self,
        query: str,
        section_id: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Vector search filtered to a single spec section."""
        query_embedding = self._embedder([query])[0]
        chroma_results  = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where={"section_id": section_id},
            include=["distances", "metadatas"],
        )
        results = []
        for ko_id, dist in zip(
            chroma_results["ids"][0], chroma_results["distances"][0]
        ):
            if ko_id not in self._ko_id_map:
                continue
            ko   = self._kos[self._ko_id_map[ko_id]]
            sim  = 1.0 - dist / 2.0
            results.append(SearchResult(
                ko_id=ko_id, ko_type=ko["ko_type"],
                primary_label=ko["primary_label"],
                score=sim, bm25_score=0.0, vector_score=sim,
                metadata=ko["metadata"], text_content=ko["text_content"],
            ))
        return sorted(results, key=lambda r: r.score, reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Persist / Load
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, output_dir: "str | Path") -> None:
        """
        Persist index to disk.

        Output layout:
          output_dir/
            chroma_db/          ← ChromaDB SQLite + HNSW index
            index_meta.json     ← KOs, graph, path index
            embedder_idf.npy    ← IDF weights for TFIDFEmbedder
            embedder_vocab.json ← Vocabulary mapping
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        meta = {
            "kos":              self._kos,
            "ko_id_map":        self._ko_id_map,
            "graph":            self._graph,
            "path_index":       self._path_index,
            "type_index":       self._type_index,
            "section_index":    self._section_index,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(out / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        self._embedder.save(out)

        chroma_dir = out / "chroma_db"
        if not chroma_dir.exists():
            self._persist_chroma_to_disk(chroma_dir)

        print(f"  Index saved to {out}/")
        print(f"    index_meta.json : {len(self._kos)} KOs")
        print(f"    chroma_db/      : ChromaDB persistent store")

    def _persist_chroma_to_disk(self, chroma_dir: Path) -> None:
        """Export an in-memory ChromaDB collection to disk."""
        chroma_dir.mkdir(exist_ok=True)
        persistent_client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            persistent_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        new_col  = persistent_client.create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        all_data = self._collection.get(
            include=["embeddings", "documents", "metadatas"]
        )
        if all_data["ids"]:
            ids, docs, embs, metas = (
                all_data["ids"], all_data["documents"],
                all_data["embeddings"], all_data["metadatas"],
            )
            BATCH_SIZE = 100
            for i in range(0, len(ids), BATCH_SIZE):
                new_col.add(
                    ids=ids[i:i + BATCH_SIZE],
                    documents=docs[i:i + BATCH_SIZE],
                    embeddings=embs[i:i + BATCH_SIZE],
                    metadatas=metas[i:i + BATCH_SIZE],
                )

    @classmethod
    def load(cls, index_dir: "str | Path") -> "VectorIndex":
        """
        Load a persisted index from disk.
        ChromaDB collection is connected lazily — not loaded into RAM.
        """
        idx_dir = Path(index_dir)
        index   = cls()

        with open(idx_dir / "index_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        index._kos              = meta["kos"]
        index._ko_id_map        = meta["ko_id_map"]
        index._graph            = meta["graph"]
        index._path_index       = meta["path_index"]
        index._type_index       = meta["type_index"]
        index._section_index    = meta["section_index"]
        index._tokenized_corpus = meta["tokenized_corpus"]

        index._embedder = TFIDFEmbedder.load(idx_dir)

        chroma_dir = idx_dir / "chroma_db"
        index._chroma_client = (
            chromadb.PersistentClient(path=str(chroma_dir))
            if chroma_dir.exists()
            else chromadb.Client()
        )
        index._collection = index._chroma_client.get_collection(cls.COLLECTION_NAME)

        index._bm25 = BM25Okapi(index._tokenized_corpus)

        print(f"  VectorIndex loaded from {idx_dir}/")
        print(f"    {len(index._kos)} KOs | ChromaDB: {index._collection.count()} docs")
        return index

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self) -> None:
        from collections import Counter
        print(f"  Knowledge objects  : {len(self._kos)}")
        print(f"  ChromaDB docs      : {self._collection.count()}")
        print(f"  Vocab size         : {len(self._embedder._vocab)}")
        print(f"  Graph nodes        : {len(self._graph)}")
        print(f"  Path index entries : {len(self._path_index)}")
        for ko_type, count in Counter(ko["ko_type"] for ko in self._kos).most_common():
            print(f"    {ko_type:<20} {count:>5}")
