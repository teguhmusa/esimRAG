# src/vector_index.py
"""
ChromaDB Vector Index untuk eUICC Knowledge Objects.

PERUBAHAN DARI hybrid_index.py:
═══════════════════════════════════════════════════════════════════════

  SEBELUM (HybridIndex)              SESUDAH (VectorIndex)
  ─────────────────────              ─────────────────────────────────
  TF-IDF sparse matrix (numpy)   →  Dense embeddings di ChromaDB
  Manual cosine similarity        →  ChromaDB HNSW index (otomatis)
  4MB .npy file + 4MB .json       →  Satu folder chroma_db/ (persistent)
  Vocab size: 1033 tokens         →  384 dimensi (all-MiniLM-L6-v2 shape)
  Tidak bisa filter by metadata   →  where={"ko_type": "ValidationRule"}
  Harus load semua ke RAM         →  ChromaDB lazy-load dari disk

  YANG TETAP SAMA:
  ─────────────────────────────────────────────────────────────────────
  ✅ BM25 index (rank_bm25)         — tidak berubah
  ✅ Graph adjacency dict           — tidak berubah
  ✅ Path lookup (O(1) dict)        — tidak berubah
  ✅ RRF fusion logic               — tidak berubah
  ✅ SearchResult dataclass         — tidak berubah
  ✅ Public API: search(), search_bm25(), lookup_path(), graph_neighbors()
     — tidak berubah, drop-in replacement

  EMBEDDING STRATEGY:
  ─────────────────────────────────────────────────────────────────────
  Karena environment ini tidak bisa download model ONNX dari internet,
  kita gunakan TF-IDF vectors kita sendiri sebagai embeddings ke ChromaDB.
  
  Ini tetap better dari pure numpy karena:
    1. Persistent storage — tidak perlu rebuild setiap run
    2. Metadata filtering — query dengan filter section_id, ko_type
    3. Incremental upsert — tambah KO baru tanpa rebuild seluruh matrix
    4. ChromaDB HNSW ANN — lebih cepat dari brute-force numpy @ pada scale besar
  
  Saat deploy ke environment dengan akses internet, cukup swap
  _embed() untuk pakai sentence-transformers atau OpenAI embeddings.
"""

import json
import re
import math
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import chromadb
from rank_bm25 import BM25Okapi

# Reuse tokenizer dan TF-IDF dari hybrid_index
from .hybrid_index import tokenize, build_tfidf_matrix, SearchResult


# ─────────────────────────────────────────────────────────────────────────────
# Embedding function
# ─────────────────────────────────────────────────────────────────────────────

class TFIDFEmbedder:
    """
    Embedding function berbasis TF-IDF yang bisa digunakan sebagai
    drop-in sebelum swap ke sentence-transformers atau OpenAI.

    Interface sama dengan ChromaDB EmbeddingFunction:
      __call__(input: list[str]) → list[list[float]]
    """

    def __init__(self):
        self._vocab: dict[str, int] = {}
        self._idf: Optional[np.ndarray] = None
        self._fitted = False

    def fit(self, corpus: list[str]):
        """Build vocabulary dan IDF dari corpus."""
        tokenized = [tokenize(text) for text in corpus]
        all_tokens = set(t for doc in tokenized for t in doc)
        self._vocab = {token: i for i, token in enumerate(sorted(all_tokens))}
        _, self._idf = build_tfidf_matrix(tokenized, self._vocab)
        self._fitted = True
        return self

    def __call__(self, input: list[str]) -> list[list[float]]:
        """Embed list of strings → list of float vectors."""
        if not self._fitted:
            raise RuntimeError("Embedder belum di-fit. Panggil fit() terlebih dahulu.")

        result = []
        for text in input:
            tokens = tokenize(text)
            vec = np.zeros(len(self._vocab), dtype=np.float32)
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

    def save(self, path: Path):
        np.save(str(path / "embedder_idf.npy"), self._idf)
        with open(path / "embedder_vocab.json", "w") as f:
            json.dump(self._vocab, f)

    @classmethod
    def load(cls, path: Path) -> "TFIDFEmbedder":
        obj = cls()
        obj._idf   = np.load(str(path / "embedder_idf.npy"))
        with open(path / "embedder_vocab.json") as f:
            obj._vocab = json.load(f)
        obj._fitted = True
        return obj


# ─────────────────────────────────────────────────────────────────────────────
# ChromaDB metadata helper
# ─────────────────────────────────────────────────────────────────────────────

def _flatten_metadata(metadata: dict) -> dict:
    """
    ChromaDB hanya support metadata values: str, int, float, bool.
    Flatten list/None values ke string.
    """
    flat = {}
    for k, v in metadata.items():
        if v is None:
            flat[k] = ""
        elif isinstance(v, list):
            flat[k] = json.dumps(v)  # serialize list → JSON string
        elif isinstance(v, (str, int, float, bool)):
            flat[k] = v
        else:
            flat[k] = str(v)
    return flat


# ─────────────────────────────────────────────────────────────────────────────
# VectorIndex — drop-in replacement untuk HybridIndex._tfidf_matrix
# ─────────────────────────────────────────────────────────────────────────────

class VectorIndex:
    """
    Hybrid index dengan ChromaDB sebagai vector store.

    Arsitektur:
      ┌─────────────────────────────────────────────────────┐
      │  Query                                              │
      │    │                                                │
      │    ├─► BM25 (rank_bm25, in-memory)                 │
      │    │     ↓ keyword scores                           │
      │    │                                                │
      │    ├─► ChromaDB (persistent vector store)           │
      │    │     ↓ semantic scores                          │
      │    │                                                │
      │    └─► RRF Fusion → ranked results                  │
      │                                                     │
      │  Direct Lookups:                                    │
      │    lookup_path()     → Path index (dict, O(1))      │
      │    graph_neighbors() → Graph index (dict)           │
      └─────────────────────────────────────────────────────┘
    """

    COLLECTION_NAME = "euicc_knowledge"

    def __init__(self):
        self._kos: list[dict] = []
        self._ko_id_map: dict[str, int] = {}
        self._tokenized_corpus: list[list[str]] = []

        # BM25 — tidak berubah
        self._bm25: Optional[BM25Okapi] = None

        # ChromaDB — BARU (menggantikan numpy TF-IDF matrix)
        self._chroma_client = None
        self._collection = None
        self._embedder: Optional[TFIDFEmbedder] = None

        # Graph, path, filter indexes — tidak berubah
        self._graph: dict[str, list[dict]] = {}
        self._path_index: dict[str, str] = {}
        self._type_index: dict[str, list[int]] = {}
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
        Build semua indexes.
        persist_dir: folder untuk ChromaDB persistent storage.
                     None = in-memory (hilang saat proses selesai).
        """
        self._kos = knowledge_objects
        self._ko_id_map = {ko["ko_id"]: i for i, ko in enumerate(knowledge_objects)}
        self._tokenized_corpus = [
            tokenize(ko["text_content"] + " " + ko["primary_label"])
            for ko in knowledge_objects
        ]

        # [1] BM25 — sama persis seperti sebelumnya
        print("  [1/4] Building BM25 index...")
        self._bm25 = BM25Okapi(self._tokenized_corpus)

        # [2] Embedder — fit TF-IDF vocab (bisa swap ke sentence-transformers)
        print("  [2/4] Fitting embedder...")
        corpus_texts = [ko["text_content"] + " " + ko["primary_label"]
                        for ko in knowledge_objects]
        self._embedder = TFIDFEmbedder()
        self._embedder.fit(corpus_texts)

        # [3] ChromaDB — BARU
        print("  [3/4] Building ChromaDB vector store...")
        self._init_chroma(persist_dir)
        self._populate_chroma(knowledge_objects, corpus_texts)

        # [4] Graph + lookup indexes — sama persis seperti sebelumnya
        print("  [4/4] Building graph + lookup indexes...")
        self._build_graph(relationships)
        self._build_path_index()
        self._build_filter_indexes()

        return self

    def _init_chroma(self, persist_dir: Optional[str]):
        """Initialize ChromaDB client — persistent atau in-memory."""
        if persist_dir:
            self._chroma_client = chromadb.PersistentClient(path=str(persist_dir))
        else:
            self._chroma_client = chromadb.Client()

        # Delete existing collection jika ada (untuk rebuild)
        try:
            self._chroma_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass

        # Buat collection dengan cosine distance
        # embedding_function=None karena kita supply embeddings manual
        self._collection = self._chroma_client.create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    def _populate_chroma(self, knowledge_objects: list[dict], corpus_texts: list[str]):
        """
        Masukkan semua KOs ke ChromaDB dengan pre-computed embeddings.
        Batch per 100 untuk efisiensi.
        """
        BATCH_SIZE = 100
        total = len(knowledge_objects)

        for batch_start in range(0, total, BATCH_SIZE):
            batch_kos   = knowledge_objects[batch_start:batch_start + BATCH_SIZE]
            batch_texts = corpus_texts[batch_start:batch_start + BATCH_SIZE]

            ids         = [ko["ko_id"] for ko in batch_kos]
            documents   = batch_texts
            embeddings  = self._embedder(batch_texts)
            metadatas   = [
                _flatten_metadata({
                    "ko_type":       ko["ko_type"],
                    "primary_label": ko["primary_label"],
                    **ko["metadata"],
                })
                for ko in batch_kos
            ]

            self._collection.add(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        print(f"    Indexed {total} documents into ChromaDB")

    # ─────────────────────────────────────────────────────────────────────────
    # Graph + lookup — identik dengan HybridIndex
    # ─────────────────────────────────────────────────────────────────────────

    def _build_graph(self, relationships: list[dict]):
        for rel in relationships:
            src, tgt = rel["source_id"], rel["target_id"]
            edge = {
                "rel_id": rel["rel_id"], "rel_type": rel["rel_type"],
                "target_id": tgt, "source_id": src,
                "properties": rel.get("properties", {}),
            }
            self._graph.setdefault(src, []).append(edge)
            self._graph.setdefault(tgt, []).append(
                {**edge, "target_id": src, "source_id": tgt, "direction": "incoming"}
            )

    def _build_path_index(self):
        for ko in self._kos:
            if ko["ko_type"] == "ValidationRule":
                path = ko["metadata"].get("normalized_path", "")
                if path:
                    self._path_index[path] = ko["ko_id"]
                    flat = path.replace(".", "_").replace("-", "_")
                    self._path_index[flat] = ko["ko_id"]

    def _build_filter_indexes(self):
        for i, ko in enumerate(self._kos):
            self._type_index.setdefault(ko["ko_type"], []).append(i)
            sec_id = ko["metadata"].get("section_id", "")
            if sec_id:
                self._section_index.setdefault(sec_id, []).append(i)

    # ─────────────────────────────────────────────────────────────────────────
    # Query: Path lookup — identik
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
    # Query: BM25 — identik
    # ─────────────────────────────────────────────────────────────────────────

    def search_bm25(
        self, query: str, top_k: int = 10, ko_type: Optional[str] = None
    ) -> list[SearchResult]:
        q_tokens = tokenize(query)
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
    # Query: Vector (ChromaDB) — BARU, menggantikan numpy @ matrix
    # ─────────────────────────────────────────────────────────────────────────

    def search_vector(
        self,
        query: str,
        top_k: int = 10,
        ko_type: Optional[str] = None,
    ) -> list[SearchResult]:
        """
        Semantic search via ChromaDB.

        PERBEDAAN dari HybridIndex.search_vector():
          Sebelum: tfidf_matrix @ query_vec  (brute-force numpy, O(n × vocab))
          Sesudah: collection.query()        (ChromaDB HNSW ANN, O(log n))

        Tambahan: support metadata filter (where={"ko_type": ...})
        """
        query_embedding = self._embedder([query])[0]

        # Metadata filter — fitur baru yang tidak ada di TF-IDF numpy
        where_filter = {"ko_type": ko_type} if ko_type else None

        chroma_results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where=where_filter,
            include=["distances", "metadatas", "documents"],
        )

        results = []
        ids        = chroma_results["ids"][0]
        distances  = chroma_results["distances"][0]
        metadatas  = chroma_results["metadatas"][0]

        for ko_id, dist, meta in zip(ids, distances, metadatas):
            # ChromaDB cosine distance: 0=identical, 2=opposite
            # Convert ke similarity score 0–1
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
    # Query: Hybrid (RRF) — identik dengan HybridIndex
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

        bm25_rank    = {r.ko_id: rank for rank, r in enumerate(bm25_results)}
        vector_rank  = {r.ko_id: rank for rank, r in enumerate(vector_results)}
        bm25_scores  = {r.ko_id: r.bm25_score  for r in bm25_results}
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
    # Graph traversal — identik
    # ─────────────────────────────────────────────────────────────────────────

    def graph_neighbors(
        self, ko_id: str, rel_type: Optional[str] = None, direction: str = "both"
    ) -> list[dict]:
        edges = self._graph.get(ko_id, [])
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
        self, ko_id: str, rel_types: Optional[list[str]] = None, depth: int = 2
    ) -> dict[str, list[dict]]:
        visited, frontier = {}, {ko_id}
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
    # Incremental upsert — FITUR BARU (tidak ada di HybridIndex)
    # ─────────────────────────────────────────────────────────────────────────

    def upsert(self, knowledge_object: dict):
        """
        Tambah atau update satu KO tanpa rebuild seluruh index.
        Fitur ini tidak tersedia di HybridIndex (harus rebuild semua).
        """
        ko = knowledge_object
        text = ko["text_content"] + " " + ko["primary_label"]
        embedding = self._embedder([text])[0]
        metadata  = _flatten_metadata({"ko_type": ko["ko_type"],
                                        "primary_label": ko["primary_label"],
                                        **ko["metadata"]})
        self._collection.upsert(
            ids=[ko["ko_id"]],
            documents=[text],
            embeddings=[embedding],
            metadatas=[metadata],
        )
        # Update in-memory lists
        if ko["ko_id"] not in self._ko_id_map:
            self._ko_id_map[ko["ko_id"]] = len(self._kos)
            self._kos.append(ko)
            self._tokenized_corpus.append(tokenize(text))
            # Rebuild BM25 (diperlukan untuk konsistensi)
            self._bm25 = BM25Okapi(self._tokenized_corpus)

    # ─────────────────────────────────────────────────────────────────────────
    # Metadata filter query — FITUR BARU
    # ─────────────────────────────────────────────────────────────────────────

    def search_by_section(self, query: str, section_id: str, top_k: int = 5) -> list[SearchResult]:
        """Query vector search yang difilter per section — tidak ada di HybridIndex."""
        query_embedding = self._embedder([query])[0]
        chroma_results  = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, self._collection.count()),
            where={"section_id": section_id},
            include=["distances", "metadatas"],
        )
        results = []
        for ko_id, dist in zip(chroma_results["ids"][0], chroma_results["distances"][0]):
            if ko_id not in self._ko_id_map:
                continue
            ko = self._kos[self._ko_id_map[ko_id]]
            results.append(SearchResult(
                ko_id=ko_id, ko_type=ko["ko_type"],
                primary_label=ko["primary_label"],
                score=1.0 - dist / 2.0, bm25_score=0.0,
                vector_score=1.0 - dist / 2.0,
                metadata=ko["metadata"], text_content=ko["text_content"],
            ))
        return sorted(results, key=lambda r: r.score, reverse=True)

    # ─────────────────────────────────────────────────────────────────────────
    # Persist / Load — BERUBAH (ChromaDB persistent + embedder files)
    # ─────────────────────────────────────────────────────────────────────────

    def save(self, output_dir: str | Path):
        """
        Persist index ke disk.

        PERUBAHAN dari HybridIndex.save():
          Sebelum: tfidf_matrix.npy (4MB) + index_meta.json (4MB)
          Sesudah: chroma_db/ folder + index_meta.json + embedder files

        ChromaDB auto-persist jika dibuat dengan PersistentClient.
        File yang disimpan:
          output_dir/
            chroma_db/          ← ChromaDB SQLite + HNSW index (BARU)
            index_meta.json     ← KOs, graph, path index (sama, tanpa tfidf)
            embedder_idf.npy    ← IDF vector untuk embedder (kecil, ~8KB)
            embedder_vocab.json ← Vocabulary (sama ukuran, format berbeda)
        """
        out = Path(output_dir)
        out.mkdir(exist_ok=True)

        # Save non-ChromaDB state
        meta = {
            "kos":             self._kos,
            "ko_id_map":       self._ko_id_map,
            "graph":           self._graph,
            "path_index":      self._path_index,
            "type_index":      self._type_index,
            "section_index":   self._section_index,
            "tokenized_corpus": self._tokenized_corpus,
        }
        with open(out / "index_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False)

        # Save embedder
        self._embedder.save(out)

        # ChromaDB sudah auto-persist jika pakai PersistentClient
        # Jika in-memory, export ke persistent sekarang
        chroma_dir = out / "chroma_db"
        if not chroma_dir.exists():
            self._persist_chroma_to_disk(chroma_dir)

        print(f"  Index saved to {out}/")
        print(f"    index_meta.json     : {len(self._kos)} KOs")
        print(f"    chroma_db/          : ChromaDB persistent store")

    def _persist_chroma_to_disk(self, chroma_dir: Path):
        """Export in-memory ChromaDB ke disk."""
        chroma_dir.mkdir(exist_ok=True)
        persistent_client = chromadb.PersistentClient(path=str(chroma_dir))
        try:
            persistent_client.delete_collection(self.COLLECTION_NAME)
        except Exception:
            pass
        new_col = persistent_client.create_collection(
            name=self.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
        )
        # Get all data from in-memory collection
        all_data = self._collection.get(include=["embeddings", "documents", "metadatas"])
        if all_data["ids"]:
            BATCH_SIZE = 100
            ids, docs, embs, metas = (
                all_data["ids"], all_data["documents"],
                all_data["embeddings"], all_data["metadatas"]
            )
            for i in range(0, len(ids), BATCH_SIZE):
                new_col.add(
                    ids=ids[i:i+BATCH_SIZE],
                    documents=docs[i:i+BATCH_SIZE],
                    embeddings=embs[i:i+BATCH_SIZE],
                    metadatas=metas[i:i+BATCH_SIZE],
                )

    @classmethod
    def load(cls, index_dir: str | Path) -> "VectorIndex":
        """
        Load persisted index dari disk.

        PERUBAHAN dari HybridIndex.load():
          Sebelum: load tfidf_matrix.npy (4MB ke RAM sekaligus)
          Sesudah: connect ke ChromaDB di disk (lazy, tidak load semua ke RAM)
        """
        idx_dir = Path(index_dir)
        index = cls()

        with open(idx_dir / "index_meta.json", encoding="utf-8") as f:
            meta = json.load(f)

        index._kos             = meta["kos"]
        index._ko_id_map       = meta["ko_id_map"]
        index._graph           = meta["graph"]
        index._path_index      = meta["path_index"]
        index._type_index      = meta["type_index"]
        index._section_index   = meta["section_index"]
        index._tokenized_corpus = meta["tokenized_corpus"]

        # Load embedder
        index._embedder = TFIDFEmbedder.load(idx_dir)

        # Connect ChromaDB (tidak load ke RAM)
        chroma_dir = idx_dir / "chroma_db"
        if chroma_dir.exists():
            index._chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
        else:
            index._chroma_client = chromadb.Client()
        index._collection = index._chroma_client.get_collection(cls.COLLECTION_NAME)

        # Rebuild BM25 (tidak serializable)
        index._bm25 = BM25Okapi(index._tokenized_corpus)

        print(f"  VectorIndex loaded from {idx_dir}/")
        print(f"    {len(index._kos)} KOs | ChromaDB: {index._collection.count()} docs")
        return index

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def stats(self):
        from collections import Counter
        print(f"  Knowledge objects  : {len(self._kos)}")
        print(f"  ChromaDB docs      : {self._collection.count()}")
        print(f"  Vocab size         : {len(self._embedder._vocab)}")
        print(f"  Graph nodes        : {len(self._graph)}")
        print(f"  Path index entries : {len(self._path_index)}")
        type_counts = Counter(ko["ko_type"] for ko in self._kos)
        for ko_type, count in type_counts.most_common():
            print(f"    {ko_type:<20} {count:>5}")