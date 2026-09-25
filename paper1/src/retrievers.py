"""
High-Performance Offline Retrieval Systems (GPU Accelerated & Vectorized Batching)
Implements 7 distinct retrieval configurations:
1. Qwen3-0.6B (Qwen/Qwen3-Embedding-0.6B)
2. BM25 (Lexical baseline)
3. BGE-M3 (BAAI/bge-m3)
4. Multilingual-E5-base (intfloat/multilingual-e5-base)
5. Multilingual-E5-large (intfloat/multilingual-e5-large)
6. LaBSE (sentence-transformers/LaBSE)
7. Hybrid-RRF (Reciprocal Rank Fusion of BM25 + Qwen3-0.6B)

Optimizations:
- Vectorized full-batch query encoding (20x-50x faster)
- GPU-native torch.topk rank extraction
- Persistent disk caching for document corpus embeddings (sub-second indexing on re-runs)
- Batch-size 128 for A100 GPU throughput
"""

import os
import time
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from tqdm import tqdm
import torch

from config import (
    EMBEDDING_MODEL_BGE,
    EMBEDDING_MODEL_E5_BASE,
    EMBEDDING_MODEL_E5_LARGE,
    EMBEDDING_MODEL_LABSE,
    EMBEDDING_MODEL_QWEN3_06B,
    EMBEDDING_MODEL_QWEN3_4B,
    CACHE_DIR,
)


class BM25Retriever:
    def __init__(self):
        self.corpus_ids = []
        self.corpus_texts = []
        self.bm25 = None

    def _tokenize(self, text: str) -> List[str]:
        return [t.lower() for t in text.split() if t.strip()]

    def index(self, corpus: List[Dict[str, str]]):
        from rank_bm25 import BM25Okapi
        self.corpus_ids = [doc["doc_id"] for doc in corpus]
        self.corpus_texts = [doc["text"] for doc in corpus]
        tokenized_corpus = [self._tokenize(t) for t in self.corpus_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [(self.corpus_ids[i], float(scores[i])) for i in top_indices]

    def retrieve_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        results = []
        for q in queries:
            results.append(self.retrieve(q, top_k=top_k))
        return results


class DenseEmbeddingRetriever:
    def __init__(
        self,
        model_name: str,
        is_e5: bool = False,
        is_qwen: bool = False,
        device: Optional[str] = None,
        batch_size: int = 128,
    ):
        self.model_name = model_name
        self.is_e5 = is_e5
        self.is_qwen = is_qwen or ("qwen" in model_name.lower())
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = None
        self.corpus_ids = []
        self.corpus_embeddings = None

    def _load_model(self):
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            print(f"[*] Loading Dense Retriever: {self.model_name} on {self.device}...")
            try:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    model_kwargs={"use_safetensors": True},
                    tokenizer_kwargs={"padding_side": "left"} if self.is_qwen else None,
                    trust_remote_code=True,
                )
            except Exception:
                self.model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    trust_remote_code=True,
                )

    def index(self, corpus: List[Dict[str, str]]):
        self.corpus_ids = [doc["doc_id"] for doc in corpus]
        texts = [doc["text"] for doc in corpus]
        
        # Check disk cache for pre-computed corpus embeddings
        safe_name = self.model_name.replace("/", "_").replace("-", "_")
        cache_file = CACHE_DIR / f"corpus_emb_{safe_name}_{len(texts)}.pt"

        if cache_file.exists():
            try:
                print(f"[*] Loading pre-computed corpus embeddings from disk cache: {cache_file.name}...")
                loaded = torch.load(cache_file, map_location=self.device, weights_only=True)
                self.corpus_embeddings = loaded.float()
                return
            except Exception:
                pass

        self._load_model()
        if self.is_e5:
            texts = [f"passage: {t}" for t in texts]

        print(f"[*] Encoding {len(texts)} documents with {self.model_name} (batch_size={self.batch_size})...")
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        self.corpus_embeddings = embeddings.float()

        # Save to cache for future instant loading
        try:
            torch.save(self.corpus_embeddings.half().cpu(), cache_file)
            print(f"[✓] Cached {self.model_name} corpus embeddings to {cache_file.name}")
        except Exception:
            pass

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        """High-speed vectorized batch retrieval with GPU-accelerated top-k."""
        self._load_model()
        if self.is_qwen:
            try:
                q_embs = self.model.encode(
                    queries,
                    prompt_name="query",
                    batch_size=64,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
            except Exception:
                q_embs = self.model.encode(
                    queries,
                    batch_size=64,
                    convert_to_tensor=True,
                    normalize_embeddings=True,
                )
        else:
            q_texts = [f"query: {q}" for q in queries] if self.is_e5 else queries
            q_embs = self.model.encode(
                q_texts,
                batch_size=64,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )

        # GPU Dot Product & Top-K Matrix Operation: [N_queries, N_docs]
        scores = torch.mm(q_embs.float(), self.corpus_embeddings.transpose(0, 1))
        top_k_actual = min(top_k, scores.shape[1])
        top_scores, top_indices = torch.topk(scores, k=top_k_actual, dim=1)

        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()

        batch_results = []
        for q_idx in range(len(queries)):
            res = [
                (self.corpus_ids[top_indices[q_idx, i]], float(top_scores[q_idx, i]))
                for i in range(top_k_actual)
            ]
            batch_results.append(res)

        return batch_results


class HybridRRFRetriever:
    """Combines BM25 and Dense ranking via Reciprocal Rank Fusion (RRF)."""
    def __init__(self, bm25_retriever: BM25Retriever, dense_retriever: DenseEmbeddingRetriever, k: int = 60):
        self.bm25 = bm25_retriever
        self.dense = dense_retriever
        self.k = k

    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        return self.retrieve_batch([query], top_k=top_k)[0]

    def retrieve_batch(self, queries: List[str], top_k: int = 10) -> List[List[Tuple[str, float]]]:
        bm25_batch = self.bm25.retrieve_batch(queries, top_k=top_k * 2)
        dense_batch = self.dense.retrieve_batch(queries, top_k=top_k * 2)

        batch_results = []
        for q_idx in range(len(queries)):
            bm25_res = bm25_batch[q_idx]
            dense_res = dense_batch[q_idx]

            rrf_scores = {}
            for rank, (doc_id, _) in enumerate(bm25_res, 1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))

            for rank, (doc_id, _) in enumerate(dense_res, 1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.k + rank))

            sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
            batch_results.append(sorted_docs)

        return batch_results


class RetrievalSuite:
    """Orchestrates all 7 retrievers across all transformed query variants with batch acceleration."""
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.systems = {}

    def setup_systems(self, corpus: List[Dict[str, str]]):
        print("=" * 70)
        print(f"[*] Setting up 7 Retrieval Systems on {self.device} (Corpus size: {len(corpus)} docs)")
        print("=" * 70)

        # 1. Qwen3-Embedding-0.6B (Primary Modern Dense Model)
        print("[1/7] Indexing Qwen3-Embedding-0.6B...")
        qwen = DenseEmbeddingRetriever(EMBEDDING_MODEL_QWEN3_06B, is_qwen=True, device=self.device)
        qwen.index(corpus)
        self.systems["Qwen3-0.6B"] = qwen

        # 2. BM25 (Lexical baseline)
        print("[2/7] Indexing BM25...")
        bm25 = BM25Retriever()
        bm25.index(corpus)
        self.systems["BM25"] = bm25

        # 3. BGE-M3
        print("[3/7] Indexing BGE-M3...")
        bge = DenseEmbeddingRetriever(EMBEDDING_MODEL_BGE, is_e5=False, device=self.device)
        bge.index(corpus)
        self.systems["BGE-M3"] = bge

        # 4. Multilingual-E5-base
        print("[4/7] Indexing Multilingual-E5-base...")
        e5_base = DenseEmbeddingRetriever(EMBEDDING_MODEL_E5_BASE, is_e5=True, device=self.device)
        e5_base.index(corpus)
        self.systems["mE5-base"] = e5_base

        # 5. Multilingual-E5-large
        print("[5/7] Indexing Multilingual-E5-large...")
        e5_large = DenseEmbeddingRetriever(EMBEDDING_MODEL_E5_LARGE, is_e5=True, device=self.device)
        e5_large.index(corpus)
        self.systems["mE5-large"] = e5_large

        # 6. LaBSE
        print("[6/7] Indexing LaBSE...")
        labse = DenseEmbeddingRetriever(EMBEDDING_MODEL_LABSE, is_e5=False, device=self.device)
        labse.index(corpus)
        self.systems["LaBSE"] = labse

        # 7. Hybrid RRF (BM25 + Qwen3-0.6B)
        print("[7/7] Configuring Hybrid-RRF (BM25 + Qwen3-0.6B)...")
        hybrid = HybridRRFRetriever(bm25, qwen)
        self.systems["Hybrid-RRF"] = hybrid

        print(f"[✓] All {len(self.systems)} retrieval systems indexed and ready.")

    def run_all(
        self,
        queries_dict: Dict[str, Dict[str, str]], # {query_id: {condition: query_text}}
        top_k: int = 10,
    ) -> Dict[str, Dict[str, Dict[str, List[Tuple[str, float]]]]]:
        """
        Runs all retrieval systems across all query conditions using batch acceleration.
        Returns: {system_name: {condition: {query_id: [(doc_id, score), ...]}}}
        """
        results = {sys_name: {} for sys_name in self.systems}

        # Identify conditions and ordered query IDs
        sample_q = next(iter(queries_dict.values()))
        conditions = list(sample_q.keys())
        query_ids = list(queries_dict.keys())

        for sys_name, system in self.systems.items():
            print(f"[*] Running Batched Retrieval for: {sys_name:<14} across {len(conditions)} conditions...", flush=True)
            for cond in conditions:
                cond_queries = [queries_dict[qid].get(cond, "") for qid in query_ids]
                batch_ranked = system.retrieve_batch(cond_queries, top_k=top_k)
                results[sys_name][cond] = dict(zip(query_ids, batch_ranked))

        return results
