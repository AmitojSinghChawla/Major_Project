"""
04_retrieve.py
--------------
All 4 retrieval methods in one file.

Each function takes a query + top_k and returns a ranked list of results.
Every result has the same structure so evaluation code stays simple.

Result format:
    {
        "rank"          : 1,
        "chunk_id"      : "abc-123",
        "score"         : 0.91,
        "modality"      : "text",
        "source_pdf"    : "paper.pdf",
        "retrieval_text": "...",
        "raw_text"      : "...",   # None for images
        "image_b64"     : None,    # present for image chunks
    }

Usage:
    from retrieve import load_indexes, retrieve
    indexes = load_indexes()
    results = retrieve("What is the revenue?", method="hybrid", indexes=indexes, top_k=5)
"""

import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder


BM25_FILE   = "bm25_index.pkl"
FAISS_FILE  = "faiss_index.bin"
META_FILE   = "index_meta.json"

EMBED_MODEL   = "all-MiniLM-L6-v2"
RERANK_MODEL  = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ===============================
# Load all indexes once
# ===============================

def load_indexes():
    """
    Load BM25, FAISS, metadata, and both models.
    Call this once at startup — not on every query.
    """
    print("Loading indexes and models...")

    with open(BM25_FILE, "rb") as f:
        bm25 = pickle.load(f)

    faiss_index = faiss.read_index(FAISS_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    embedder = SentenceTransformer(EMBED_MODEL)
    reranker = CrossEncoder(RERANK_MODEL)

    print("All indexes loaded.")

    return {
        "bm25"      : bm25,
        "faiss"     : faiss_index,
        "meta"      : meta,
        "embedder"  : embedder,
        "reranker"  : reranker,
    }


# ===============================
# Helper: build result dict
# ===============================

def _make_result(rank, score, meta_entry):
    return {
        "rank"          : rank,
        "chunk_id"      : meta_entry["chunk_id"],
        "score"         : round(float(score), 4),
        "modality"      : meta_entry["modality"],
        "source_pdf"    : meta_entry["source_pdf"],
        "retrieval_text": meta_entry["retrieval_text"],
        "raw_text"      : meta_entry.get("raw_text"),
        # image_b64 not stored in meta — look up from chunks.jsonl by chunk_id if needed
        "image_b64"     : None,
    }


# ===============================
# 1. Sparse — BM25
# ===============================

def retrieve_bm25(query, bm25, meta, top_k=5):
    tokenized_query = query.lower().split()
    scores          = bm25.get_scores(tokenized_query)
    top_indices     = np.argsort(scores)[::-1][:top_k]

    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append(_make_result(rank, scores[idx], meta[idx]))

    return results


# ===============================
# 2. Dense — FAISS
# ===============================

def retrieve_dense(query, faiss_index, meta, embedder, top_k=5):
    query_vec = embedder.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)

    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for rank, (idx, score) in enumerate(zip(indices[0], scores[0]), start=1):
        if idx == -1:   # FAISS returns -1 if fewer results than top_k
            continue
        results.append(_make_result(rank, score, meta[idx]))

    return results


# ===============================
# 3. Hybrid — BM25 + Dense (RRF)
# ===============================

def retrieve_hybrid(query, bm25, faiss_index, meta, embedder, top_k=5, rrf_k=60):
    """
    Reciprocal Rank Fusion (RRF) — standard way to combine two ranked lists.
    Each chunk gets a score from both systems. RRF combines them fairly.

    RRF formula: score(d) = sum( 1 / (k + rank(d)) ) across systems
    k=60 is the standard default from the original RRF paper.
    """

    # Get more candidates than top_k so fusion has enough to work with
    fetch_k = top_k * 3

    bm25_results  = retrieve_bm25(query, bm25, meta, top_k=fetch_k)
    dense_results = retrieve_dense(query, faiss_index, meta, embedder, top_k=fetch_k)

    # Build RRF score map keyed by chunk_id
    rrf_scores = {}

    for result in bm25_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + result["rank"])

    for result in dense_results:
        cid = result["chunk_id"]
        rrf_scores[cid] = rrf_scores.get(cid, 0) + 1 / (rrf_k + result["rank"])

    # Sort by RRF score descending
    sorted_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]

    # Build final results — look up meta by chunk_id
    chunk_id_to_meta = {m["chunk_id"]: m for m in meta}

    results = []
    for rank, cid in enumerate(sorted_ids, start=1):
        m = chunk_id_to_meta[cid]
        results.append(_make_result(rank, rrf_scores[cid], m))

    return results


# ===============================
# 4. Hybrid + Reranker
# ===============================

def retrieve_hybrid_reranked(query, bm25, faiss_index, meta, embedder, reranker, top_k=5):
    """
    Step 1: Run hybrid retrieval to get candidates (fetch more than top_k).
    Step 2: Reranker (cross-encoder) scores each candidate against the query.
    Step 3: Re-sort by reranker score and return top_k.

    Cross-encoder reads query + document together → much better relevance judgment
    but too slow to run on all chunks, so we only run it on hybrid's top candidates.
    """

    # Fetch more candidates for reranker to work with
    candidates = retrieve_hybrid(
        query, bm25, faiss_index, meta, embedder,
        top_k=top_k * 3
    )

    if not candidates:
        return []

    # Reranker scores (query, document) pairs
    pairs  = [(query, c["retrieval_text"]) for c in candidates]
    scores = reranker.predict(pairs)

    # Attach reranker score and re-sort
    for candidate, score in zip(candidates, scores):
        candidate["score"] = round(float(score), 4)

    candidates.sort(key=lambda x: x["score"], reverse=True)

    # Re-assign ranks and return top_k
    results = []
    for rank, candidate in enumerate(candidates[:top_k], start=1):
        candidate["rank"] = rank
        results.append(candidate)

    return results


# ===============================
# Unified retrieve() function
# ===============================

def retrieve(query, method, indexes, top_k=5):
    """
    Single entry point for all retrieval methods.

    method options: "bm25" | "dense" | "hybrid" | "hybrid_reranker"
    """

    bm25        = indexes["bm25"]
    faiss_index = indexes["faiss"]
    meta        = indexes["meta"]
    embedder    = indexes["embedder"]
    reranker    = indexes["reranker"]

    if method == "bm25":
        return retrieve_bm25(query, bm25, meta, top_k)

    elif method == "dense":
        return retrieve_dense(query, faiss_index, meta, embedder, top_k)

    elif method == "hybrid":
        return retrieve_hybrid(query, bm25, faiss_index, meta, embedder, top_k)

    elif method == "hybrid_reranker":
        return retrieve_hybrid_reranked(query, bm25, faiss_index, meta, embedder, reranker, top_k)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose: bm25, dense, hybrid, hybrid_reranker")
