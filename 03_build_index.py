"""
03_build_index.py
-----------------
Reads chunks.jsonl and builds two indexes:

  bm25_index.pkl   — BM25 sparse index (keyword matching)
  faiss_index.bin  — FAISS dense index (semantic vectors)
  index_meta.json  — chunk_id + metadata list aligned to both indexes

Run this once after ingestion. Re-run if chunks.jsonl changes.

Usage:
    python 03_build_index.py
"""

import json
import pickle
import faiss
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


CHUNKS_FILE  = "chunks.jsonl"
BM25_FILE    = "bm25_index.pkl"
FAISS_FILE   = "faiss_index.bin"
META_FILE    = "index_meta.json"

# Model used for dense embeddings
# all-MiniLM-L6-v2 is small, fast, and good enough for a thesis
EMBED_MODEL  = "all-MiniLM-L6-v2"


# ===============================
# Load chunks
# ===============================

def load_chunks(path):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line))
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ===============================
# Build BM25 index
# ===============================

def build_bm25(chunks):
    """
    Tokenize each chunk's retrieval_text and build a BM25Okapi index.
    Tokenization is simple whitespace split — good enough for BM25.
    """
    corpus = [chunk["retrieval_text"].lower().split() for chunk in chunks]
    bm25   = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents")
    return bm25


# ===============================
# Build FAISS dense index
# ===============================

def build_faiss(chunks, model):
    """
    Embed each chunk's retrieval_text and build a FAISS flat L2 index.
    FlatIP = inner product (cosine similarity with normalized vectors).
    """
    texts      = [chunk["retrieval_text"] for chunk in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True)

    # Normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    dim   = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


# ===============================
# Save index metadata
# ===============================

def build_meta(chunks):
    """
    Store chunk_id + modality + source_pdf aligned to index position.
    Position 0 in meta = position 0 in BM25 corpus = position 0 in FAISS.

    NOTE: image_b64 is NOT stored here — it stays in chunks.jsonl.
    Storing base64 images in index_meta.json would make it enormous.
    Retrievers return chunk_id; you look up image_b64 from chunks.jsonl if needed.
    """
    meta = []
    for chunk in chunks:
        meta.append({
            "chunk_id"      : chunk["chunk_id"],
            "modality"      : chunk["modality"],
            "source_pdf"    : chunk["source_pdf"],
            "page_number"   : chunk["page_number"],
            "retrieval_text": chunk["retrieval_text"],
            "raw_text"      : chunk.get("raw_text"),
            # image_b64 intentionally excluded — look up from chunks.jsonl by chunk_id
        })
    return meta


# ===============================
# Main
# ===============================

def main():

    # 1. Load all chunks
    chunks = load_chunks(CHUNKS_FILE)

    # 2. Build and save BM25
    bm25 = build_bm25(chunks)
    with open(BM25_FILE, "wb") as f:
        pickle.dump(bm25, f)
    print(f"Saved BM25 index → {BM25_FILE}")

    # 3. Load embedding model and build FAISS
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    model = SentenceTransformer(EMBED_MODEL)
    faiss_index = build_faiss(chunks, model)
    faiss.write_index(faiss_index, FAISS_FILE)
    print(f"Saved FAISS index → {FAISS_FILE}")

    # 4. Save metadata (position-aligned to both indexes)
    meta = build_meta(chunks)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"Saved index metadata → {META_FILE}")

    print("\nIndexing complete. Files created:")
    print(f"  {BM25_FILE}")
    print(f"  {FAISS_FILE}")
    print(f"  {META_FILE}")


if __name__ == "__main__":
    main()
