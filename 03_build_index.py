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
import pickle # this lib is important for saving/loading the BM25 index, which is a Python object that can't be saved as JSON.
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
    with open(path, "r", encoding="utf-8") as f: #utf-8 is important to ensure that we can read any text without encoding issues, especially if the PDFs contain special characters or non-English text.
        for line in f:
            line = line.strip()
            if line:
                chunks.append(json.loads(line)) # Each line in chunks.jsonl is a JSON object representing a chunk. We read it line by line, parse the JSON, and append it to the chunks list.
    print(f"Loaded {len(chunks)} chunks from {path}")
    return chunks


# ===============================
# Build BM25 index
# ===============================
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download("stopwords")

stop_words = set(stopwords.words("english"))
stemmer = PorterStemmer()

def tokenize(text):
    # 1. Lowercase
    text = text.lower()

    # 2. Remove punctuation + non-alphanumeric
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # 3. Tokenize
    tokens = text.split()

    # 4. Remove stopwords + short tokens
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # 5. Stem
    tokens = [stemmer.stem(t) for t in tokens]

    return tokens

def build_bm25(chunks):
    corpus = [tokenize(chunk["retrieval_text"]) for chunk in chunks]
    bm25 = BM25Okapi(corpus)
    print(f"BM25 index built over {len(corpus)} documents")
    return bm25


def build_faiss(chunks, model):

    texts      = [chunk["retrieval_text"] for chunk in chunks] # again tokenize the retrieval_text, but this time we keep it as raw text because the embedding model will handle tokenization internally. We just need a list of strings to pass to the model.encode() function.

    embeddings = model.encode(texts, show_progress_bar=True, convert_to_numpy=True) # We use the SentenceTransformer model to encode the list of texts into dense vector embeddings. The show_progress_bar=True argument will display a progress bar in the terminal, which is helpful if we have a large number of chunks. The convert_to_numpy=True argument ensures that the output is a NumPy array, which is the format that FAISS expects for building the index.

    # Normalize so inner product == cosine similarity
    faiss.normalize_L2(embeddings)
    # FAISS can use different types of indexes. For simplicity, we use IndexFlatIP, which is a brute-force index that computes inner product (dot product) between the query vector and all indexed vectors. Since we've normalized the embeddings to have unit length, the inner product will effectively be the cosine similarity. This is a simple and effective choice for small to medium-sized datasets. For larger datasets, we might consider more complex index types that allow for faster approximate nearest neighbor search.

    dim   = embeddings.shape[1] # The dimensionality of the embeddings is determined by the second dimension of the embeddings array. This is needed to initialize the FAISS index correctly.
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index

# we are creating a metadata list that is aligned with the order of the chunks and the indexes. Each entry in the meta list corresponds to a chunk and contains the relevant metadata fields. This allows us to easily look up the metadata for any retrieved chunk by using its index in the meta list, which will match the index in both the BM25 and FAISS results.
def build_meta(chunks):

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
