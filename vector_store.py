"""Thin wrapper around ChromaDB: one collection per grade.

Keeps the rest of the codebase from having to know about ChromaDB's API
directly, and centralizes the "one client per process" and "cosine
distance space" choices so ingestion and retrieval always agree.
"""
from typing import List

import chromadb

from config import CHROMA_PERSIST_DIR
from embeddings import get_embedder
from text_processing import Chunk

_client = None

# Chunks are added in batches so embedding a whole textbook doesn't require
# holding every embedding in memory / one giant call into Chroma at once.
_ADD_BATCH_SIZE = 256


def get_client() -> chromadb.ClientAPI:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    return _client


def get_collection(grade_name: str):
    return get_client().get_or_create_collection(
        name=grade_name,
        metadata={"hnsw:space": "cosine"},
    )


def has_documents(grade_name: str) -> bool:
    try:
        return get_collection(grade_name).count() > 0
    except Exception:
        return False


def reset_collection(grade_name: str) -> None:
    try:
        get_client().delete_collection(grade_name)
    except Exception:
        pass


def add_chunks(grade_name: str, chunks: List[Chunk]) -> None:
    if not chunks:
        return

    collection = get_collection(grade_name)
    embedder = get_embedder()
    texts = [c.text for c in chunks]
    embeddings = embedder.encode(texts, show_progress_bar=False, batch_size=32).tolist()
    ids = [f"{grade_name}_{i}" for i in range(len(chunks))]
    metadatas = [{"page": c.page} for c in chunks]

    for start in range(0, len(chunks), _ADD_BATCH_SIZE):
        end = start + _ADD_BATCH_SIZE
        collection.add(
            documents=texts[start:end],
            embeddings=embeddings[start:end],
            ids=ids[start:end],
            metadatas=metadatas[start:end],
        )


def query(grade_name: str, question: str, top_k: int) -> dict:
    collection = get_collection(grade_name)
    embedder = get_embedder()
    query_vec = embedder.encode([question])[0].tolist()
    return collection.query(
        query_embeddings=[query_vec],
        n_results=min(top_k, max(collection.count(), 1)),
        include=["documents", "distances", "metadatas"],
    )
