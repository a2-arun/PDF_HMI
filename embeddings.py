"""Loads the sentence-embedding model exactly once per process.

The old code created a fresh `SentenceTransformer` on every single
question, which reloads several hundred MB of model weights per query.
Caching it here means it's loaded once when the app/ingest script starts.
"""
from functools import lru_cache

from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME


@lru_cache(maxsize=1)
def get_embedder() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL_NAME)
