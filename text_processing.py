"""Splits cleaned page text into overlapping chunks for embedding.

Chunking is done per-page (rather than on one giant concatenated string) so
each chunk can carry a page-number, which lets the app cite where an answer
came from and gives students/teachers a way to verify it against the book.
"""
from dataclasses import dataclass
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, MIN_CHUNK_CHARS
from pdf_utils import Page


@dataclass
class Chunk:
    text: str
    page: int


def chunk_pages(
    pages: List[Page],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    min_chunk_chars: int = MIN_CHUNK_CHARS,
) -> List[Chunk]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks: List[Chunk] = []
    seen = set()

    for page in pages:
        if not page.text.strip():
            continue
        for piece in splitter.split_text(page.text):
            piece = piece.strip()
            if len(piece) < min_chunk_chars or piece in seen:
                continue
            seen.add(piece)
            chunks.append(Chunk(text=piece, page=page.number))

    return chunks
