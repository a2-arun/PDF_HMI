"""Builds the textbook knowledge base: PDF -> clean text -> chunks -> ChromaDB.

Run this once per textbook (or whenever a textbook changes) before asking
questions:

    python ingest.py                 # build every grade that isn't built yet
    python ingest.py --grade Grade_6 # build (or check) just one grade
    python ingest.py --force         # rebuild everything from scratch

This is deliberately separate from the Streamlit app: a student asking a
question should never trigger a multi-minute PDF re-extraction + re-embedding
pass, which is what the original prototype did on every single query.
"""
import argparse
from pathlib import Path

import pdf_utils
import text_processing
import vector_store
from config import GRADE_PDFS


def ingest_grade(grade_name: str, pdf_path: Path, force: bool = False) -> None:
    if not force and vector_store.has_documents(grade_name):
        print(f"[skip] {grade_name}: already built (use --force to rebuild)")
        return

    if not pdf_path.exists():
        print(f"[warn] {grade_name}: PDF not found at '{pdf_path}', skipping")
        return

    print(f"[ingest] {grade_name}: extracting text from {pdf_path.name}")
    pages = pdf_utils.extract_clean_pages(str(pdf_path))

    print(f"[ingest] {grade_name}: chunking {len(pages)} pages")
    chunks = text_processing.chunk_pages(pages)
    print(f"[ingest] {grade_name}: {len(chunks)} chunks produced")

    if not chunks:
        print(f"[warn] {grade_name}: no usable text extracted, nothing stored")
        return

    if force:
        vector_store.reset_collection(grade_name)

    print(f"[ingest] {grade_name}: embedding + storing in ChromaDB")
    vector_store.add_chunks(grade_name, chunks)
    print(f"[done] {grade_name}: ready ({len(chunks)} chunks)")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the textbook knowledge base.")
    parser.add_argument("--grade", choices=sorted(GRADE_PDFS), help="Only ingest this grade")
    parser.add_argument("--force", action="store_true", help="Rebuild even if already built")
    args = parser.parse_args()

    targets = [args.grade] if args.grade else sorted(GRADE_PDFS)
    for grade_name in targets:
        ingest_grade(grade_name, Path(GRADE_PDFS[grade_name]), force=args.force)


if __name__ == "__main__":
    main()
