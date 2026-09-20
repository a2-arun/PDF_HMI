"""PDF text extraction and cleaning.

Textbook PDFs come with a lot of noise that hurts retrieval quality if it
ends up in the vector store: figure/table captions, page numbers, and
publisher boilerplate (e.g. "Reprint 2024-25") repeated on nearly every
page. This module extracts page-by-page text and strips that noise out
before it ever reaches the chunker/embedder.
"""
import re
from collections import Counter
from dataclasses import dataclass
from typing import List

import pdfplumber

from config import BOILERPLATE_LINE_RATIO, BOILERPLATE_MIN_PAGES

_FIGURE_REF_RE = re.compile(r"\b(fig(?:ure)?|table)\.?\s*\d+(\.\d+)*\b[:.]?", re.IGNORECASE)
_PAGE_NUMBER_RE = re.compile(r"^\s*(page\s*)?\d{1,4}\s*$", re.IGNORECASE)
_MULTI_SPACE_RE = re.compile(r"[ \t]+")
_HYPHEN_BREAK_RE = re.compile(r"([A-Za-z]{2,})-$")


@dataclass
class Page:
    number: int
    text: str


def extract_pages(pdf_path: str) -> List[Page]:
    """Read every page of a PDF into raw (uncleaned) text."""
    pages: List[Page] = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            pages.append(Page(number=i, text=raw))
    return pages


def _clean_page_lines(raw_text: str) -> List[str]:
    lines: List[str] = []
    for raw_line in raw_text.splitlines():
        line = _FIGURE_REF_RE.sub("", raw_line)
        line = _MULTI_SPACE_RE.sub(" ", line).strip()
        if not line or _PAGE_NUMBER_RE.match(line):
            continue
        # Re-join words split across a line break, e.g. "envi-" + "ronment".
        if lines and _HYPHEN_BREAK_RE.search(lines[-1]):
            lines[-1] = _HYPHEN_BREAK_RE.sub(r"\1", lines[-1]) + line
        else:
            lines.append(line)
    return lines


def clean_document(pages: List[Page]) -> List[Page]:
    """Clean every page, then drop lines that repeat across many pages.

    Repeated lines (running headers, footers, copyright watermarks) are
    detected by frequency rather than a fixed blocklist, so this works
    across differently-formatted textbooks without per-book tuning.
    """
    page_lines = [_clean_page_lines(p.text) for p in pages]

    line_counts: Counter = Counter()
    for lines in page_lines:
        for line in set(lines):
            line_counts[line] += 1

    total_pages = len(pages)
    boilerplate = set()
    if total_pages >= BOILERPLATE_MIN_PAGES:
        boilerplate = {
            line
            for line, count in line_counts.items()
            if count / total_pages >= BOILERPLATE_LINE_RATIO
        }

    cleaned_pages = []
    for page, lines in zip(pages, page_lines):
        kept = [line for line in lines if line not in boilerplate]
        cleaned_pages.append(Page(number=page.number, text="\n".join(kept)))
    return cleaned_pages


def extract_clean_pages(pdf_path: str) -> List[Page]:
    """Convenience wrapper: extract + clean in one call."""
    return clean_document(extract_pages(pdf_path))
