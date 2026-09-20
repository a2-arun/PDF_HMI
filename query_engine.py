"""High-level "ask a question, get a grounded answer" entry point.

This is the actual RAG step: retrieve relevant textbook chunks for the
question, decide whether they're actually relevant enough to answer from,
and only then hand them to the language model to phrase a natural answer.
"""
from dataclasses import dataclass, field
from typing import List

import llm
import vector_store
from config import RELEVANCE_DISTANCE_THRESHOLD, TOP_K

NOT_COVERED_MESSAGE = (
    "I couldn't find anything about that in this grade's textbook. Try rephrasing "
    "your question, or check with your teacher if it's covered in a different chapter."
)


@dataclass
class Answer:
    text: str
    sources: List[int] = field(default_factory=list)
    used_llm: bool = False
    grounded: bool = False


def answer_question(question: str, grade_name: str, top_k: int = TOP_K) -> Answer:
    question = question.strip()
    if not question:
        return Answer(text="Please type a question first.")

    if not vector_store.has_documents(grade_name):
        return Answer(
            text=f"The knowledge base for {grade_name.replace('_', ' ')} hasn't been "
            f"built yet. Run `python ingest.py --grade {grade_name}` first."
        )

    results = vector_store.query(grade_name, question, top_k=top_k)
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    relevant = [
        (doc, (meta or {}).get("page"))
        for doc, dist, meta in zip(documents, distances, metadatas)
        if dist <= RELEVANCE_DISTANCE_THRESHOLD
    ]

    if not relevant:
        return Answer(text=NOT_COVERED_MESSAGE, grounded=False)

    context = "\n\n".join(doc for doc, _ in relevant)
    pages = sorted({page for _, page in relevant if page is not None})

    if not llm.is_available():
        return Answer(
            text=(
                "(The local language model isn't running, so here is the relevant "
                "textbook excerpt directly:)\n\n" + context
            ),
            sources=pages,
            used_llm=False,
            grounded=True,
        )

    try:
        text = llm.generate(question, context)
    except llm.OllamaUnavailableError as exc:
        return Answer(text=str(exc), sources=pages, used_llm=False, grounded=True)

    if not text:
        text = "(The language model returned an empty answer.) Here's the relevant excerpt:\n\n" + context

    return Answer(text=text, sources=pages, used_llm=True, grounded=True)
