# Enlighten Bot: RAG-based Educational Assistant for a Humanoid Robot

An HMI + Retrieval-Augmented Generation (RAG) layer that lets a school
student ask questions in natural language and get answers **grounded in
their own prescribed textbook**, instead of the open-ended (and
unverifiable) answers a general-purpose LLM would give.

This is the knowledge/HMI layer of a larger planned humanoid educational
robot. It does not train or fine-tune a language model — it gives an
existing, off-the-shelf model controlled access to a curated knowledge base
built from Grade 6–9 textbooks.

## Why RAG instead of just asking an LLM

A general-purpose LLM answering "What is friction?" draws on its entire
training data — it can be off-syllabus, pitched at the wrong level, or
simply wrong. For a classroom assistant, the answer needs to be traceable
back to the actual curriculum. RAG solves this by splitting the problem in
two:

- **Retrieval** — find the passages of the *actual textbook* relevant to
  the question (ChromaDB + sentence embeddings).
- **Generation** — have a language model turn those passages into a clear,
  natural explanation (a local model served by Ollama).

The model never has to "know" the textbook; it just has to explain whatever
is handed to it. That also means the curriculum can change without
retraining anything — re-run ingestion on the new PDFs and the same model
keeps working.

## Architecture

```
                     INGESTION (run once per textbook, via ingest.py)
Grade PDF ──pdfplumber──▶ raw page text
                              │  strip figure/table refs, page numbers,
                              │  repeated headers/footers/watermarks
                              ▼
                        cleaned page text
                              │  RecursiveCharacterTextSplitter
                              │  (500 chars, 80 overlap, per-page)
                              ▼
                     chunks (text + source page)
                              │  all-mpnet-base-v2 sentence embeddings
                              ▼
                   ChromaDB collection (one per grade)


                     QUESTION ANSWERING (app.py / query_engine.py)
Student question ──embed──▶ ChromaDB similarity search ──▶ top-k chunks
                                                                │
                                        distance too high for all chunks?
                                        │yes                    │no
                                        ▼                       ▼
                             "not covered in this        chunks + question
                              textbook" (no LLM call)     ──▶ Ollama (local LLM)
                                                                │
                                                                ▼
                                                     natural-language answer
                                                     + cited textbook pages
```

## Project layout

| File | Responsibility |
|---|---|
| `config.py` | All tunables (chunk size, model names, thresholds) via env vars |
| `pdf_utils.py` | PDF → cleaned per-page text (pdfplumber + noise removal) |
| `text_processing.py` | Cleaned pages → overlapping chunks with page metadata |
| `embeddings.py` | Loads the sentence-transformer model once (cached) |
| `vector_store.py` | ChromaDB wrapper — one collection per grade |
| `llm.py` | Prompting + calling the local Ollama model, with graceful fallback |
| `query_engine.py` | Orchestrates retrieval → relevance check → generation |
| `ingest.py` | CLI: builds the knowledge base from `grade_pdfs/` |
| `app.py` | Streamlit HMI — grade selection, chat, source citations |

Ingestion and question-answering are deliberately separate. The original
prototype re-extracted and re-embedded the *entire textbook* on every single
question — this version builds each grade's knowledge base once, and the
app only ever does a fast vector lookup at question time.

## Setup

```bash
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r requirements.txt
```

1. **Add textbooks** — place `Grade6.pdf` … `Grade9.pdf` into `grade_pdfs/`
   (see `grade_pdfs/README.md`).
2. **Install a local LLM runtime** — install [Ollama](https://ollama.com),
   then pull a model:
   ```bash
   ollama pull llama3.2
   ```
   The app runs fine without Ollama too: if it can't reach it, it falls
   back to showing the raw retrieved textbook excerpt instead of a
   generated answer, so retrieval can still be demoed/tested standalone.
3. **Build the knowledge base**:
   ```bash
   python ingest.py
   ```
4. **Run the app**:
   ```bash
   streamlit run app.py
   ```

Configuration (model names, chunk size, relevance threshold, etc.) can be
overridden via a `.env` file — see `.env.example`.

## Is Ollama actually needed?

Not for retrieval — ChromaDB and the embedding model are independent of it.
It's only needed for the "turn retrieved text into a natural explanation"
step. Options, roughly in order of quality vs. resource cost:

| Model (via Ollama) | Size | Notes |
|---|---|---|
| `tinyllama` | ~1.1B | Very cheap to run, but frequently produces short, repetitive, or off-topic answers — this is what the earliest prototype used |
| `phi3:mini` | ~3.8B | Good instruction-following for its size |
| `llama3.2` (default here) | 1B/3B variants | Noticeably better grounding and coherence than tinyllama at a similar footprint |

For a robot with real hardware constraints, this is the trade-off to tune:
bigger local model → better answers but more RAM/compute; smaller model →
cheaper to deploy but weaker language quality. The model name is a single
config value (`OLLAMA_MODEL`) specifically so this can be swapped without
touching code. A cloud API could also be swapped in behind the same
`llm.py` interface if offline operation isn't a hard requirement.

## Current status

**Working prototype**, not a production system:

- ✅ PDF ingestion, cleaning (figure refs, page numbers, repeated
  boilerplate), chunking with page-level provenance
- ✅ Semantic retrieval via ChromaDB + `all-mpnet-base-v2`
- ✅ Per-grade knowledge bases, built once via a separate CLI step
- ✅ Relevance thresholding — off-syllabus questions get an honest "not
  covered" response instead of a hallucinated answer
- ✅ Local, swappable LLM backend with a graceful no-LLM fallback
- ✅ Streamlit HMI with grade selection, chat history, and page citations

**Not yet done** (see write-up for full context):

- ⬜ Systematic evaluation against a real question/answer test set
- ⬜ Tuning the relevance-distance threshold against real student questions
- ⬜ Production deployment / packaging for actual robot hardware
- ⬜ Voice input/output layer (this project covers the HMI/text layer only)

The system is **not guaranteed hallucination-free**. Even with retrieval
working correctly, small local LLMs can still misstate what the retrieved
text says — the relevance gate reduces *off-syllabus* answers, it doesn't
fact-check the generated text against the source. That evaluation work is
the natural next step.
