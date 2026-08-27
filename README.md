# Policy2Course RAG

A small Retrieval-Augmented Generation (RAG) app that converts a company policy PDF into a **grounded 5-module employee training course**.

## Why this project
Naive prompting can introduce rules that do not exist in the source policy. This project retrieves relevant policy passages first, then asks the LLM to generate only from that evidence.

## Features
- PDF parsing with **PyMuPDF**
- Page-aware chunking with overlap
- Text embeddings with **Sentence Transformers**
- Vector retrieval using **ChromaDB** and cosine similarity
- **Gemini API** for grounded generation
- Exactly **5 modules** as structured JSON
- Page citations inside lessons and quiz items
- Grounded policy Q&A
- Retrieval-inspection tab for debugging relevance

## Architecture
```text
Policy PDF → PyMuPDF → page-aware chunks → embeddings → ChromaDB
                                         ↓
                                  top-k retrieval
                                         ↓
                                   Gemini API
                                         ↓
                         5-module course JSON + Q&A
```

## Setup
```bash
git clone <YOUR_REPOSITORY_URL>
cd idiongo-policy-rag
python -m venv .venv
```

Windows:
```bash
.venv\Scripts\activate
```

macOS/Linux:
```bash
source .venv/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Copy `.env.example` to `.env` and add your Gemini API key:
```env
GEMINI_API_KEY=your_key_here
GEMINI_MODEL=gemini-3.7-flash
```

Run:
```bash
streamlit run app.py
```

## How the RAG pipeline works
1. **Ingest** — read the PDF page-by-page with PyMuPDF.
2. **Chunk** — create overlapping chunks while preserving page metadata.
3. **Embed** — encode chunks with `all-MiniLM-L6-v2`.
4. **Index** — store embeddings and metadata in ChromaDB.
5. **Retrieve** — use multiple policy-oriented queries to improve coverage.
6. **Generate** — Gemini receives only retrieved evidence and returns a 5-module JSON course scaffold.
7. **Verify** — the UI exposes retrieved passages so the user can inspect evidence.

## Hard problems
- **Chunk boundaries:** rules and exceptions may be split across sections/pages.
- **Retrieval coverage:** a single query can miss important policy topics.
- **Hallucination:** generated content must stay within the uploaded policy.
- **Structured output:** JSON must remain valid and consistent.
- **Sensitive data:** production deployments need workspace isolation and access controls.

## Production improvements
- semantic/heading-aware chunking
- metadata filters by department and sensitivity
- cross-encoder re-ranking
- persistent pgvector/Chroma
- RAGAS/TruLens evaluation
- Pydantic schema validation
- OCR fallback for scanned PDFs
- tests + CI
- human approval before publishing generated training

## Tech stack
Python · Streamlit · PyMuPDF · Sentence Transformers · ChromaDB · Gemini API

## Author
Musa Baba Usman
