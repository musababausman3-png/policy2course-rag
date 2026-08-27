from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from typing import Iterable

import chromadb
import pymupdf
from google import genai
from sentence_transformers import SentenceTransformer


@dataclass
class Chunk:
    chunk_id: str
    page: int
    text: str


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages = []
    for page_idx, page in enumerate(doc, start=1):
        text = re.sub(r"\s+", " ", page.get_text("text")).strip()
        if text:
            pages.append((page_idx, text))
    return pages


def chunk_pages(pages: Iterable[tuple[int, str]], chunk_size: int = 900, overlap: int = 150) -> list[Chunk]:
    chunks = []
    for page_num, text in pages:
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                digest = hashlib.sha1(f"{page_num}:{start}:{chunk_text}".encode()).hexdigest()[:12]
                chunks.append(Chunk(f"p{page_num}-{digest}", page_num, chunk_text))
            if end >= len(text):
                break
            start = max(0, end - overlap)
    return chunks


class PolicyRAG:
    def __init__(self, collection_name: str = "policy_chunks", embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.embedder = SentenceTransformer(embedding_model)
        self.client = chromadb.Client()
        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass
        self.collection = self.client.create_collection(name=collection_name, metadata={"hnsw:space": "cosine"})

    def index(self, chunks: list[Chunk]) -> None:
        texts = [c.text for c in chunks]
        embeddings = self.embedder.encode(texts, normalize_embeddings=True).tolist()
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=texts,
            metadatas=[{"page": c.page} for c in chunks],
            embeddings=embeddings,
        )

    def retrieve(self, query: str, k: int = 5) -> list[dict]:
        q_embedding = self.embedder.encode([query], normalize_embeddings=True).tolist()
        result = self.collection.query(query_embeddings=q_embedding, n_results=k, include=["documents", "metadatas", "distances"])
        hits = []
        for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
            hits.append({"page": int(meta["page"]), "text": doc, "distance": float(dist), "similarity": round(1.0 - float(dist), 4)})
        return hits

    def _gemini(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is missing. Add it to your .env file.")
        return genai.Client(api_key=api_key)

    def generate_course(self) -> dict:
        queries = [
            "policy purpose scope definitions and key principles",
            "employee responsibilities required conduct and compliance",
            "procedures approvals reporting escalation and documentation",
            "exceptions prohibited actions violations consequences penalties",
            "security privacy safety records contacts and implementation",
        ]
        evidence, seen = [], set()
        for q in queries:
            for hit in self.retrieve(q, k=5):
                key = (hit["page"], hit["text"])
                if key not in seen:
                    seen.add(key)
                    evidence.append(hit)
        context = "\n\n".join(f"[SOURCE page {h['page']}]\n{h['text']}" for h in evidence)
        prompt = f'''You are an instructional designer creating employee training ONLY from the supplied company policy evidence.

Create exactly 5 training modules.

Grounding rules:
- Do not invent policies, penalties, deadlines, contacts, or procedures.
- Every module must cite one or more page numbers from the evidence.
- If evidence is insufficient for a detail, omit it.
- Produce valid JSON only, with no markdown fences.

JSON schema:
{{
  "course_title": "string",
  "course_goal": "string",
  "modules": [
    {{
      "module_number": 1,
      "title": "string",
      "learning_objectives": ["string", "string"],
      "lessons": [{{"title": "string", "summary": "string", "source_pages": [1, 2]}}],
      "quiz": [{{"question": "string", "options": ["A", "B", "C", "D"], "answer": "A", "explanation": "string", "source_pages": [1]}}]
    }}
  ]
}}

POLICY EVIDENCE:
{context}
'''
        model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        response = self._gemini().models.generate_content(model=model, contents=prompt)
        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.text.strip(), flags=re.I | re.S)
        return json.loads(raw)

    def answer_question(self, question: str, k: int = 5) -> dict:
        hits = self.retrieve(question, k=k)
        context = "\n\n".join(f"[SOURCE page {h['page']}]\n{h['text']}" for h in hits)
        prompt = f'''Answer the user's question using ONLY the policy evidence below.
If the answer is not supported, say: "I could not verify that from the uploaded policy."
Give a concise answer and finish with source page numbers.

QUESTION:
{question}

POLICY EVIDENCE:
{context}
'''
        model = os.getenv("GEMINI_MODEL", "gemini-3.7-flash")
        response = self._gemini().models.generate_content(model=model, contents=prompt)
        return {"answer": response.text, "retrieved": hits}
