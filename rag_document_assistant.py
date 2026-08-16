"""RAG Document Assistant — single-file FastAPI demo.

Run:
  pip install fastapi "uvicorn[standard]" python-multipart pypdf sentence-transformers chromadb openai
  export OPENAI_API_KEY="..."          # optional; local retrieval still works without it
  uvicorn rag_document_assistant:app --reload
Open http://127.0.0.1:8000

This application intentionally does not request or expose hidden chain-of-thought.
It uses a short answer policy, few-shot output examples, grounded retrieval,
source citations, and an evidence check instead.
"""

from __future__ import annotations

import os
import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import chromadb
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
CHROMA_DIR = DATA_DIR / "chroma"
COLLECTION_NAME = "documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5
CHUNK_SIZE = 900
CHUNK_OVERLAP = 160

DATA_DIR.mkdir(exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="RAG Document Assistant")
embedder = SentenceTransformer(EMBEDDING_MODEL)
chroma = chromadb.PersistentClient(path=str(CHROMA_DIR))
collection = chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    top_k: int = Field(default=TOP_K, ge=1, le=10)


class Source(BaseModel):
    document: str
    page: int | None
    chunk_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[Source]


class EvaluationExample(BaseModel):
    question: str
    expected_answer_contains: list[str]


class EvaluationResult(BaseModel):
    question: str
    answer: str
    grounded: bool
    passed: bool
    missing_expected_terms: list[str]
    source_count: int


def read_document(path: Path) -> list[tuple[str, int | None]]:
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return [(path.read_text(encoding="utf-8", errors="ignore"), None)]
    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return [(page.extract_text() or "", index + 1) for index, page in enumerate(reader.pages)]
    raise HTTPException(status_code=400, detail="Only .pdf and .txt files are supported.")


def chunk_text(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("? ", start, end), text.rfind("! ", start, end))
            if boundary > start + CHUNK_SIZE // 2:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def add_document(path: Path) -> int:
    documents, metadatas, ids = [], [], []
    for text, page in read_document(path):
        for number, chunk in enumerate(chunk_text(text)):
            chunk_id = f"{path.stem}-{page or 0}-{number}-{uuid.uuid4().hex[:8]}"
            documents.append(chunk)
            metadatas.append({"document": path.name, "page": page or 0, "chunk_number": number})
            ids.append(chunk_id)
    if not documents:
        raise HTTPException(status_code=400, detail="No readable text was found in this file.")
    embeddings = embedder.encode(documents, normalize_embeddings=True).tolist()
    collection.add(ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings)
    return len(documents)


def retrieve(question: str, top_k: int) -> list[dict[str, Any]]:
    if collection.count() == 0:
        return []
    vector = embedder.encode([question], normalize_embeddings=True).tolist()
    result = collection.query(query_embeddings=vector, n_results=min(top_k, collection.count()), include=["documents", "metadatas", "distances"])
    hits = []
    for doc, metadata, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        hits.append({"text": doc, "metadata": metadata, "score": 1 - float(distance)})
    return hits


def make_sources(hits: list[dict[str, Any]]) -> list[Source]:
    return [
        Source(
            document=hit["metadata"]["document"],
            page=hit["metadata"].get("page") or None,
            chunk_id=f"{hit['metadata']['document']}#{hit['metadata'].get('chunk_number', 0)}",
            excerpt=hit["text"][:360] + ("…" if len(hit["text"]) > 360 else ""),
        )
        for hit in hits
    ]


def answer_with_llm(question: str, hits: list[dict[str, Any]]) -> str | None:
    key = os.getenv("OPENAI_API_KEY")
    if not key or OpenAI is None:
        return None
    context = "\n\n".join(
        f"[S{i + 1}: {h['metadata']['document']}, page {h['metadata'].get('page') or 'n/a'}]\n{h['text']}"
        for i, h in enumerate(hits)
    )
    system = """You answer questions using ONLY the supplied document excerpts.
Do not reveal private reasoning or chain-of-thought. Give a concise answer, then cite supporting sources as [S1], [S2], etc.
If the excerpts do not contain the answer, say exactly: \"I don't have enough evidence in the uploaded documents to answer that.\"

Few-shot examples:
User: What is the refund deadline?
Evidence: [S1] Refund requests must be submitted within 30 days of purchase.
Assistant: Refund requests must be submitted within 30 days of purchase. [S1]

User: Who is the CEO?
Evidence: [S1] This document describes customer support procedures.
Assistant: I don't have enough evidence in the uploaded documents to answer that."""
    client = OpenAI(api_key=key)
    response = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        temperature=0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Question: {question}\n\nEvidence:\n{context}"},
        ],
    )
    return response.choices[0].message.content.strip()


def fallback_answer(hits: list[dict[str, Any]]) -> str:
    if not hits or hits[0]["score"] < 0.22:
        return "I don't have enough evidence in the uploaded documents to answer that."
    sentences = re.split(r"(?<=[.!?])\s+", hits[0]["text"])
    evidence = " ".join(sentences[:3]).strip()
    return f"I found this relevant evidence: {evidence} [S1]"


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "indexed_chunks": collection.count(), "embedding_model": EMBEDDING_MODEL}


@app.post("/upload")
def upload(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename:
        raise HTTPException(status_code=400, detail="A file name is required.")
    destination = UPLOAD_DIR / Path(file.filename).name
    with destination.open("wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    count = add_document(destination)
    return {"message": "Document indexed.", "file": destination.name, "chunks_added": count, "total_chunks": collection.count()}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    hits = retrieve(request.question, request.top_k)
    sources = make_sources(hits)
    answer = answer_with_llm(request.question, hits) if hits else None
    answer = answer or fallback_answer(hits)
    grounded = not answer.startswith("I don't have enough evidence") and bool(hits)
    return AskResponse(answer=answer, grounded=grounded, sources=sources)


@app.post("/evaluate", response_model=list[EvaluationResult])
def evaluate(examples: list[EvaluationExample]) -> list[EvaluationResult]:
    results = []
    for example in examples:
        response = ask(AskRequest(question=example.question))
        lower_answer = response.answer.lower()
        missing = [term for term in example.expected_answer_contains if term.lower() not in lower_answer]
        results.append(EvaluationResult(
            question=example.question,
            answer=response.answer,
            grounded=response.grounded,
            passed=response.grounded and not missing,
            missing_expected_terms=missing,
            source_count=len(response.sources),
        ))
    return results


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>RAG Document Assistant</title>
<style>body{font-family:system-ui,sans-serif;max-width:900px;margin:36px auto;padding:0 18px;color:#172033}h1{margin-bottom:4px}.card{border:1px solid #dbe2ef;border-radius:12px;padding:18px;margin:16px 0}input,textarea,button{font:inherit}input,textarea{width:100%;box-sizing:border-box;margin:8px 0;padding:10px;border:1px solid #b9c5d6;border-radius:8px}textarea{height:96px}button{background:#155eef;color:white;border:0;border-radius:8px;padding:10px 15px;cursor:pointer}pre{white-space:pre-wrap;background:#f6f8fb;padding:14px;border-radius:8px}.source{border-left:3px solid #7c9aff;padding-left:10px;margin:10px 0}</style></head>
<body><h1>RAG Document Assistant</h1><p>Upload a PDF or TXT file, then ask grounded questions. Answers include source excerpts.</p>
<div class="card"><h2>1. Upload</h2><input id="file" type="file" accept=".pdf,.txt"><button onclick="upload()">Index document</button><pre id="uploadResult"></pre></div>
<div class="card"><h2>2. Ask</h2><textarea id="question" placeholder="Ask a question about the uploaded documents..."></textarea><button onclick="ask()">Ask</button><div id="answer"></div></div>
<script>
async function upload(){const f=document.getElementById('file').files[0];if(!f)return;const fd=new FormData();fd.append('file',f);const r=await fetch('/upload',{method:'POST',body:fd});document.getElementById('uploadResult').textContent=JSON.stringify(await r.json(),null,2)}
async function ask(){const q=document.getElementById('question').value;const r=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:q})});const d=await r.json();let html='<h3>Answer</h3><pre>'+escapeHtml(d.answer)+'</pre><h3>Sources</h3>';html+=d.sources.map(s=>'<div class="source"><strong>'+escapeHtml(s.document)+(s.page?' — page '+s.page:'')+'</strong><br>'+escapeHtml(s.excerpt)+'</div>').join('')||'<p>No sources found.</p>';document.getElementById('answer').innerHTML=html}
function escapeHtml(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}
</script></body></html>'''
