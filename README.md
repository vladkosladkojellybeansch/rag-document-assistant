# RAG Document Assistant

A Python FastAPI application for uploading PDF/TXT documents and answering questions from retrieved evidence. It uses local `sentence-transformers` embeddings and persistent ChromaDB storage, displays source excerpts, and can optionally use an OpenAI model to synthesize a concise evidence-grounded answer.

## Features

- Upload PDF and TXT documents
- Extract and chunk document text with overlap
- Create local embeddings using `all-MiniLM-L6-v2`
- Store and search vectors in ChromaDB
- Simple web interface and FastAPI endpoints
- Show document name, page number, and retrieved excerpts
- Optional LLM synthesis with few-shot answer-format examples
- Do not request or reveal chain-of-thought; provide concise sourced answers instead
- Evaluate answers using expected terms and groundedness

## Setup

```bash
pip install -r requirements.txt
uvicorn rag_document_assistant:app --reload
```

Open `http://127.0.0.1:8000`.

## Optional LLM setup

The app works without an API key by returning retrieved evidence. To enable generated grounded answers:

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

## API

- `GET /health` — service status and indexed chunk count
- `POST /upload` — multipart upload of a `.pdf` or `.txt` file
- `POST /ask` — JSON: `{"question": "...", "top_k": 5}`
- `POST /evaluate` — JSON list of `{"question": "...", "expected_answer_contains": ["term"]}`

## Evaluation example

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '[{"question":"What is the refund deadline?","expected_answer_contains":["30 days"]}]'
```

## Security

Do not commit `.env`, API keys, uploaded documents, or the local `data/` vector-store directory. They are excluded by `.gitignore`.
