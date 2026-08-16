# AI Portfolio: RAG and Automation

This repository contains two beginner-friendly AI/software portfolio projects built with Python and FastAPI.

## 1. RAG Document Assistant

`rag_document_assistant.py` uploads PDF or TXT files, chunks the extracted text, stores local embeddings in ChromaDB, retrieves relevant excerpts, and returns source-grounded answers.

Run it:

```bash
pip install -r requirements.txt
uvicorn rag_document_assistant:app --reload
```

Open `http://127.0.0.1:8000`.

Main endpoints: `GET /health`, `POST /upload`, `POST /ask`, and `POST /evaluate`.

## 2. Auditable AI CSV Automation Tool

`ai_automation_tool.py` ingests a CSV, classifies and summarises selected text columns, and creates a downloadable audit CSV. The audit output keeps the source text hash, timestamp, provider/model, prompt version, raw model response, parsed result, status, and error field for every row.

Run it separately on port 8001:

```bash
uvicorn ai_automation_tool:app --reload --port 8001
```

Open `http://127.0.0.1:8001`, upload `examples/sample_support_tickets.csv`, select `subject` and `message`, then process the file.

### Processing modes

- **Local deterministic mode:** runs with no API key. It uses transparent keyword rules and extractive summaries, making it useful for testing the workflow and audit output.
- **OpenAI mode:** set `OPENAI_API_KEY`, select OpenAI in the interface, and the app uses a few-shot prompt to return strict JSON: a category, concise summary, confidence, and evidence phrases.

```bash
export OPENAI_API_KEY="your-key"
export OPENAI_MODEL="gpt-4o-mini"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your-key"
$env:OPENAI_MODEL="gpt-4o-mini"
```

The prompt deliberately requests final structured outputs rather than private chain-of-thought. The audit log records observable input/output evidence so results can be reviewed.

## Shared setup

```bash
pip install -r requirements.txt
```

The `.gitignore` excludes `.env`, local vector stores, uploaded documents, generated outputs, virtual environments, and caches. Never commit API keys or private documents.
