"""AI CSV Automation Tool.

Run:
  pip install -r requirements.txt
  uvicorn ai_automation_tool:app --reload --port 8001
Open http://127.0.0.1:8001

Upload a CSV, choose the columns to analyse, and receive an auditable CSV with a
classification, short summary, confidence, provider/model, prompt version,
timestamp, source hash, raw model output, and any parsing error.

The OpenAI path uses few-shot examples for a structured JSON response. It asks
for a concise final label and evidence, not private chain-of-thought.
Without OPENAI_API_KEY, the local deterministic classifier still works.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse
from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

APP_DIR = Path(__file__).resolve().parent
DATA_DIR = APP_DIR / "automation_data"
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
for directory in (UPLOAD_DIR, OUTPUT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

PROMPT_VERSION = "support-triage-v1"
DEFAULT_MODEL = "gpt-4o-mini"
CATEGORIES = ["billing", "account_access", "bug", "feature_request", "general"]
KEYWORDS = {
    "billing": ["payment", "charged", "invoice", "billing", "refund", "subscription", "unpaid"],
    "account_access": ["password", "login", "sign in", "reset", "access", "email link", "locked"],
    "bug": ["bug", "error", "broken", "blank", "crash", "does not work", "failed"],
    "feature_request": ["feature", "add", "would like", "could you", "request", "export"],
}

app = FastAPI(title="Auditable AI CSV Automation Tool")
JOBS: dict[str, dict[str, Any]] = {}


class ProcessRequest(BaseModel):
    job_id: str
    text_columns: list[str] = Field(min_length=1)
    provider: str = Field(default="local", pattern="^(local|openai)$")
    model: str = DEFAULT_MODEL
    max_rows: int = Field(default=500, ge=1, le=5000)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def source_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def make_text(row: pd.Series, columns: list[str]) -> str:
    return " | ".join(f"{column}: {str(row.get(column, '')).strip()}" for column in columns).strip()


def local_classify(text: str) -> tuple[dict[str, Any], str]:
    lower = text.lower()
    scores = {category: sum(word in lower for word in words) for category, words in KEYWORDS.items()}
    category = max(scores, key=scores.get)
    if scores[category] == 0:
        category = "general"
    words = re.findall(r"\S+", re.sub(r"\s+", " ", text))
    summary = " ".join(words[:36])
    if len(words) > 36:
        summary += "…"
    confidence = round(min(0.95, 0.45 + scores.get(category, 0) * 0.15), 2)
    output = {
        "category": category,
        "summary": summary or "No analysable text.",
        "confidence": confidence,
        "evidence": [word for word in KEYWORDS.get(category, []) if word in lower][:3],
    }
    return output, json.dumps(output, ensure_ascii=False)


def openai_classify(text: str, model: str) -> tuple[dict[str, Any], str]:
    if OpenAI is None or not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not configured.")
    system = f"""You are a careful support-ticket triage assistant.
Return ONLY valid JSON with exactly these keys: category, summary, confidence, evidence.
category must be one of: {', '.join(CATEGORIES)}.
summary must be a factual sentence no longer than 35 words.
confidence must be a number from 0 to 1.
evidence must be a short list of words or phrases taken from the input.
Do not provide chain-of-thought, hidden reasoning, or extra keys.

Few-shot examples:
Input: Subject: Payment failed | Message: I was charged but my subscription remains unpaid.
Output: {{"category":"billing","summary":"The customer was charged but their subscription still appears unpaid.","confidence":0.94,"evidence":["charged","subscription","unpaid"]}}

Input: Subject: Password reset | Message: The reset email link has expired.
Output: {{"category":"account_access","summary":"The customer cannot reset their password because the email link expired.","confidence":0.97,"evidence":["password reset","email link expired"]}}"""
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    response = client.chat.completions.create(
        model=model,
        temperature=0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Input: {text}"},
        ],
    )
    raw = response.choices[0].message.content or "{}"
    parsed = json.loads(raw)
    category = parsed.get("category", "general")
    if category not in CATEGORIES:
        category = "general"
    parsed["category"] = category
    parsed["summary"] = str(parsed.get("summary", ""))[:500]
    try:
        parsed["confidence"] = max(0.0, min(1.0, float(parsed.get("confidence", 0))))
    except (TypeError, ValueError):
        parsed["confidence"] = 0.0
    if not isinstance(parsed.get("evidence"), list):
        parsed["evidence"] = []
    return parsed, raw


def process_dataframe(frame: pd.DataFrame, text_columns: list[str], provider: str, model: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for index, row in frame.iterrows():
        input_text = make_text(row, text_columns)
        record = row.to_dict()
        record.update({
            "audit_row_id": str(uuid.uuid4()),
            "audit_timestamp_utc": utc_now(),
            "audit_prompt_version": PROMPT_VERSION,
            "audit_provider": provider,
            "audit_model": model if provider == "openai" else "local-keyword-v1",
            "audit_source_hash_sha256": source_hash(input_text),
            "audit_input_text": input_text,
        })
        try:
            output, raw = openai_classify(input_text, model) if provider == "openai" else local_classify(input_text)
            record.update({
                "ai_category": output.get("category", "general"),
                "ai_summary": output.get("summary", ""),
                "ai_confidence": output.get("confidence", 0),
                "ai_evidence": json.dumps(output.get("evidence", []), ensure_ascii=False),
                "audit_raw_model_output": raw,
                "audit_status": "success",
                "audit_error": "",
            })
        except Exception as exc:
            record.update({
                "ai_category": "",
                "ai_summary": "",
                "ai_confidence": "",
                "ai_evidence": "[]",
                "audit_raw_model_output": "",
                "audit_status": "error",
                "audit_error": str(exc),
            })
        records.append(record)
    return pd.DataFrame(records)


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return HTML


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "openai_available": bool(OpenAI and os.getenv("OPENAI_API_KEY")), "prompt_version": PROMPT_VERSION}


@app.post("/upload-csv")
def upload_csv(file: UploadFile = File(...)) -> dict[str, Any]:
    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Upload a CSV file.")
    job_id = uuid.uuid4().hex
    path = UPLOAD_DIR / f"{job_id}_{Path(file.filename).name}"
    with path.open("wb") as destination:
        destination.write(file.file.read())
    try:
        frame = pd.read_csv(path)
    except Exception as exc:
        path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"Could not read CSV: {exc}") from exc
    JOBS[job_id] = {"path": path, "name": file.filename, "columns": list(frame.columns), "rows": len(frame)}
    return {"job_id": job_id, "file": file.filename, "rows": len(frame), "columns": list(frame.columns)}


@app.post("/process")
def process(request: ProcessRequest) -> dict[str, Any]:
    job = JOBS.get(request.job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Unknown job_id. Upload a CSV first.")
    invalid = [column for column in request.text_columns if column not in job["columns"]]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Unknown text columns: {invalid}")
    frame = pd.read_csv(job["path"]).head(request.max_rows)
    result = process_dataframe(frame, request.text_columns, request.provider, request.model)
    output_name = f"audit_{request.job_id}.csv"
    output_path = OUTPUT_DIR / output_name
    result.to_csv(output_path, index=False)
    job["output_path"] = output_path
    job["processed_rows"] = len(result)
    return {
        "message": "Processing complete.",
        "processed_rows": len(result),
        "status_counts": result["audit_status"].value_counts().to_dict(),
        "download_url": f"/download/{request.job_id}",
        "preview": result.head(5).to_dict(orient="records"),
    }


@app.get("/download/{job_id}")
def download(job_id: str) -> FileResponse:
    job = JOBS.get(job_id)
    if not job or "output_path" not in job:
        raise HTTPException(status_code=404, detail="No processed output exists for this job.")
    return FileResponse(job["output_path"], media_type="text/csv", filename=job["output_path"].name)


HTML = r'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Auditable AI CSV Automation</title>
<style>body{font-family:system-ui,sans-serif;max-width:960px;margin:36px auto;padding:0 18px;color:#172033}.card{border:1px solid #dbe2ef;border-radius:12px;padding:18px;margin:16px 0}button{background:#155eef;color:white;border:0;border-radius:8px;padding:10px 15px;cursor:pointer;margin:8px 8px 8px 0}select{min-height:120px;width:100%;padding:8px}pre{white-space:pre-wrap;max-height:360px;overflow:auto;background:#f6f8fb;padding:14px;border-radius:8px}.muted{color:#536075}</style></head>
<body><h1>Auditable AI CSV Automation</h1><p>Upload a CSV, select text columns, classify and summarise rows, then download an audit-ready output CSV.</p>
<div class="card"><h2>1. Upload CSV</h2><input id="file" type="file" accept=".csv"><button onclick="uploadCsv()">Upload</button><pre id="upload"></pre></div>
<div class="card"><h2>2. Process</h2><label>Text columns (Ctrl/Cmd-click for more than one)</label><select id="columns" multiple></select><br><label><input type="radio" name="provider" value="local" checked> Local deterministic model</label><br><label><input type="radio" name="provider" value="openai"> OpenAI API model</label><br><button onclick="processCsv()">Classify and summarise</button><a id="download" hidden><button>Download audit CSV</button></a><pre id="result"></pre></div>
<p class="muted">Each output row includes input hash, provider/model, prompt version, timestamp, raw response, parsed fields and errors.</p>
<script>let jobId=null;async function uploadCsv(){const file=document.getElementById('file').files[0];if(!file)return;const fd=new FormData();fd.append('file',file);const r=await fetch('/upload-csv',{method:'POST',body:fd});const d=await r.json();document.getElementById('upload').textContent=JSON.stringify(d,null,2);if(!r.ok)return;jobId=d.job_id;const s=document.getElementById('columns');s.innerHTML=d.columns.map(c=>'<option value="'+esc(c)+'">'+esc(c)+'</option>').join('')}async function processCsv(){if(!jobId)return;const cols=[...document.getElementById('columns').selectedOptions].map(x=>x.value);if(!cols.length)return;const provider=document.querySelector('input[name="provider"]:checked').value;const r=await fetch('/process',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({job_id:jobId,text_columns:cols,provider:provider})});const d=await r.json();document.getElementById('result').textContent=JSON.stringify(d,null,2);if(r.ok){const a=document.getElementById('download');a.href=d.download_url;a.hidden=false}}function esc(s){return String(s).replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]))}</script></body></html>'''
