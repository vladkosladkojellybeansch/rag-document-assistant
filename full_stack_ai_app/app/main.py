"""Full-Stack AI Workspace with Authentication.

Run locally:
  cd full_stack_ai_app
  cp .env.example .env
  pip install -r requirements.txt
  uvicorn app.main:app --reload

Open http://127.0.0.1:8000
"""

from __future__ import annotations

import hashlib
import os
import secrets
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from passlib.context import CryptContext
from pydantic import BaseModel, EmailStr
from sqlalchemy import Column, DateTime, String, Text, create_engine, ForeignKey
from sqlalchemy.orm import DeclarativeBase, Session, relationship, sessionmaker

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Configuration
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DATA_DIR}/app.db")
SECRET_KEY = os.getenv("APP_SECRET_KEY", secrets.token_hex(32))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

app = FastAPI(title="Full-Stack AI Workspace")
app.mount("/static", StaticFiles(directory=BASE_DIR / "app" / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "app" / "templates")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    drafts = relationship("Draft", back_populates="owner", cascade="all, delete-orphan")


class Draft(Base):
    __tablename__ = "drafts"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String(36), ForeignKey("users.id"), nullable=False, index=True)
    task_type = Column(String(50), nullable=False)
    input_text = Column(Text, nullable=False)
    output_text = Column(Text, nullable=False)
    provider = Column(String(20), nullable=False)
    model = Column(String(50), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    owner = relationship("User", back_populates="drafts")


Base.metadata.create_all(bind=engine)


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


def create_session_token() -> str:
    return secrets.token_urlsafe(32)


def get_current_user(request: Request, db: Session = Depends(get_db)) -> User | None:
    token = request.cookies.get("session")
    if not token:
        return None
    user = db.query(User).filter(User.id == token).first()
    return user


def require_user(request: Request, db: Session = Depends(get_db)) -> User:
    user = get_current_user(request, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


class RegisterRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class DraftRequest(BaseModel):
    task_type: str
    input_text: str


TASK_PROMPTS = {
    "professional_email": """You are an expert business communicator.
Turn the user's rough notes into a polished, professional email.
Return ONLY the final email text -- no chain-of-thought, no commentary.

Few-shot examples:
Notes: Follow up with client John about proposal sent last week. Ask if they have questions. Propose meeting Thursday 2pm.
Email:
Subject: Follow-up on Proposal -- Thursday 2pm?

Hi John,

I wanted to follow up on the proposal I sent last week. Do you have any questions or would you like to discuss? I'm available Thursday at 2pm for a quick call.

Best regards,
[Your Name]

Notes: Thank interviewer Sarah for phone screen yesterday. Reiterate interest in Senior Developer role. Ask about next steps.
Email:
Subject: Thank you -- Senior Developer Phone Screen

Hi Sarah,

Thank you for the conversation yesterday. I remain very interested in the Senior Developer role and enjoyed learning more about the team. Could you let me know what the next steps look like?

Best,
[Your Name]""",
    "lesson_resource": """You are an experienced curriculum designer.
Create a concise, classroom-ready lesson resource from the teacher's outline.
Return ONLY the formatted resource -- no chain-of-thought, no commentary.

Few-shot examples:
Outline: Topic: Photosynthesis, Grade: 9, Duration: 50 min, Objectives: explain light-dependent reactions, Key vocabulary: chlorophyll, thylakoid, ATP, NADPH, Activities: diagram labelling, mini-lecture, exit ticket question.
Resource:
# Lesson Resource: Photosynthesis (Grade 9, 50 min)

**Learning Objectives**
- Explain the light-dependent reactions of photosynthesis.
- Identify the role of chlorophyll, thylakoids, ATP, and NADPH.

**Key Vocabulary**
chlorophyll, thylakoid, ATP, NADPH

**Lesson Flow**
1. **Hook (5 min)** -- Show a time-lapse of a plant growing; ask "Where does the energy come from?"
2. **Mini-lecture (15 min)** -- Light-dependent reactions overview with diagram.
3. **Guided Practice (20 min)** -- Students label thylakoid diagram in pairs.
4. **Exit Ticket (10 min)** -- "Describe in one sentence how light energy becomes chemical energy."

**Materials**
- Projector, printed thylakoid diagrams, exit-ticket slips.""",
    "job_cover_letter": """You are a career coach.
Write a tailored cover letter from the user's bullet points.
Return ONLY the cover letter -- no chain-of-thought, no commentary.

Few-shot examples:
Bullets: Applying for ML Engineer at Acme Corp. 3 years PyTorch experience. Built recommendation system serving 10M users. Published paper on efficient transformers. Passionate about scalable ML.
Cover Letter:
Dear Hiring Manager,

I am writing to apply for the ML Engineer position at Acme Corp. With three years of PyTorch experience, I recently built a recommendation system serving 10 million users and published a paper on efficient transformers. I am passionate about deploying scalable ML solutions and would welcome the opportunity to contribute to your team.

Sincerely,
[Your Name]""",
}

DEFAULT_TASKS = list(TASK_PROMPTS.keys())


def generate_draft_local(task_type: str, input_text: str) -> str:
    """Deterministic offline fallback for demo/testing without API key."""
    templates = {
        "professional_email": f"Subject: Follow-up on your notes\n\nDear recipient,\n\nBased on your input: {input_text[:200]}...\n\nI will follow up shortly.\n\nBest regards,\n[Your Name]",
        "lesson_resource": f"# Lesson Resource (Offline Draft)\n\n**Topic**: {input_text[:100]}\n\n**Objectives**:\n- Understand key concepts\n- Apply knowledge\n\n**Activities**:\n1. Introduction\n2. Guided practice\n3. Assessment\n",
        "job_cover_letter": f"Dear Hiring Manager,\n\nI am writing to express my interest based on: {input_text[:200]}...\n\nMy background aligns well with this role.\n\nSincerely,\n[Your Name]",
    }
    return templates.get(task_type, f"[Offline draft for {task_type}]\n{input_text}")


def generate_draft_openai(task_type: str, input_text: str) -> str:
    if OpenAI is None or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI not configured")
    system_prompt = TASK_PROMPTS.get(task_type, "You are a helpful assistant.")
    client = OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        temperature=0.2,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Notes: {input_text}"},
        ],
    )
    return response.choices[0].message.content.strip()


def generate_draft(task_type: str, input_text: str) -> tuple[str, str, str]:
    """Returns (output_text, provider, model)."""
    if OPENAI_API_KEY and OpenAI is not None:
        try:
            output = generate_draft_openai(task_type, input_text)
            return output, "openai", OPENAI_MODEL
        except Exception:
            pass
    output = generate_draft_local(task_type, input_text)
    return output, "local", "deterministic-fallback"


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return RedirectResponse(url="/dashboard")


@app.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.post("/register")
async def register(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    if db.query(User).filter(User.email == email).first():
        return templates.TemplateResponse("register.html", {"request": request, "error": "Email already registered"}, status_code=400)
    user = User(email=email, hashed_password=hash_password(password))
    db.add(user)
    db.commit()
    db.refresh(user)
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="session", value=user.id, httponly=True, samesite="lax")
    return response


@app.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@app.post("/login")
async def login(request: Request, email: str = Form(...), password: str = Form(...), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == email).first()
    if not user or not verify_password(password, user.hashed_password):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid credentials"}, status_code=400)
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="session", value=user.id, httponly=True, samesite="lax")
    return response


@app.post("/logout")
async def logout():
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie(key="session")
    return response


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request, user: User = Depends(require_user), db: Session = Depends(get_db)):
    drafts = db.query(Draft).filter(Draft.user_id == user.id).order_by(Draft.created_at.desc()).all()
    return templates.TemplateResponse("dashboard.html", {"request": request, "user": user, "drafts": drafts, "tasks": DEFAULT_TASKS})


@app.post("/drafts")
async def create_draft(request: Request, task_type: str = Form(...), input_text: str = Form(...), user: User = Depends(require_user), db: Session = Depends(get_db)):
    if task_type not in TASK_PROMPTS:
        raise HTTPException(status_code=400, detail="Unknown task type")
    output_text, provider, model = generate_draft(task_type, input_text)
    draft = Draft(
        user_id=user.id,
        task_type=task_type,
        input_text=input_text,
        output_text=output_text,
        provider=provider,
        model=model,
    )
    db.add(draft)
    db.commit()
    db.refresh(draft)
    return RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)


@app.get("/health")
async def health():
    return {"status": "ok", "openai_configured": bool(OPENAI_API_KEY and OpenAI)}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
