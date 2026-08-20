# Full-Stack AI Workspace

A small, portfolio-ready full-stack AI application with authentication, database persistence, API integration, tests, Docker support, and deployment instructions.

## Features

- **User authentication**: register, login, logout with bcrypt password hashing and HttpOnly session cookies.
- **Per-user data isolation**: each user only sees their own AI-generated drafts.
- **AI draft generation**: choose a task type (professional email, lesson resource, cover letter) and provide rough notes; the app returns a polished draft.
- **Optional OpenAI integration**: set `OPENAI_API_KEY` for GPT-powered generation; works offline with a deterministic local fallback.
- **Few-shot structured prompts**: prompt templates request concise final outputs only -- no hidden chain-of-thought.
- **SQLite + SQLAlchemy** for simple local persistence; easy to swap to PostgreSQL.
- **Automated tests** with pytest covering registration, authentication, draft creation, and user isolation.
- **Docker & Docker Compose** for containerised deployment.
- **Render / Railway** one-click deploy ready.

## Quick Start (Local)

```bash
cd full_stack_ai_app
cp .env.example .env
# Edit .env and add a long random APP_SECRET_KEY
# Optionally add OPENAI_API_KEY for GPT generation

pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open <http://127.0.0.1:8000>. Register an account, then create drafts.

## Run with Docker

```bash
docker compose up --build
```

Open <http://localhost:8000>.

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `APP_SECRET_KEY` | **Yes** | Long random string for session signing. Generate with `python -c "import secrets; print(secrets.token_hex(32))"`. |
| `OPENAI_API_KEY` | No | OpenAI API key for GPT generation. Leave empty to use local fallback. |
| `OPENAI_MODEL` | No | Model name (default `gpt-4o-mini`). |
| `DATABASE_URL` | No | SQLAlchemy database URL (default SQLite file `./data/app.db`). |

## Running Tests

```bash
pip install -r requirements.txt
pytest -v
```

## Deployment

### Render

1. Push this branch to GitHub.
2. In Render dashboard: **New → Web Service** → connect your repo.
3. Build command: `pip install -r full_stack_ai_app/requirements.txt`
4. Start command: `uvicorn full_stack_ai_app.app.main:app --host 0.0.0.0 --port $PORT`
5. Add environment variables: `APP_SECRET_KEY`, `OPENAI_API_KEY` (optional), `DATABASE_URL` (use Render PostgreSQL for production).
6. Create a **Disk** mounted at `/app/data` for SQLite persistence, or switch `DATABASE_URL` to the managed PostgreSQL URL.

### Railway

1. New Project → Deploy from GitHub repo → select this branch.
2. Railway auto-detects Dockerfile; or use Nixpacks with the start command above.
3. Add variables in the Variables tab.
4. Attach a PostgreSQL plugin and set `DATABASE_URL` to the provided connection string.

## Project Structure

```
full_stack_ai_app/
├── app/
│   ├── main.py              # FastAPI application, models, routes
│   ├── templates/           # Jinja2 HTML templates
│   │   ├── base.html
│   │   ├── register.html
│   │   ├── login.html
│   │   └── dashboard.html
│   └── static/
│       └── styles.css
├── tests/
│   └── test_app.py          # pytest suite
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .gitignore
└└── README.md
```

## Portfolio Talking Points

- **Authentication & authorization** with secure password hashing, session cookies, and per-user data isolation.
- **Database design** with SQLAlchemy ORM, relationships, and migrations-ready structure.
- **AI integration** using OpenAI SDK with structured few-shot prompts, and a deterministic offline fallback for reliability.
- **Testing** with FastAPI TestClient, in-memory SQLite, and isolation checks.
- **Containerisation** with multi-stage Dockerfile and Compose for local parity.
- **Production awareness**: environment-based config, secret management, HttpOnly cookies, and deployment docs for managed platforms.

## License

MIT
