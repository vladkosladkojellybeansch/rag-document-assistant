import os
import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

os.environ["APP_SECRET_KEY"] = "test-secret-key-for-testing-only"
os.environ["OPENAI_API_KEY"] = ""
os.environ["DATABASE_URL"] = "sqlite:///./test.db"

from app.main import app, Base, engine, SessionLocal, User, Draft, pwd_context, create_session_token  # noqa: E402

@pytest.fixture(scope="function")
def db_session():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    with TestClient(app) as c:
        yield c


def test_register_and_login(client):
    resp = client.post("/register", data={"email": "test@example.com", "password": "password123"}, follow_redirects=False)
    assert resp.status_code == 303
    assert "session" in resp.cookies
    session_cookie = resp.cookies["session"]
    resp2 = client.get("/dashboard", cookies={"session": session_cookie})
    assert resp2.status_code == 200
    assert "test@example.com" in resp2.text


def test_login_rejects_wrong_password(client):
    client.post("/register", data={"email": "test2@example.com", "password": "password123"})
    resp = client.post("/login", data={"email": "test2@example.com", "password": "wrong"}, follow_redirects=False)
    assert resp.status_code == 401
    assert "Invalid credentials" in resp.text


def test_dashboard_requires_auth(client):
    resp = client.get("/dashboard", follow_redirects=False)
    assert resp.status_code in (303, 401)


def test_draft_generation_and_isolation(client, db_session):
    # Register two users
    client.post("/register", data={"email": "a@example.com", "password": "password123"})
    resp_a = client.post("/login", data={"email": "a@example.com", "password": "password123"}, follow_redirects=False)
    cookie_a = resp_a.cookies["session"]

    client.post("/register", data={"email": "b@example.com", "password": "password123"})
    resp_b = client.post("/login", data={"email": "b@example.com", "password": "password123"}, follow_redirects=False)
    cookie_b = resp_b.cookies["session"]

    # User A creates a draft
    resp = client.post("/draft", data={"task_type": "professional_email", "input_notes": "Meeting tomorrow 10am. Confirm budget."}, cookies={"session": cookie_a}, follow_redirects=False)
    assert resp.status_code == 303

    # User A sees their draft
    dash_a = client.get("/dashboard", cookies={"session": cookie_a})
    assert "Meeting tomorrow" in dash_a.text
    assert "Confirm budget" in dash_a.text

    # User B does NOT see user A's draft
    dash_b = client.get("/dashboard", cookies={"session": cookie_b})
    assert "Meeting tomorrow" not in dash_b.text

    # Direct DB check
    user_a = db_session.query(User).filter_by(email="a@example.com").first()
    user_b = db_session.query(User).filter_by(email="b@example.com").first()
    drafts_a = db_session.query(Draft).filter_by(user_id=user_a.id).all()
    drafts_b = db_session.query(Draft).filter_by(user_id=user_b.id).all()
    assert len(drafts_a) == 1
    assert len(drafts_b) == 0


def test_logout_clears_session(client):
    client.post("/register", data={"email": "logout@example.com", "password": "password123"})
    resp = client.post("/login", data={"email": "logout@example.com", "password": "password123"}, follow_redirects=False)
    session_cookie = resp.cookies["session"]
    resp2 = client.post("/logout", cookies={"session": session_cookie}, follow_redirects=False)
    assert resp2.status_code == 303
    assert "session" not in resp2.cookies


def test_local_fallback_when_no_openai_key(client, db_session):
    client.post("/register", data={"email": "local@example.com", "password": "password123"})
    resp = client.post("/login", data={"email": "local@example.com", "password": "password123"}, follow_redirects=False)
    cookie = resp.cookies["session"]
    resp2 = client.post("/draft", data={"task_type": "professional_email", "input_notes": "Test notes"}, cookies={"session": cookie}, follow_redirects=False)
    assert resp2.status_code == 303
    dash = client.get("/dashboard", cookies={"session": cookie})
    assert "Local fallback" in dash.text or "offline response" in dash.text.lower()
