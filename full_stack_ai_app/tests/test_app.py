import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.main import app, Base, get_db, hash_password, User, Draft

SQLALCHEMY_DATABASE_URL = "sqlite://"

test_engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
)
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
Base.metadata.create_all(bind=test_engine)


def override_get_db():
    db = TestingSessionLocal()
    try:
        yield db
    finally:
        db.close()


app.dependency_overrides[get_db] = override_get_db

client = TestClient(app)


def test_register_and_login():
    # Register
    resp = client.post("/register", data={"email": "test@example.com", "password": "password123"}, follow_redirects=False)
    assert resp.status_code == 302
    assert "session" in resp.cookies
    session_cookie = resp.cookies["session"]
    
    # Access dashboard with session
    resp = client.get("/dashboard", cookies={"session": session_cookie})
    assert resp.status_code == 200
    assert "test@example.com" in resp.text


def test_login_rejects_wrong_password():
    client.post("/register", data={"email": "user2@example.com", "password": "password123"})
    resp = client.post("/login", data={"email": "user2@example.com", "password": "wrong"}, follow_redirects=False)
    assert resp.status_code == 400
    assert "Invalid credentials" in resp.text


def test_draft_creation_and_isolation():
    # User A
    resp_a = client.post("/register", data={"email": "a@example.com", "password": "password123"}, follow_redirects=False)
    cookie_a = resp_a.cookies["session"]
    
    # User B
    resp_b = client.post("/register", data={"email": "b@example.com", "password": "password123"}, follow_redirects=False)
    cookie_b = resp_b.cookies["session"]
    
    # User A creates draft
    resp = client.post("/drafts", data={"task_type": "professional_email", "input_text": "Follow up with client about proposal"}, cookies={"session": cookie_a}, follow_redirects=False)
    assert resp.status_code == 302
    
    # User A sees their draft
    resp = client.get("/dashboard", cookies={"session": cookie_a})
    assert "Follow up with client" in resp.text
    
    # User B does NOT see User A's draft
    resp = client.get("/dashboard", cookies={"session": cookie_b})
    assert "Follow up with client" not in resp.text
    
    # User B creates their own draft
    resp = client.post("/drafts", data={"task_type": "lesson_resource", "input_text": "Photosynthesis lesson"}, cookies={"session": cookie_b}, follow_redirects=False)
    assert resp.status_code == 302
    
    # Verify isolation
    resp = client.get("/dashboard", cookies={"session": cookie_a})
    assert "Photosynthesis lesson" not in resp.text
    resp = client.get("/dashboard", cookies={"session": cookie_b})
    assert "Photosynthesis lesson" in resp.text


def test_local_fallback_generates_output():
    resp = client.post("/register", data={"email": "c@example.com", "password": "password123"}, follow_redirects=False)
    cookie = resp.cookies["session"]
    resp = client.post("/drafts", data={"task_type": "professional_email", "input_text": "Test notes for fallback"}, cookies={"session": cookie}, follow_redirects=False)
    assert resp.status_code == 302
    resp = client.get("/dashboard", cookies={"session": cookie})
    assert "Test notes for fallback" in resp.text
    assert "professional_email" in resp.text or "Professional Email" in resp.text


def test_logout_clears_session():
    resp = client.post("/register", data={"email": "d@example.com", "password": "password123"}, follow_redirects=False)
    cookie = resp.cookies["session"]
    resp = client.post("/logout", cookies={"session": cookie}, follow_redirects=False)
    assert resp.status_code == 302
    assert "session" not in resp.cookies
    resp = client.get("/dashboard", cookies={"session": cookie})
    assert resp.status_code in (302, 401)


def test_password_hashing():
    hashed = hash_password("mysecret")
    assert hashed != "mysecret"
    assert hash_password("mysecret") != hashed  # bcrypt salt differs
    from passlib.context import CryptContext
    ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
    assert ctx.verify("mysecret", hashed)
    assert not ctx.verify("wrong", hashed)


if __name__ == "__main__":
    pytest.main(["-v", __file__])
