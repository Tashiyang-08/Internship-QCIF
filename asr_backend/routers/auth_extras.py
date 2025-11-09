# asr_backend/routers/auth_extras.py
from __future__ import annotations
import os, secrets, sqlite3, smtplib
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, EmailStr
from typing import Optional

# ---- ENV (configure as needed) ----
SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
MAIL_FROM = os.getenv("MAIL_FROM", "no-reply@example.com")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")
DEBUG_MAIL_LINKS = os.getenv("DEBUG_MAIL_LINKS", "1") == "1"

# Your app likely already has these:
#   - get_db(): returns sqlite3.Connection
#   - auth dependency get_current_user()
#   - users table with: id, email, full_name, password_hash, is_verified
# If "is_verified" doesn’t exist, we’ll add it at router startup.

def _get_db(request: Request) -> sqlite3.Connection:
    # main.py should set request.state.db = sqlite conn per request
    return request.state.db

def _ensure_schema(db: sqlite3.Connection):
    # Tokens table for email verification + reset
    db.execute("""
    CREATE TABLE IF NOT EXISTS auth_tokens(
      id INTEGER PRIMARY KEY AUTOINCREMENT,
      user_id INTEGER NOT NULL,
      kind TEXT NOT NULL,              -- 'verify' | 'reset'
      token TEXT NOT NULL UNIQUE,
      expires_at TEXT NOT NULL
    )
    """)
    # is_verified column if missing
    cols = [c[1] for c in db.execute("PRAGMA table_info(users)").fetchall()]
    if "is_verified" not in cols:
        db.execute("ALTER TABLE users ADD COLUMN is_verified INTEGER DEFAULT 0")
    db.commit()

def _insert_token(db: sqlite3.Connection, user_id: int, kind: str, minutes: int = 60*24) -> str:
    token = secrets.token_urlsafe(32)
    exp = (datetime.now(timezone.utc) + timedelta(minutes=minutes)).isoformat()
    db.execute(
        "INSERT INTO auth_tokens(user_id, kind, token, expires_at) VALUES(?,?,?,?)",
        (user_id, kind, token, exp)
    )
    db.commit()
    return token

def _send_email(to: str, subject: str, body_html: str, body_text: Optional[str] = None):
    if not (SMTP_HOST and SMTP_USER and SMTP_PASS):
        # Dev mode: just log
        print("=== DEV EMAIL ===")
        print("TO:", to)
        print("SUBJECT:", subject)
        print(body_text or body_html)
        print("=================")
        return
    msg = EmailMessage()
    msg["From"] = MAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(body_text or "See HTML version.")
    msg.add_alternative(body_html, subtype="html")
    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
        s.starttls()
        s.login(SMTP_USER, SMTP_PASS)
        s.send_message(msg)

router = APIRouter(prefix="/auth", tags=["auth-extras"])

@router.on_event("startup")
def _startup():
    # Will be executed when included by FastAPI app (main imports & includes router)
    pass

# Request models
class EmailOnly(BaseModel):
    email: EmailStr

class ResetBody(BaseModel):
    token: str
    new_password: str

@router.post("/send-verification")
def send_verification_link(body: EmailOnly, request: Request):
    db = _get_db(request)
    _ensure_schema(db)
    row = db.execute("SELECT id, email, is_verified FROM users WHERE email = ?", (body.email,)).fetchone()
    if not row:
        # For privacy, do not leak existence. Respond ok.
        return {"ok": True}
    if row[2]:  # is_verified
        return {"ok": True}

    token = _insert_token(db, user_id=row[0], kind="verify", minutes=60*24*3)
    link = f"{FRONTEND_BASE_URL}/#/verify?token={token}"
    _send_email(
        to=row[1],
        subject="Verify your email",
        body_html=f"<p>Click to verify your email:</p><p><a href='{link}'>{link}</a></p>",
        body_text=f"Verify your email: {link}",
    )
    return {"ok": True, "link": link} if DEBUG_MAIL_LINKS else {"ok": True}

@router.get("/verify")
def verify_email(token: str, request: Request):
    db = _get_db(request)
    _ensure_schema(db)
    t = db.execute("SELECT user_id, expires_at FROM auth_tokens WHERE token=? AND kind='verify'", (token,)).fetchone()
    if not t:
        raise HTTPException(status_code=400, detail="Invalid token")
    if datetime.fromisoformat(t[1]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Token expired")

    db.execute("UPDATE users SET is_verified=1 WHERE id=?", (t[0],))
    db.execute("DELETE FROM auth_tokens WHERE token=?", (token,))
    db.commit()
    return {"ok": True}

@router.post("/request-reset")
def request_reset(body: EmailOnly, request: Request):
    db = _get_db(request)
    _ensure_schema(db)
    row = db.execute("SELECT id, email FROM users WHERE email = ?", (body.email,)).fetchone()
    if not row:
        return {"ok": True}
    token = _insert_token(db, user_id=row[0], kind="reset", minutes=60)
    link = f"{FRONTEND_BASE_URL}/#/reset?token={token}"
    _send_email(
        to=row[1],
        subject="Reset your password",
        body_html=f"<p>Reset your password:</p><p><a href='{link}'>{link}</a></p>",
        body_text=f"Reset your password: {link}",
    )
    return {"ok": True, "link": link} if DEBUG_MAIL_LINKS else {"ok": True}

# You likely already have a helper to hash passwords; re-use it.
def _hash_password(pw: str) -> str:
    import hashlib, os
    salt = os.urandom(16)
    return salt.hex() + ":" + hashlib.pbkdf2_hmac("sha256", pw.encode(), salt, 100_000).hex()

@router.post("/reset")
def do_reset(body: ResetBody, request: Request):
    db = _get_db(request)
    _ensure_schema(db)
    t = db.execute("SELECT user_id, expires_at FROM auth_tokens WHERE token=? AND kind='reset'", (body.token,)).fetchone()
    if not t:
        raise HTTPException(status_code=400, detail="Invalid token")
    if datetime.fromisoformat(t[1]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Token expired")

    phash = _hash_password(body.new_password)
    db.execute("UPDATE users SET password_hash=? WHERE id=?", (phash, t[0]))
    db.execute("DELETE FROM auth_tokens WHERE token=?", (body.token,))
    db.commit()
    return {"ok": True}
