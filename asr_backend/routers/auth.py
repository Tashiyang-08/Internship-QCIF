# asr_backend/routers/auth.py
from __future__ import annotations

import os
import json
import hmac
import hashlib
import secrets
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status, Form
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr

from sqlalchemy import (
    Column, Integer, String, Boolean, DateTime, UniqueConstraint,
    create_engine, select
)
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ---------------- DB (SQLite by default) ----------------
SQLITE_URL = os.getenv("SQLITE_URL", "sqlite:///./asr.db")
engine = create_engine(
    SQLITE_URL,
    connect_args={"check_same_thread": False} if SQLITE_URL.startswith("sqlite:///") else {},
    future=True,
)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    __table_args__ = (UniqueConstraint("email", name="uq_users_email"),)

    id = Column(Integer, primary_key=True)
    email = Column(String(255), nullable=False, unique=True, index=True)
    full_name = Column(String(255), default="")
    password_hash = Column(String(255), nullable=False)
    is_verified = Column(Boolean, default=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

Base.metadata.create_all(bind=engine)

# ---------------- Password hashing ----------------
_DEF_ITERS = 130_000

def _hash_password(password: str, *, iterations: int = _DEF_ITERS) -> str:
    salt = secrets.token_hex(16)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), iterations)
    return f"pbkdf2_sha256${iterations}${salt}${dk.hex()}"

def _verify_password(password: str, stored: str) -> bool:
    try:
        scheme, iters_s, salt_hex, digest_hex = stored.split("$", 3)
        if scheme != "pbkdf2_sha256":
            return False
        iterations = int(iters_s)
        dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt_hex), iterations)
        return hmac.compare_digest(dk.hex(), digest_hex)
    except Exception:
        return False

# ---------------- Dev token (simple bearer) ----------------
def _make_token(email: str) -> str:
    return f"dev-{secrets.token_hex(8)}:{email}"

def _parse_token(token: str) -> str:
    if token and token.startswith("dev-") and ":" in token:
        return token.split(":", 1)[1]
    raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# ---------------- Schemas ----------------
class LoginBody(BaseModel):
    email: EmailStr
    password: str

class RegisterBody(BaseModel):
    email: EmailStr
    password: str
    full_name: Optional[str] = None
    is_admin: Optional[bool] = False

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str
    is_verified: bool
    is_admin: bool
    created_at: datetime
    class Config:
        from_attributes = True

# ---------------- Helpers ----------------
def _db() -> Session:
    return SessionLocal()

def _get_user_by_email(db: Session, email: str) -> Optional[User]:
    stmt = select(User).where(User.email == email.lower().strip())
    return db.execute(stmt).scalar_one_or_none()

def _create_user(db: Session, *, email: str, password: str, full_name: str = "", is_admin: bool = False) -> User:
    u = User(
        email=email.lower().strip(),
        full_name=full_name or "",
        password_hash=_hash_password(password),
        is_verified=True,
        is_admin=bool(is_admin),
    )
    db.add(u)
    db.commit()
    db.refresh(u)
    return u

# Optional: import old JSON users once
def _maybe_migrate_json_users():
    path = os.path.abspath("users_db.json")
    if not os.path.exists(path):
        return
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return
        db = _db()
        added = 0
        try:
            for email, u in data.items():
                if not isinstance(u, dict):
                    continue
                if _get_user_by_email(db, email):
                    continue
                pwd = u.get("password") or u.get("password_hash") or "changeme"
                if "$" in str(pwd) and str(pwd).startswith("pbkdf2_sha256$"):
                    ph = str(pwd)
                else:
                    ph = _hash_password(str(pwd))
                newu = User(
                    email=email.lower().strip(),
                    full_name=u.get("full_name") or "",
                    password_hash=ph,
                    is_verified=True,
                    is_admin=bool(u.get("is_admin", False)),
                )
                db.add(newu); added += 1
            db.commit()
        finally:
            db.close()
        if added:
            os.replace(path, path + ".imported.bak")
    except Exception:
        pass

_maybe_migrate_json_users()

# ---------------- Router ----------------
router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# -------- Register (accept JSON or form) --------
@router.post("/register", response_model=UserOut)
async def register(
    request: Request,
    email_form: Optional[str] = Form(None),
    password_form: Optional[str] = Form(None),
    full_name_form: Optional[str] = Form(None),
):
    db = _db()
    try:
        email: Optional[str] = None
        password: Optional[str] = None
        full_name: Optional[str] = None
        # Prefer JSON if present
        if request.headers.get("content-type", "").startswith("application/json"):
            data = await request.json()
            email = (data.get("email") or "").strip().lower()
            password = data.get("password") or ""
            full_name = data.get("full_name") or ""
        else:
            email = (email_form or "").strip().lower()
            password = password_form or ""
            full_name = full_name_form or ""
        if not email or not password:
            raise HTTPException(status_code=400, detail="Email and password required")
        if _get_user_by_email(db, email):
            raise HTTPException(status_code=400, detail="Email already registered")
        u = _create_user(db, email=email, password=password, full_name=full_name)
        return u
    finally:
        db.close()

# -------- Login (accept JSON or form username/password) --------
@router.post("/login")
async def login(
    request: Request,
    username: Optional[str] = Form(None),
    password_form: Optional[str] = Form(None),
):
    db = _db()
    try:
        # Detect payload
        email: Optional[str] = None
        password: Optional[str] = None
        ctype = request.headers.get("content-type", "")
        if ctype.startswith("application/json"):
            try:
                data = await request.json()
                email = (data.get("email") or data.get("username") or "").strip().lower()
                password = data.get("password") or ""
            except Exception:
                pass
        if not email:
            # OAuth2PasswordRequestForm style (username/password fields)
            email = (username or "").strip().lower()
            password = password_form or ""
        if not email or not password:
            raise HTTPException(status_code=401, detail="Invalid email or password")

        u = _get_user_by_email(db, email)
        if not u or not _verify_password(password, u.password_hash):
            raise HTTPException(status_code=401, detail="Invalid email or password")
        return {"access_token": _make_token(u.email), "token_type": "bearer"}
    finally:
        db.close()

@router.get("/me", response_model=UserOut)
def me(token: str = Depends(oauth2_scheme)):
    email = _parse_token(token)
    db = _db()
    try:
        u = _get_user_by_email(db, email)
        if not u:
            raise HTTPException(status_code=401, detail="Invalid token")
        return u
    finally:
        db.close()

# Stubs to keep UI buttons happy
@router.get("/verify")
def verify(): return {"ok": True}

@router.post("/request-password-reset")
def request_password_reset():
    return {"message": "Reset email sent", "link": "http://localhost:5173/#/reset?token=DEV"}

@router.post("/reset-password")
def reset_password():
    return {"message": "Password updated"}
