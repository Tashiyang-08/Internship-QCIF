from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
from typing import Callable, Dict, Any

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

from asr_backend.db.core import SessionLocal
from asr_backend.db.models import User  # we use is_admin flag, not Role

# ----- config -----
SECRET_KEY = os.environ.get("ASR_SECRET_KEY", "CHANGE_ME_DEV_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ASR_TOKEN_MIN", "60"))

# bcrypt can be tricky on Windows; pbkdf2_sha256 is stable.
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ---- DB session dependency ----
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ---- password hashing helpers ----
def hash_password(raw: str) -> str:
    return pwd_context.hash(raw)

def verify_password(raw: str, hashed: str) -> bool:
    return pwd_context.verify(raw, hashed)

# ---- JWT helpers ----
def create_access_token(
    subject: str | Dict[str, Any],
    *,
    expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES,
) -> str:
    """
    Accepts either a string subject (stored in 'sub') OR a dict payload.
    """
    now = datetime.now(timezone.utc)

    if isinstance(subject, dict):
        payload = subject.copy()
    else:
        payload = {"sub": str(subject)}

    payload.setdefault("iat", int(now.timestamp()))
    payload.setdefault("exp", int((now + timedelta(minutes=expires_minutes)).timestamp()))

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token: str) -> dict:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return data or {}
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# ---- current user dependency ----
def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    data = _decode_token(token)
    sub = (data.get("sub") or "").strip()

    if not sub:
        raise HTTPException(status_code=401, detail="Invalid token (no sub)")

    if sub.isdigit():
        user = db.query(User).filter(User.id == int(sub)).first()
    else:
        user = db.query(User).filter(User.email == sub).first()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

# ---- role/permission helpers ----
def require_role(*roles: str) -> Callable[[User], User]:
    """
    Minimal role check compatible with 'is_admin'.
    Usage: dependencies=[Depends(require_role("admin"))]
    """
    def _dep(user: User = Depends(get_current_user)) -> User:
        if not roles:
            return user
        if "admin" in roles and not getattr(user, "is_admin", False):
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _dep

def require_admin(user: User = Depends(get_current_user)) -> User:
    if not getattr(user, "is_admin", False):
        raise HTTPException(status_code=403, detail="Admin only")
    return user
