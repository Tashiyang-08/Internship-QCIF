from __future__ import annotations
from datetime import datetime, timedelta, timezone
import os
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy.orm import Session

# ✅ Use absolute imports to the DB package (sibling of "asr")
from asr_backend.db.core import SessionLocal
from asr_backend.db.models import User, Role  # if you don’t have Role, you can remove it & use is_admin

# ----- config -----
SECRET_KEY = os.environ.get("ASR_SECRET_KEY", "CHANGE_ME_DEV_SECRET")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ASR_TOKEN_MIN", "60"))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

# ----- DB session dependency -----
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ----- password hashing -----
def hash_password(raw: str) -> str:
    return pwd_context.hash(raw)

def verify_password(raw: str, hashed: str) -> bool:
    return pwd_context.verify(raw, hashed)

# ----- jwt -----
def create_access_token(sub: str, *, expires_minutes: int = ACCESS_TOKEN_EXPIRE_MINUTES) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": sub,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=expires_minutes)).timestamp()),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def _decode_token(token: str) -> str:
    try:
        data = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        sub = data.get("sub")
        if not sub:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token payload")
        return str(sub)
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")

# ----- dependencies -----
def get_current_user(token: str = Depends(oauth2_scheme),
                     db: Session = Depends(get_db)) -> User:
    email = _decode_token(token)
    user = db.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def require_role(*roles: Role):
    def _checker(user: User = Depends(get_current_user)) -> User:
        # if you don’t use an Enum Role, swap this with:  if "admin" in roles and not user.is_admin: ...
        if roles and user.role not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return _checker
