from __future__ import annotations
from pydantic import BaseModel, EmailStr
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ..db.core import Base, engine, SessionLocal
from ..db.models import User, Role
from ..security import hash_password, verify_password, create_access_token, get_db, get_current_user

# Create tables if they don't exist (safe on startup)
Base.metadata.create_all(bind=engine)

router = APIRouter(prefix="/auth", tags=["auth"])

class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    name: str | None = None
    role: Role | None = None  # allow admin bootstrap

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

@router.post("/register", response_model=TokenOut)
def register(payload: RegisterIn, db: Session = Depends(get_db)):
    exists = db.query(User).filter(User.email == payload.email.lower()).first()
    if exists:
        raise HTTPException(status_code=400, detail="Email already registered")
    role = payload.role or Role.researcher
    user = User(
        email=payload.email.lower(),
        name=payload.name,
        role=role,
        password_hash=hash_password(payload.password),
    )
    db.add(user); db.commit(); db.refresh(user)
    token = create_access_token(sub=user.email)
    return TokenOut(access_token=token)

class LoginIn(BaseModel):
    username: EmailStr  # OAuth2 Password spec uses 'username'
    password: str

@router.post("/login", response_model=TokenOut)
def login(form: LoginIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form.username.lower()).first()
    if not user or not verify_password(form.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_access_token(sub=user.email)
    return TokenOut(access_token=token)

class MeOut(BaseModel):
    id: int
    email: EmailStr
    name: str | None
    role: Role

@router.get("/me", response_model=MeOut)
def me(user: User = Depends(get_current_user)):
    return MeOut(id=user.id, email=user.email, name=user.name, role=user.role)
