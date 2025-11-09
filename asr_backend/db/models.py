# asr_backend/db/models.py
from __future__ import annotations
from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Text, Float, DateTime, Boolean, ForeignKey
)
from sqlalchemy.orm import relationship

from .core import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)

    email = Column(String(255), unique=True, index=True, nullable=False)
    full_name = Column(String(255))
    password_hash = Column(String(255), nullable=False)
    is_admin = Column(Boolean, default=False, nullable=False)

    # Email verification
    email_verified = Column(Boolean, default=False, nullable=False)
    verify_code = Column(String(12), nullable=True)
    verify_sent_at = Column(DateTime, nullable=True)
    verify_attempts = Column(Integer, default=0, nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    # Relationship
    transcriptions = relationship("Transcription", back_populates="owner")

class Transcription(Base):
    __tablename__ = "transcriptions"
    id = Column(Integer, primary_key=True, index=True)

    filename = Column(String(512), nullable=False)
    language = Column(String(32), default="en", nullable=False)
    duration_sec = Column(Float, default=0.0, nullable=False)
    text = Column(Text, default="", nullable=False)
    summary = Column(Text, default="", nullable=False)
    model = Column(String(128), default="", nullable=False)

    # <-- this is the column your DB is missing
    media_path = Column(String(1024), nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)

    owner_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    owner = relationship("User", back_populates="transcriptions")
