# asr_backend/db/crud.py
from __future__ import annotations
from typing import List, Optional

from sqlalchemy.orm import Session

from .core import SessionLocal
from .models import Transcription

def insert_transcription(
    *,
    filename: str,
    language: str,
    duration_sec: float,
    text: str,
    summary: str,
    model: str,
    media_path: Optional[str] = None,
    owner_id: Optional[int] = None,
) -> Transcription:
    db: Session = SessionLocal()
    try:
        rec = Transcription(
            filename=filename,
            language=language,
            duration_sec=duration_sec,
            text=text,
            summary=summary,
            model=model,
            media_path=media_path,
            owner_id=owner_id,
        )
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec
    finally:
        db.close()

def list_transcriptions(offset: int = 0, limit: int = 50) -> List[Transcription]:
    db: Session = SessionLocal()
    try:
        return (
            db.query(Transcription)
              .order_by(Transcription.created_at.desc())
              .offset(max(0, offset))
              .limit(limit)
              .all()
        )
    finally:
        db.close()

def get_transcription(rec_id: int) -> Optional[dict]:
    db: Session = SessionLocal()
    try:
        rec = db.query(Transcription).filter(Transcription.id == rec_id).first()
        return None if not rec else {
            "id": rec.id,
            "filename": rec.filename,
            "language": rec.language,
            "duration_sec": rec.duration_sec,
            "text": rec.text,
            "summary": rec.summary,
            "model": rec.model,
            "media_path": rec.media_path,
            "created_at": rec.created_at.isoformat(),
            "owner_id": rec.owner_id,
        }
    finally:
        db.close()

def delete_transcription(rec_id: int) -> bool:
    db: Session = SessionLocal()
    try:
        rec = db.query(Transcription).filter(Transcription.id == rec_id).first()
        if not rec:
            return False
        db.delete(rec)
        db.commit()
        return True
    finally:
        db.close()
