from __future__ import annotations
from typing import Optional, List, Dict, Any

from sqlalchemy import desc
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

def list_transcriptions(*, offset: int = 0, limit: int = 50) -> List[Dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        q = (
            db.query(Transcription)
            .order_by(desc(Transcription.created_at), desc(Transcription.id))
            .offset(offset)
            .limit(limit)
        )
        rows = q.all()
        return [
            {
                "id": r.id,
                "filename": r.filename,
                "language": r.language,
                "duration_sec": r.duration_sec,
                "text": r.text,
                "summary": r.summary,
                "model": r.model,
                "media_path": r.media_path,
                "created_at": r.created_at,
                "owner_id": r.owner_id,
            }
            for r in rows
        ]
    finally:
        db.close()

def get_transcription(rec_id: int) -> Optional[Dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        r = db.query(Transcription).filter(Transcription.id == rec_id).first()
        if not r:
            return None
        return {
            "id": r.id,
            "filename": r.filename,
            "language": r.language,
            "duration_sec": r.duration_sec,
            "text": r.text,
            "summary": r.summary,
            "model": r.model,
            "media_path": r.media_path,
            "created_at": r.created_at,
            "owner_id": r.owner_id,
        }
    finally:
        db.close()

def delete_transcription(rec_id: int) -> bool:
    db: Session = SessionLocal()
    try:
        r = db.query(Transcription).filter(Transcription.id == rec_id).first()
        if not r:
            return False
        db.delete(r)
        db.commit()
        return True
    finally:
        db.close()
