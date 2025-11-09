from __future__ import annotations
from typing import List, Optional
from pydantic import BaseModel
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from ..db.core import get_db
from ..db.models import Transcription, User, Role
from ..security import get_current_user, require_role

router = APIRouter(prefix="/tx", tags=["transcriptions"])

# ---------- Schemas ----------
class TxOut(BaseModel):
    id: int
    filename: str
    language: str
    duration_sec: float
    text: str
    summary: str
    model: str
    media_path: str | None = None
    created_at: str
    owner_id: int | None

    class Config:
        from_attributes = True

# ---------- Endpoints ----------
@router.get("/", response_model=List[TxOut])
def list_my_transcriptions(
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    q = db.query(Transcription)
    if user.role != Role.admin:
        q = q.filter(Transcription.owner_id == user.id)
    return q.order_by(Transcription.id.desc()).all()

@router.get("/{tx_id}", response_model=TxOut)
def get_tx(tx_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    tx = db.query(Transcription).get(tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Not found")
    if user.role != Role.admin and tx.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    return tx

@router.delete("/{tx_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_tx(tx_id: int, db: Session = Depends(get_db), user: User = Depends(get_current_user)):
    tx = db.query(Transcription).get(tx_id)
    if not tx:
        raise HTTPException(status_code=404, detail="Not found")
    if user.role != Role.admin and tx.owner_id != user.id:
        raise HTTPException(status_code=403, detail="Forbidden")
    db.delete(tx); db.commit()
