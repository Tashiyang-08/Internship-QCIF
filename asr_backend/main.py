# asr_backend/main.py
from __future__ import annotations

# ---------------- Env + PATH ----------------
import os

# Load .env early
try:
    from dotenv import load_dotenv  # pip install python-dotenv
    load_dotenv()
except Exception:
    pass

# Fast CPU defaults (you can override in .env)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_THREADING_LAYER", "SEQUENTIAL")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")

# Device/compute selection for faster-whisper (can be overridden in .env)
os.environ.setdefault("FASTER_WHISPER_DEVICE", os.getenv("FASTER_WHISPER_DEVICE", "cpu"))  # "cuda" or "cpu"
os.environ.setdefault("FASTER_WHISPER_COMPUTE_TYPE", os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8"))  # "int8"/"float16"/"int8_float16"/"int16"
FW_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu").lower()
FW_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8").lower()

# App-level tuning knobs (readable from .env)
ASR_DEFAULT_MODEL = os.getenv("ASR_DEFAULT_MODEL", "faster-whisper-tiny")
ASR_SUMMARY_MODE = os.getenv("ASR_SUMMARY", "quick").lower()      # "off" | "quick" | "hf"
ASR_SUMMARY_SHORT_MAX_SEC = int(os.getenv("ASR_SUMMARY_SHORT_MAX_SEC", "35"))  # skip summary for very short clips
ASR_WARMUP = os.getenv("ASR_WARMUP", "1") == "1"                  # preload a tiny model at startup

# Optional custom display names for speakers (UI can override too)
SPEAKER_RENAME = {
    # "SPEAKER_00": "Interviewer",
    # "SPEAKER_01": "Candidate",
}

# Add ffmpeg (adjust path for your machine)
_FF_BIN = r"C:\Users\tashi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
if os.path.isdir(_FF_BIN) and _FF_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + _FF_BIN

import io
import re
import math
import shutil
import tempfile
import logging
import time
import uuid
import subprocess
from datetime import datetime, timezone

# ---- NEW (email + reset) ----
import ssl, smtplib
from email.message import EmailMessage
from datetime import datetime as _dt, timedelta as _td
import jwt  # pip install PyJWT

from typing import Optional, List, Dict, Any, Tuple

from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Depends, Request, Body, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.encoders import jsonable_encoder
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

# --- JSON sanitizer for NumPy & friends (prevents 500s on jsonable_encoder) ---
import numpy as _np
def _to_py(obj):
    if isinstance(obj, (_np.bool_,)):
        return bool(obj)
    if isinstance(obj, (_np.integer,)):
        return int(obj)
    if isinstance(obj, (_np.floating,)):
        return float(obj)
    if isinstance(obj, _np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_to_py(x) for x in obj]
    return obj

# ---------------- Sentence-case normalizer (safe import) ----------------
try:
    import pysbd
    _SEGMENTER = pysbd.Segmenter(language="en", clean=True)
except Exception:
    _SEGMENTER = None

# ---------------- Optional HF summarizer ----------------
try:
    from asr_backend.asr.summarizer import summarize_text as hf_summarize
    _HAS_HF_SUMMARIZER = True
except Exception as e:
    logging.warning("HF summarizer unavailable: %s", e)
    hf_summarize = None
    _HAS_HF_SUMMARIZER = False

# ---------------- ASR registry + chooser ----------------
try:
    from asr_backend.asr.registry import available_models, get_model as _get_model_raw
except Exception as e:
    logging.exception("Failed to import asr_backend.asr.registry: %s", e)
    def available_models(): return []
    def _get_model_raw(name: str, *args, **kwargs): raise RuntimeError(f"asr registry unavailable: {e}")

def get_model_with_hints(name: str, device: Optional[str] = None, compute_type: Optional[str] = None):
    """Safe wrapper: tries to pass device/compute_type; if registry doesn't accept them, falls back."""
    try:
        return _get_model_raw(name, device=device, compute_type=compute_type)
    except TypeError:
        logging.debug("Registry get_model does not accept device/compute_type; falling back.")
        return _get_model_raw(name)

try:
    from asr_backend.asr.chooser import transcribe_with_auto_choice as _auto_transcribe
    _HAS_CHOOSER = True
except Exception as e:
    logging.warning("Auto chooser not available: %s", e)
    _HAS_CHOOSER = False
    _auto_transcribe = None

# ---------------- DB wiring ----------------
try:
    from asr_backend.db import (
        insert_transcription,
        list_transcriptions as db_list,
        get_transcription as db_get,
        delete_transcription as db_del,
    )
    from asr_backend.db.core import Base, engine
    from asr_backend.db import models  # ensure tables import
    _AUTO_CREATE_TABLES = True
except Exception as e:
    logging.warning("DB unavailable, using in-memory stubs. %s", e)
    _AUTO_CREATE_TABLES = False
    _MEM = []
    class _Rec:
        def __init__(self, **kw): self.__dict__.update(kw)
    def insert_transcription(**kw):
        rec = _Rec(id=len(_MEM)+1, created_at=datetime.now(timezone.utc).isoformat(), **kw)
        _MEM.append(rec); return rec
    def db_list(offset=0, limit=50):
        return [{"id": r.id, "filename": r.filename, "language": r.language,
                 "duration_sec": r.duration_sec, "text": r.text, "summary": r.summary,
                 "model": r.model, "created_at": r.created_at} for r in _MEM[offset:offset+limit]]
    def db_get(rec_id: int):
        for r in _MEM:
            if r.id == rec_id: return r.__dict__
        return None
    def db_del(rec_id: int):
        n = len(_MEM)
        _MEM[:] = [r for r in _MEM if r.id != rec_id]
        return len(_MEM) != n

# ---------------- Auth wiring ----------------
try:
    from asr_backend.security import (
        get_db, get_current_user, require_role,
        hash_password, verify_password, create_access_token,
    )
    from asr_backend.db.core import SessionLocal
    from asr_backend.db.models import User
    _AUTH_ENABLED = True
except Exception as e:
    logging.warning("Auth disabled (security import failed): %s", e)
    _AUTH_ENABLED = False
    SessionLocal = None
    User = None
    def get_current_user(*args, **kwargs):
        raise HTTPException(status_code=503, detail="Auth disabled")

# ---- NEW (email + reset env) ----
SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-change-me")
JWT_ALG = "HS256"
RESET_TOKEN_TTL_MIN = int(os.getenv("RESET_TOKEN_TTL_MIN", "30"))

SMTP_HOST = os.getenv("SMTP_HOST", "")
SMTP_PORT = int(os.getenv("SMTP_PORT", "465"))   # 465 SSL, 587 STARTTLS
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASS = os.getenv("SMTP_PASS", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", SMTP_USER or "no-reply@example.com")
FRONTEND_BASE_URL = os.getenv("FRONTEND_BASE_URL", "http://localhost:5173")


# ---------------- Utilities ----------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

def _normalize_case(text: str) -> str:
    if not text or _SEGMENTER is None:
        return text
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return text
    if sum(c.isupper() for c in letters) / len(letters) >= 0.7:
        t = text.lower().strip()
        sents = [s.strip().capitalize() for s in _SEGMENTER.segment(t)]
        return " ".join(sents)
    return text

_STOP = set("""
a an the and or of to in on for with as at by from into over after before up down out about than then
is are was were be been being do does did doing have has had having can could should would may might must
i you he she it we they me him her us them my your his her its our their this that these those here there
""".split())

def _quick_summary(text: str, target_words: int = 110) -> str:
    clean = re.sub(r"\s+", " ", (text or "")).strip()
    if not clean:
        return ""
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', clean) if s.strip()]
    toks_per, df = [], {}
    for s in sents:
        toks = [t.lower() for t in re.findall(r"[A-Za-z']+", s)]
        toks = [t for t in toks if t not in _STOP and len(t) > 2]
        toks_per.append(toks)
        for w in set(toks):
            df[w] = df.get(w, 0) + 1
    N = len(sents)
    scores = []
    for i, toks in enumerate(toks_per):
        if not toks:
            scores.append((0.0, i)); continue
        tf = {}
        for w in toks:
            tf[w] = tf.get(w, 0) + 1
        L = float(len(toks)) or 1.0
        sc = 0.0
        for w, c in tf.items():
            idf = 1.0 + math.log((1 + N) / (1 + df.get(w, 1)))
            sc += (c / L) * idf
        sc *= 1.0 + 0.1 * (1.0 - i / max(1, N - 1))
        scores.append((sc, i))
    scores.sort(reverse=True)
    chosen, words = [], 0
    for _, i in scores:
        w = len(sents[i].split())
        if words == 0 or words + w <= int(target_words * 1.15):
            chosen.append(i); words += w
        if words >= target_words: break
    chosen.sort()
    return " ".join(sents[i] for i in chosen)

# ---------------- Speaker tagging helpers ----------------
def assign_speakers_basic(
    segments: List[Dict[str, Any]],
    diar_spans: List[Dict[str, Any]],
    *,
    default_label: str = "SPEAKER_00",
    min_overlap: float = 0.05,
    use_midpoint_fallback: bool = True
) -> List[Dict[str, Any]]:
    def overlap(a0, a1, b0, b1):
        return max(0.0, min(a1, b1) - max(a0, b0))

    out = []
    for seg in segments:
        s0 = float(seg.get("start") or 0.0)
        s1 = float(seg.get("end") or s0)
        best_ov = 0.0
        best_lbl = None

        for sp in diar_spans:
            d0 = float(sp.get("start") or 0.0)
            d1 = float(sp.get("end") or d0)
            ov = overlap(s0, s1, d0, d1)
            if ov > best_ov:
                best_ov = ov
                best_lbl = sp.get("speaker")

        if (best_lbl is None or best_ov < min_overlap) and use_midpoint_fallback and diar_spans:
            mid = 0.5 * (s0 + s1)
            for sp in diar_spans:
                d0 = float(sp.get("start") or 0.0)
                d1 = float(sp.get("end") or d0)
                if d0 <= mid <= d1:
                    best_lbl = sp.get("speaker"); break
            if best_lbl is None:
                best_lbl = min(
                    diar_spans,
                    key=lambda sp: min(abs(float(sp.get("start", 0))-s0), abs(float(sp.get("end", 0))-s1))
                ).get("speaker")

        new_seg = dict(seg)
        new_seg["speaker"] = best_lbl or default_label
        out.append(new_seg)
    return out

def assign_speakers_to_segments_sticky(
    segments: List[Dict[str, Any]],
    diar_spans: List[Dict[str, Any]],
    *,
    min_overlap: float = 0.05,
    stickiness_margin: float = 0.12,
    use_midpoint_fallback: bool = True,
) -> List[Dict[str, Any]]:
    """
    1) Assign via overlap/midpoint (basic)
    2) Pass to reduce rapid flip-flop: if a tiny block in between two longer same-speaker blocks, flip it.
    """
    if not segments:
        return segments
    basic = assign_speakers_basic(segments, diar_spans, min_overlap=min_overlap, use_midpoint_fallback=use_midpoint_fallback)
    # sort chronologically (keep original order indexes too)
    enriched = []
    for i, s in enumerate(basic):
        start = float(s.get("start", 0.0)); end = float(s.get("end", start))
        enriched.append((i, start, end, s.get("speaker", "SPEAKER_00")))
    enriched.sort(key=lambda x: x[1])

    # flip small islands
    for k in range(1, len(enriched)-1):
        i_prev, s_prev, e_prev, sp_prev = enriched[k-1]
        i_mid, s_mid, e_mid, sp_mid = enriched[k]
        i_next, s_next, e_next, sp_next = enriched[k+1]

        dur_mid = e_mid - s_mid
        # if surrounded by same speaker and short, flip to that speaker
        if sp_prev == sp_next and sp_mid != sp_prev and dur_mid <= stickiness_margin:
            enriched[k] = (i_mid, s_mid, e_mid, sp_prev)

    # restore original order
    enriched.sort(key=lambda x: x[0])
    out = []
    for i, s in enumerate(basic):
        _, _, _, new_sp = enriched[i]
        ns = dict(s); ns["speaker"] = new_sp
        out.append(ns)
    return out

def build_speaker_tagged_text(
    segments_with_speakers: List[Dict[str, Any]],
    *,
    speaker_rename: Dict[str, str] | None = None
) -> str:
    def name(lbl: str) -> str:
        if speaker_rename and lbl in speaker_rename:
            return speaker_rename[lbl]
        if isinstance(lbl, str) and lbl.startswith("SPEAKER_"):
            n = lbl.split("_")[-1]
            try:
                return f"Speaker {int(n):02d}"
            except Exception:
                return f"Speaker {n}"
        return str(lbl or "Speaker")

    lines: List[Tuple[str, str]] = []
    cur_spk, cur_text = None, []

    for seg in sorted(segments_with_speakers, key=lambda s: float(s.get("start", 0.0))):
        spk = seg.get("speaker") or "SPEAKER_00"
        txt = (seg.get("text") or "").strip()
        if not txt:
            continue
        if cur_spk is None:
            cur_spk, cur_text = spk, [txt]
        elif spk == cur_spk:
            cur_text.append(txt)
        else:
            lines.append((cur_spk, " ".join(cur_text)))
            cur_spk, cur_text = spk, [txt]
    if cur_spk is not None and cur_text:
        lines.append((cur_spk, " ".join(cur_text)))

    return "\n".join(f"[{name(spk)}] {text}" for spk, text in lines)

def _canonicalize_diar_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Sort by start time and merge same-label neighbors with <=0.5s gap."""
    if not spans:
        return spans
    srt = sorted(spans, key=lambda sp: (float(sp.get("start", 0.0)), float(sp.get("end", 0.0))))
    merged: List[Dict[str, Any]] = []
    for sp in srt:
        st = float(sp.get("start", 0.0)); en = float(sp.get("end", st)); lb = sp.get("speaker", "SPEAKER_00")
        if merged and merged[-1]["speaker"] == lb and st - merged[-1]["start"] <= 0.5:
            merged[-1]["end"] = max(merged[-1]["end"], en)
        else:
            merged.append({"start": st, "end": en, "speaker": lb})
    return merged

def _remap_segments_by_first_appearance(
    segments: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], Dict[str, str]]:
    """
    Ensures first *appearing* speaker becomes SPEAKER_00, second new voice SPEAKER_01, etc.
    Returns (remapped_segments, mapping_dict).
    """
    if not segments:
        return segments, {}
    segs_sorted = sorted(segments, key=lambda s: float(s.get("start", 0.0)))
    mapping: Dict[str, str] = {}
    next_idx = 0
    out_sorted = []
    for seg in segs_sorted:
        s = dict(seg)
        lbl = str(s.get("speaker", "SPEAKER_00"))
        if lbl not in mapping:
            mapping[lbl] = f"SPEAKER_{next_idx:02d}"
            next_idx += 1
        s["speaker"] = mapping[lbl]
        out_sorted.append(s)
    return out_sorted, mapping

# ---------------- Optional diarization: basic + fallback ----------------
import inspect
_diarize_basic_funcs = None
try:
    # Make sure asr_backend/asr/diarize_basic.py is your 16 kHz energy-VAD version
    from asr_backend.asr.diarize_basic import (
        diarize_speakers as _basic_diar,
        label_segments_with_speakers as _basic_label,  # not used here
    )
    logging.info("diarize_basic imported from: %s", inspect.getfile(_basic_diar))
    _diarize_basic_funcs = (_basic_diar, _basic_label)
    logging.info("diarize_basic available and ready.")
except Exception as e:
    logging.warning("diarize_basic not available: %s", e)

def _fallback_diarize_mfcc(wav_path: str, max_speakers: Optional[int]) -> List[Dict[str, Any]]:
    import numpy as np
    import soundfile as sf
    import librosa
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.metrics import silhouette_score

    y, sr = sf.read(wav_path)
    if hasattr(y, "ndim") and y.ndim == 2:
        y = y.mean(axis=1)
    y = y.astype("float32")
    if sr <= 0 or len(y) < sr // 2:
        return []

    # 0.5s windows with 0.25s hop
    win = int(0.5 * sr)
    hop = int(0.25 * sr)
    frames = []
    times = []
    for i in range(0, max(0, len(y) - win), hop):
        seg = y[i:i+win]
        # skip silence/very low energy
        if np.sqrt((seg**2).mean()) < 0.007:
            continue
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        frames.append(feat)
        times.append((i / sr, (i + win) / sr))

    if not frames:
        return []

    X = np.vstack(frames).astype("float32")

    # If user supplied a number, honor it; otherwise estimate with silhouette.
    if max_speakers and max_speakers > 0:
        candidate_k = int(max_speakers)
    else:
        # Try k = 2..min(6, n_frames-1); if nothing looks good, fall back to k=1
        best_k, best_score = 1, -1.0
        upper = min(6, max(2, len(X) - 1))
        for k in range(2, upper + 1):
            try:
                labels_k = AgglomerativeClustering(n_clusters=k).fit_predict(X)
                score = silhouette_score(X, labels_k, metric="euclidean")
                if score > best_score:
                    best_score, best_k = score, k
            except Exception:
                continue
        candidate_k = best_k if best_score >= 0.15 else 1

    if candidate_k == 1:
        # Single speaker: one merged span covering the voiced windows
        if not times:
            return []
        start = float(times[0][0])
        end = float(times[-1][1])
        return [{"start": start, "end": end, "speaker": "SPEAKER_00"}]

    labels = AgglomerativeClustering(n_clusters=candidate_k).fit_predict(X)

    spans: List[Tuple[float, float, int]] = []
    for (st, en), sp in zip(times, labels):
        spans.append((st, en, int(sp)))
    spans.sort()

    # Merge adjacent windows with same label (<=0.5s gap)
    merged: List[Tuple[float, float, int]] = []
    for st, en, sp in spans:
        if merged and merged[-1][2] == sp and (st - merged[-1][1]) <= 0.5:
            merged[-1] = (merged[-1][0], en, sp)
        else:
            merged.append((st, en, sp))

    return [{"start": float(s), "end": float(e), "speaker": f"SPEAKER_{int(sp):02d}"} for (s, e, sp) in merged]

def _run_any_diarization(wav_path: str, max_speakers: Optional[int], prefer: str = "auto") -> List[Dict[str, Any]]:
    if prefer not in {"auto", "basic", "fallback"}:
        prefer = "auto"

    if prefer in {"auto", "basic"} and _diarize_basic_funcs:
        try:
            diar, _labeler = _diarize_basic_funcs
            spans_raw = diar(wav_path, max_speakers=max_speakers or None)
            spans = [{"start": float(s), "end": float(e), "speaker": f"SPEAKER_{int(sp):02d}"} for (s, e, sp) in spans_raw]
            logging.info("[diarization] using=basic spans=%d", len(spans))
            return spans
        except Exception as e:
            logging.warning("diarize_basic failed (%s). Falling back.", e)

    try:
        spans = _fallback_diarize_mfcc(wav_path, max_speakers)
        logging.info("[diarization] using=fallback spans=%d", len(spans))
        return spans
    except Exception as e:
        logging.exception("Fallback diarization failed: %s", e)

    logging.warning("[diarization] disabled/no spans")
    return []

# ---------------- FastAPI app ----------------
logging.basicConfig(level=logging.DEBUG)
app = FastAPI(title="ASR Backend", version="2.9-sticky-auto-remap+reset")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

async def _log_requests(request, call_next):
    try:
        print(f"--> {request.method} {request.url.path}")
        resp = await call_next(request)
        print(f"<-- {resp.status_code} {request.url.path}")
        return resp
    except Exception as e:
        print(f"!! 500 {request.url.path}: {e}")
        raise
app.add_middleware(BaseHTTPMiddleware, dispatch=_log_requests)

# Auto-create tables if DB present
try:
    if 'Base' in globals() and 'engine' in globals() and _AUTO_CREATE_TABLES:
        @app.on_event("startup")
        def _ensure_db():
            try:
                Base.metadata.create_all(bind=engine)
            except Exception as e:
                logging.warning("DB auto-create failed: %s", e)
except Exception:
    pass

# ---------- OPTIONAL: warm up a tiny model once so first call is instant ----------
@app.on_event("startup")
def _warm_up_asr():
    if not ASR_WARMUP:
        return
    try:
        model_name = ASR_DEFAULT_MODEL or "faster-whisper-tiny"
        asr_engine = get_model_with_hints(model_name, device=FW_DEVICE, compute_type=FW_COMPUTE)
        import numpy as np, soundfile as sf, tempfile, os as _os
        sr = 16000
        tmpdir = tempfile.mkdtemp(prefix="asr_warm_")
        p = _os.path.join(tmpdir, "silence.wav")
        sf.write(p, np.zeros(sr // 2, dtype="float32"), sr)
        _ = asr_engine.transcribe(p, language="en")
        shutil.rmtree(tmpdir, ignore_errors=True)
        logging.info("ASR warm-up complete for %s (%s/%s)", model_name, FW_DEVICE, FW_COMPUTE)
    except Exception as e:
        logging.warning("ASR warm-up skipped: %s", e)

# ---------------- Schemas ----------------
class RegisterIn(BaseModel):
    email: EmailStr
    password: str
    full_name: str | None = None
    is_admin: bool = False

class TokenOut(BaseModel):
    access_token: str
    token_type: str = "bearer"

class UserOut(BaseModel):
    id: int
    email: EmailStr
    full_name: str | None = None
    is_admin: bool = False

# ---- NEW (email helpers) ----
def _create_reset_token(email: str) -> str:
    payload = {
        "sub": email,
        "scope": "password_reset",
        "exp": _dt.utcnow() + _td(minutes=RESET_TOKEN_TTL_MIN),
    }
    return jwt.encode(payload, SECRET_KEY, algorithm="HS256")

def _decode_reset_token(token: str) -> str:
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=["HS256"])
        if payload.get("scope") != "password_reset":
            raise HTTPException(status_code=400, detail="Invalid reset token")
        return str(payload.get("sub") or "")
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=400, detail="Reset link expired")
    except jwt.InvalidTokenError:
        raise HTTPException(status_code=400, detail="Invalid reset token")

def _send_mail(to: str, subject: str, text: str, html: str | None = None):
    # In dev, if SMTP isn’t configured, just log the link so you can copy it
    if not (SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and EMAIL_FROM):
        print(f"[DEV EMAIL] to={to}\nsubject={subject}\n{text}")
        return
    msg = EmailMessage()
    msg["From"] = EMAIL_FROM
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")
    if SMTP_PORT == 465:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context()) as s:
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)
    else:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
            s.starttls(context=ssl.create_default_context())
            s.login(SMTP_USER, SMTP_PASS)
            s.send_message(msg)

# ---------------- Basic endpoints ----------------
@app.get("/")
def root():
    return {"message": "Hello from ASR backend!", "ok": True}

@app.get("/ping")
def ping():
    return {"message": "pong", "ok": True, "ts": _now_utc_iso()}

@app.get("/models")
def models_endpoint():
    try:
        ms = available_models() or []
        uniq = [m for i, m in enumerate(ms) if m and ms.index(m) == i]
        if "auto" in uniq:
            uniq = ["auto"] + [m for m in uniq if m != "auto"]
        if ASR_DEFAULT_MODEL and ASR_DEFAULT_MODEL in uniq and ASR_DEFAULT_MODEL != "auto":
            uniq = ["auto", ASR_DEFAULT_MODEL] + [m for m in uniq if m not in ("auto", ASR_DEFAULT_MODEL)]
        return {"models": (uniq or ["auto"])}
    except Exception as e:
        import traceback
        return JSONResponse(
            {"models": ["auto"], "error": f"{type(e).__name__}: {e}",
             "trace": traceback.format_exc()}, status_code=200)

# ---------------- Auth endpoints ----------------
if 'SessionLocal' in globals() and 'User' in globals() and _AUTH_ENABLED:
    @app.post("/auth/register", response_model=UserOut)
    def register_user(payload: RegisterIn):
        db = SessionLocal()
        try:
            existing = db.query(User).filter(User.email == payload.email).first()
            if existing:
                raise HTTPException(status_code=400, detail="Email already registered")
            user = User(
                email=payload.email,
                full_name=payload.full_name,
                password_hash=hash_password(payload.password),
                is_admin=bool(payload.is_admin),
            )
            db.add(user); db.commit(); db.refresh(user)
            return UserOut(id=user.id, email=user.email,
                           full_name=user.full_name, is_admin=user.is_admin)
        finally:
            db.close()

    @app.post("/auth/login", response_model=TokenOut)
    async def login(
        request: Request,
        form_data: OAuth2PasswordRequestForm = Depends(None),   # allow JSON or form
        body: dict | None = Body(None)
    ):
        email = None
        password = None

        if form_data is not None and form_data.username:
            email = form_data.username
            password = form_data.password

        if (not email) or (not password):
            try:
                data = body or await request.json()
                email = email or data.get("username") or data.get("email")
                password = password or data.get("password")
            except Exception:
                pass

        if not email or not password:
            raise HTTPException(status_code=422, detail="username/email and password required")

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user or not verify_password(password, user.password_hash):
                # Frontend can show "Reset password" CTA when it sees 401
                raise HTTPException(status_code=401, detail="Invalid credentials")
            token = create_access_token(str(user.email))
            return TokenOut(access_token=token)
        finally:
            db.close()

    @app.get("/auth/me", response_model=UserOut)
    def me(current=Depends(get_current_user)):
        u = current
        return UserOut(id=u.id, email=u.email, full_name=u.full_name, is_admin=u.is_admin)

    # ---- NEW: request password reset ----
    @app.post("/auth/request-password-reset")
    async def request_password_reset(payload: dict = Body(...)):
        email = (payload or {}).get("email")
        if not email:
            raise HTTPException(status_code=422, detail="email required")

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            # Avoid enumeration: always reply the same
            if not user:
                return {"message": "If this email exists, a reset link has been sent."}

            token = _create_reset_token(email)
            link = f"{FRONTEND_BASE_URL}#/reset?token={token}"

            html = f"""
            <div style="font-family:Arial,Helvetica,sans-serif">
              <h2>Password reset</h2>
              <p>Click the button below to reset your password (expires in {RESET_TOKEN_TTL_MIN} minutes):</p>
              <p><a href="{link}" style="background:#4f46e5;color:#fff;padding:10px 16px;border-radius:8px;text-decoration:none">
                  Reset password</a></p>
              <p>Or copy this link:<br><code>{link}</code></p>
              <p style="color:#6b7280">If you didn’t request this, you can ignore this email.</p>
            </div>
            """
            _send_mail(email, "Reset your password", f"Reset link: {link}", html=html)

            # Include link in dev so you can click it without SMTP
            return {"message": "If this email exists, a reset link has been sent.", "link": link}
        finally:
            db.close()

    # ---- NEW: complete password reset ----
    @app.post("/auth/reset-password")
    async def reset_password(payload: dict = Body(...)):
        token = (payload or {}).get("token")
        new_password = (payload or {}).get("new_password")
        if not token or not new_password:
            raise HTTPException(status_code=422, detail="token and new_password required")

        email = _decode_reset_token(token)

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.email == email).first()
            if not user:
                raise HTTPException(status_code=404, detail="User not found")
            user.password_hash = hash_password(new_password)
            db.commit()
            return {"message": "Password updated"}
        finally:
            db.close()

else:
    @app.post("/auth/register")
    def _register_unavailable(): raise HTTPException(status_code=503, detail="Register Unavailable")
    @app.post("/auth/login")
    async def _login_unavailable(
        request: Request,
        username: str | None = Form(None),
        password: str | None = Form(None),
        body: dict | None = Body(None)
    ):
        if not username:
            try:
                data = body or await request.json()
                username = data.get("username") or data.get("email")
            except Exception:
                pass
        if not username:
            raise HTTPException(status_code=422, detail="username/email required")
        return {"access_token": "dev-token", "token_type": "bearer"}
    @app.get("/auth/me")
    def _me_unavailable(): raise HTTPException(status_code=503, detail="Me Unavailable")

# ---------------- Transcribe (audio+video, auto model/lang, diarize) ----------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),     # "auto" or ISO code
    model: Optional[str] = Form("auto"),        # "auto" or explicit key
    summarize: Optional[bool] = Form(False),    # UI can still force it
    diarize: Optional[bool] = Form(False),
    num_speakers: Optional[int] = Form(None),
    diarizer: Optional[str] = Form("auto"),     # "auto"|"basic"|"fallback"
    x_request_id: Optional[str] = Header(default=None),
):
    req_id = x_request_id or str(uuid.uuid4())
    t0 = time.time()
    tmpdir = tempfile.mkdtemp(prefix="asr_")
    src_file = os.path.join(tmpdir, file.filename)
    logging.info("[transcribe] start id=%s file=%s", req_id, file.filename)
    try:
        with open(src_file, "wb") as f:
            shutil.copyfileobj(file.file, f)

        # Standardize to mono 16k WAV (works for both audio/video)
        audio_path = os.path.join(tmpdir, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src_file, "-vn", "-ac", "1", "-ar", "16000", audio_path],
            check=True
        )
        wav_path = audio_path

        # --- Transcribe ---
        lang_arg = None if (language or "auto").lower() == "auto" else language
        chosen_model = None

        if (model or "auto").lower() == "auto":
            if _HAS_CHOOSER:
                out = _auto_transcribe(wav_path, language=lang_arg)
                chosen_model = out.get("model") or "auto"
            else:
                fallback = ASR_DEFAULT_MODEL or "faster-whisper-tiny"
                logging.info("Auto chooser unavailable -> using %s", fallback)
                asr_engine = get_model_with_hints(fallback, device=FW_DEVICE, compute_type=FW_COMPUTE)
                out = asr_engine.transcribe(wav_path, language=lang_arg)
                chosen_model = out.get("model") or fallback
        else:
            asr_engine = get_model_with_hints(model, device=FW_DEVICE, compute_type=FW_COMPUTE)
            out = asr_engine.transcribe(wav_path, language=lang_arg)
            chosen_model = out.get("model") or model

        # Language guard for false 'cy' (Welsh)
        detected_lang = (out.get("language") or "").lower()
        lang_prob = float(out.get("language_prob") or out.get("language_probability") or 0.0)
        text_initial = (out.get("text") or "").strip()

        def _looks_english(s: str) -> bool:
            if not s: return False
            letters = re.findall(r"[A-Za-z]", s)
            return len(letters) >= max(40, int(0.6 * len(s)))

        if lang_arg is None and detected_lang == "cy" and lang_prob >= 0.95 and _looks_english(text_initial):
            logging.info("[lang-guard] Forcing English due to likely false 'cy' detection.")
            if (model or "auto").lower() == "auto":
                if _HAS_CHOOSER:
                    out = _auto_transcribe(wav_path, language="en")
                    chosen_model = out.get("model") or "auto"
                else:
                    asr_engine = get_model_with_hints(ASR_DEFAULT_MODEL or "faster-whisper-tiny", device=FW_DEVICE, compute_type=FW_COMPUTE)
                    out = asr_engine.transcribe(wav_path, language="en")
                    chosen_model = out.get("model") or (ASR_DEFAULT_MODEL or "faster-whisper-tiny")
            else:
                asr_engine = get_model_with_hints(model, device=FW_DEVICE, compute_type=FW_COMPUTE)
                out = asr_engine.transcribe(wav_path, language="en")

        # Normalize case
        out["text"] = _normalize_case(out.get("text") or "")
        for seg in out.get("segments") or []:
            seg["text"] = _normalize_case(seg.get("text") or "")
        text_plain = (out.get("text") or "").strip()
        segs_tagged: List[Dict[str, Any]] = out.get("segments") or []
        duration_sec = float(out.get("duration_sec") or 0.0)

        # Ensure at least one time-bounded segment for diarization
        if (not segs_tagged) and text_plain and duration_sec > 0:
            segs_tagged = [{"start": 0.0, "end": duration_sec, "text": text_plain}]

        # --- Diarization (optional) ---
        diar_spans: List[Dict[str, Any]] = []
        speaker_text: str = text_plain

        if diarize:
            # Treat 2 as "auto" unless user explicitly chooses another positive integer
            auto_num = None if (num_speakers in (None, 0, 2)) else int(num_speakers)
            diar_spans = _run_any_diarization(wav_path, max_speakers=auto_num, prefer=diarizer)
            diar_spans = _canonicalize_diar_spans(diar_spans)
            logging.info("[diarization] requested=%s num_speakers_in=%s -> used=%s got_spans=%d",
                         diarizer, num_speakers, auto_num, len(diar_spans))
            if diar_spans and segs_tagged:
                segs_tagged = assign_speakers_to_segments_sticky(
                    segs_tagged, diar_spans,
                    min_overlap=0.05,
                    stickiness_margin=0.12,
                    use_midpoint_fallback=True
                )
                # Remap so the very first heard voice becomes SPEAKER_00
                segs_tagged, label_map = _remap_segments_by_first_appearance(segs_tagged)
                # Remap spans too (cosmetic consistency)
                if diar_spans:
                    diar_spans = [
                        {"start": float(sp["start"]), "end": float(sp["end"]),
                         "speaker": label_map.get(sp["speaker"], sp["speaker"])}
                        for sp in diar_spans
                    ]
                speaker_text = build_speaker_tagged_text(segs_tagged, speaker_rename=SPEAKER_RENAME or None)
            else:
                logging.warning("[diarization] No spans; transcript will be untagged.")

        # --- Summarize (fast defaults) ---
        summary, bullets, summary_meta = "", [], None
        allow_summary = summarize
        if ASR_SUMMARY_MODE == "off":
            allow_summary = False
        if duration_sec <= ASR_SUMMARY_SHORT_MAX_SEC:
            allow_summary = False

        if allow_summary and text_plain:
            if ASR_SUMMARY_MODE == "hf" and _HAS_HF_SUMMARIZER:
                try:
                    summary, bullets, summary_meta = await run_in_threadpool(
                        hf_summarize, text_plain, target_words=110
                    )
                except Exception as e:
                    logging.exception("HF summarize failed, using quick fallback: %s", e)
                    summary = _quick_summary(text_plain, 110)
            else:
                summary = _quick_summary(text_plain, 110)

        # --- Persist ---
        rec = insert_transcription(
            filename=file.filename,
            language=out.get("language") or (language if language != "auto" else "auto"),
            duration_sec=duration_sec,
            text=speaker_text,        # store speaker-tagged view (for UI)
            summary=summary,
            model=chosen_model,
        )

        created = getattr(rec, "created_at", None)
        if isinstance(created, datetime):
            created = created.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        elif not created:
            created = _now_utc_iso()

        # Optional tiny debug (ensure pure Python scalars)
        debug = {
            "orig_seg_count": int(len((out.get("segments") or []))),
            "final_seg_count": int(len(segs_tagged or [])),
            "has_diar_spans": bool(len(diar_spans) > 0),
        }

        payload = {
            "id": getattr(rec, "id", None),
            "filename": getattr(rec, "filename", None),
            "language": out.get("language") or (language if language != "auto" else "auto"),
            "language_prob": out.get("language_prob") or out.get("language_probability"),
            "duration_sec": duration_sec,
            "text": speaker_text,                 # speaker-tagged if diarized (or plain if not)
            "text_plain": text_plain,             # untagged paragraph
            "summary": summary,                   # summary is untagged
            "bullets": bullets,
            "summary_meta": summary_meta,
            "segments": segs_tagged,              # segments include "speaker" if diarized
            "speakers": diar_spans,               # raw diarization spans (remapped to match segments)
            "model": chosen_model,
            "latency_sec": round(time.time() - t0, 3),
            "created_at": created,
            "request_id": req_id,
            "speaker_map": SPEAKER_RENAME or {},
            "debug": debug,
        }
        logging.info(
            "[transcribe] done id=%s lang=%s dur=%.2fs model=%s latency=%.3fs spans=%d",
            req_id, payload.get("language"), payload.get("duration_sec") or 0.0,
            payload.get("model"), payload["latency_sec"], len(diar_spans or [])
        )
        return JSONResponse(content=jsonable_encoder(_to_py(payload)))

    except HTTPException:
        raise
    except Exception as e:
        logging.exception("Transcribe failed")
        return JSONResponse({"error": f"{type(e).__name__}: {e}", "filename": file.filename, "request_id": req_id}, status_code=500)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------- SRT export ----------------
@app.post("/transcribe/srt")
async def transcribe_srt(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model: Optional[str] = Form("auto"),
    diarize: Optional[bool] = Form(False),
    num_speakers: Optional[int] = Form(None),
    diarizer: Optional[str] = Form("auto"),
):
    tmpdir = tempfile.mkdtemp(prefix="asr_")
    src_file = os.path.join(tmpdir, file.filename)
    try:
        with open(src_file, "wb") as f:
            shutil.copyfileobj(file.file, f)

        audio_path = os.path.join(tmpdir, "audio.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src_file, "-ac", "1", "-ar", "16000", audio_path],
            check=True
        )

        lang_arg = None if (language or "auto").lower() == "auto" else language

        if (model or "auto").lower() == "auto":
            if _HAS_CHOOSER:
                out = _auto_transcribe(audio_path, language=lang_arg)
            else:
                asr_engine = get_model_with_hints(ASR_DEFAULT_MODEL or "faster-whisper-tiny", device=FW_DEVICE, compute_type=FW_COMPUTE)
                out = asr_engine.transcribe(audio_path, language=lang_arg)
        else:
            asr_engine = get_model_with_hints(model, device=FW_DEVICE, compute_type=FW_COMPUTE)
            out = asr_engine.transcribe(audio_path, language=lang_arg)

        segments = out.get("segments") or []
        text_plain = (out.get("text") or "").strip()
        duration = float(out.get("duration_sec") or 0.0)

        # Optional diarization for SRT
        if diarize and segments:
            auto_num = None if (num_speakers in (None, 0, 2)) else int(num_speakers)
            spans = _run_any_diarization(audio_path, max_speakers=auto_num, prefer=diarizer)
            spans = _canonicalize_diar_spans(spans)
            if spans:
                segments = assign_speakers_to_segments_sticky(
                    segments, spans,
                    min_overlap=0.05,
                    stickiness_margin=0.12,
                    use_midpoint_fallback=True
                )
                segments, _ = _remap_segments_by_first_appearance(segments)

        has_timings = segments and all(("start" in s and "end" in s) for s in segments)
        if not has_timings:
            segments = [{"start": 0.0, "end": duration or 0.0, "text": text_plain}]

        def to_tc(t: float) -> str:
            ms = int(round((float(t) % 1) * 1000)); s = int(float(t)) % 60
            m = (int(float(t)) // 60) % 60; h = int(float(t)) // 3600
            return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

        buf = io.StringIO()
        for i, seg in enumerate(sorted(segments, key=lambda s: float(s.get("start", 0.0))), 1):
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or 0.0)
            line = _normalize_case((seg.get("text") or "").strip())
            spk = seg.get("speaker")
            if spk:
                if isinstance(spk, str) and spk.startswith("SPEAKER_"):
                    try:
                        idx = int(spk.split("_")[-1])
                        spk_name = f"Speaker {idx:02d}"
                    except Exception:
                        spk_name = spk
                else:
                    spk_name = str(spk)
                line = f"[{spk_name}] {line}"
            if not line:
                continue
            buf.write(f"{i}\n{to_tc(start)} --> {to_tc(end)}\n{line}\n\n")

        srt_bytes = io.BytesIO(buf.getvalue().encode("utf-8"))
        srt_name = os.path.splitext(file.filename)[0] + ".srt"
        return StreamingResponse(
            srt_bytes, media_type="text/plain; charset=utf-8",
            headers={"Content-Disposition": f'attachment; filename="{srt_name}"'},
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

# ---------------- History ----------------
@app.get("/transcriptions")
def list_transcriptions(offset: int = 0, limit: int = 50):
    return db_list(offset=offset, limit=limit)

@app.get("/transcriptions/{rec_id}")
def get_transcription(rec_id: int):
    rec = db_get(rec_id)
    if not rec:
        raise HTTPException(status_code=404, detail="Not found")
    return rec

@app.delete("/transcriptions/{rec_id}")
def delete_transcription(rec_id: int):
    ok = db_del(rec_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Not found")
    return {"ok": True}

# ---------------- Run ----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("asr_backend.main:app", host="127.0.0.1", port=8000, reload=True, log_level="debug")
