from __future__ import annotations  # MUST be the first line

# ---------------- Stdlib imports ----------------
import io
import re
import math
import os
import shutil
import tempfile
import logging
import time
import uuid
import subprocess
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Tuple

# ---------------- FastAPI / Starlette ----------------
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.encoders import jsonable_encoder
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles  # for downloadable files

# ---------------- Logging ----------------
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
for noisy in ["numba", "numba.core", "llvmlite", "asyncio", "urllib3", "httpx", "matplotlib", "PIL"]:
    logging.getLogger(noisy).setLevel(logging.WARNING)

# ---------------- Env + PATH ----------------
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
os.environ.setdefault("FASTER_WHISPER_DEVICE", os.getenv("FASTER_WHISPER_DEVICE", "cpu"))
os.environ.setdefault("FASTER_WHISPER_COMPUTE_TYPE", os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8"))
FW_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu").lower()
FW_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8").lower()

# App-level tuning knobs (readable from .env)
ASR_DEFAULT_MODEL = os.getenv("ASR_DEFAULT_MODEL", "faster-whisper-tiny")
ASR_SUMMARY_MODE = os.getenv("ASR_SUMMARY", "quick").lower()      # "off" | "quick" | "hf"
ASR_SUMMARY_SHORT_MAX_SEC = int(os.getenv("ASR_SUMMARY_SHORT_MAX_SEC", "35"))
ASR_WARMUP = os.getenv("ASR_WARMUP", "1") == "1"                  # preload a tiny model at startup

# Language-guard envs
LANGUAGE_GUARD_ENABLE = os.getenv("LANGUAGE_GUARD_ENABLE", "1") == "1"
LANGUAGE_GUARD_MIN_PROB = float(os.getenv("LANGUAGE_GUARD_MIN_PROB", "0.70"))

# Output dir for downloadable files
OUTPUT_DIR = os.path.abspath(os.getenv("OUTPUT_DIR", "./outputs"))
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional custom display names for speakers (UI can override too)
SPEAKER_RENAME = {
    # "SPEAKER_00": "Interviewer",
    # "SPEAKER_01": "Candidate",
}

# Add ffmpeg (adjust path for your machine)
_FF_BIN = r"C:\Users\tashi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin"
if os.path.isdir(_FF_BIN) and _FF_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = os.environ.get("PATH", "") + ";" + _FF_BIN

# ---------------- JSON sanitizer for NumPy & friends ----------------
import numpy as _np
def _to_py(obj):
    if isinstance(obj, (_np.bool_,)): return bool(obj)
    if isinstance(obj, (_np.integer,)): return int(obj)
    if isinstance(obj, (_np.floating,)): return float(obj)
    if isinstance(obj, _np.ndarray): return obj.tolist()
    if isinstance(obj, dict): return {k: _to_py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)): return [_to_py(x) for x in obj]
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

# ---------------- FastAPI app ----------------
app = FastAPI(title="ASR Backend", version="2.9-sticky-auto-remap+reset")

# CORS (use FRONTEND_URL from .env; fallback to localhost:5173)
CORS_ORIGINS = [os.getenv("FRONTEND_URL", "http://localhost:5173")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve downloadable files
app.mount("/files", StaticFiles(directory=OUTPUT_DIR), name="files")

# Log requests (simple)
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

# Mount AUTH router
from asr_backend.routers.auth import router as auth_router
app.include_router(auth_router)

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
        import numpy as np, soundfile as sf, tempfile as _tf, os as _os
        sr = 16000
        tmpdir = _tf.mkdtemp(prefix="asr_warm_")
        p = _os.path.join(tmpdir, "silence.wav")
        sf.write(p, np.zeros(sr // 2, dtype="float32"), sr)
        _ = asr_engine.transcribe(p, language="en")
        shutil.rmtree(tmpdir, ignore_errors=True)
        logging.info("ASR warm-up complete for %s (%s/%s)", model_name, FW_DEVICE, FW_COMPUTE)
    except Exception as e:
        logging.warning("ASR warm-up skipped: %s", e)

# ---------------- Helpers for summary ----------------
def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")

_STOP = set("""
a an the and or of to in on for with as at by from into over after before up down out about than then
is are was were be been being do does did doing have has had having can could should would may might must
i you he she it we they me him her us them my your his her its our their this that these those here there
""".split())

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

# ---------------- Speaker diarization helpers ----------------
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
    if not segments:
        return segments
    basic = assign_speakers_basic(segments, diar_spans, min_overlap=min_overlap, use_midpoint_fallback=use_midpoint_fallback)
    enriched = []
    for i, s in enumerate(basic):
        start = float(s.get("start", 0.0)); end = float(s.get("end", start))
        enriched.append((i, start, end, s.get("speaker", "SPEAKER_00")))
    enriched.sort(key=lambda x: x[1])
    for k in range(1, len(enriched)-1):
        i_prev, s_prev, e_prev, sp_prev = enriched[k-1]
        i_mid, s_mid, e_mid, sp_mid = enriched[k]
        i_next, s_next, e_next, sp_next = enriched[k+1]
        dur_mid = e_mid - s_mid
        if sp_prev == sp_next and sp_mid != sp_prev and dur_mid <= stickiness_margin:
            enriched[k] = (i_mid, s_mid, e_mid, sp_prev)
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
            return speaker_rename(lbl)
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

# ---------------- Optional diarization orchestrator ----------------
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
    win = int(0.5 * sr)
    hop = int(0.25 * sr)
    frames = []
    times = []
    for i in range(0, max(0, len(y) - win), hop):
        seg = y[i:i+win]
        if np.sqrt((seg**2).mean()) < 0.007:
            continue
        mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13)
        feat = np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])
        frames.append(feat)
        times.append((i / sr, (i + win) / sr))
    if not frames:
        return []
    X = np.vstack(frames).astype("float32")
    if max_speakers and max_speakers > 0:
        candidate_k = int(max_speakers)
    else:
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
    merged: List[Tuple[float, float, int]] = []
    for st, en, sp in spans:
        if merged and merged[-1][2] == sp and (st - merged[-1][1]) <= 0.5:
            merged[-1] = (merged[-1][0], en, sp)
        else:
            merged.append((st, en, sp))
    return [{"start": float(s), "end": float(e), "speaker": f"SPEAKER_{int(sp):02d}"} for (s, e, sp) in merged]

def _run_any_diarization(wav_path: str, max_speakers: Optional[int], prefer: str = "auto") -> List[Dict[str, Any]]:
    import inspect
    _diarize_basic_funcs = None
    try:
        from asr_backend.asr.diarize_basic import diarize_speakers as _basic_diar, label_segments_with_speakers as _basic_label
        logging.info("diarize_basic imported from: %s", inspect.getfile(_basic_diar))
        _diarize_basic_funcs = (_basic_diar, _basic_label)
        logging.info("diarize_basic available and ready.")
    except Exception as e:
        logging.warning("diarize_basic not available: %s", e)
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

# ---------------- helpers for SRT ----------------
def _tc(t: float) -> str:
    ms = int(round((float(t) % 1) * 1000)); s = int(float(t)) % 60
    m = (int(float(t)) // 60) % 60; h = int(float(t)) // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"

def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
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
        buf.write(f"{i}\n{_tc(start)} --> {_tc(end)}\n{line}\n\n")
    return buf.getvalue()

# ---------------- Transcribe (audio+video, auto model/lang, diarize) ----------------
@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    language: Optional[str] = Form("auto"),
    model: Optional[str] = Form("auto"),
    summarize: Optional[bool] = Form(False),
    diarize: Optional[bool] = Form(False),
    num_speakers: Optional[int] = Form(None),
    diarizer: Optional[str] = Form("auto"),  # "auto"|"basic"|"fallback"
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

        # -------- Stronger Language Guard --------
        detected_lang = (out.get("language") or "").lower()
        lang_prob = float(out.get("language_prob") or out.get("language_probability") or 0.0)
        text_initial = (out.get("text") or "").strip()

        def _looks_english(s: str) -> bool:
            if not s:
                return False
            letters = re.findall(r"[A-Za-z]", s)
            if len(letters) < 40:
                return False
            common = 0
            for w in (" the ", " and ", " to ", " of ", " in ", " that ", " is ", " it ", " for ", " on "):
                if w in f" {s.lower()} ":
                    common += 1
            return common >= 2 or (len(letters) >= max(40, int(0.6 * len(s))))

        _known_false_pos = {"cy", "ga", "gd", "kw"}

        def _should_force_en(det_lang: str, prob: float, s: str) -> bool:
            if lang_arg is not None:
                return False  # user explicitly chose a language
            if det_lang in {"en", "eng"}:
                return False
            min_prob = max(0.70, LANGUAGE_GUARD_MIN_PROB)
            return (det_lang in _known_false_pos or prob >= min_prob) and _looks_english(s)

        if LANGUAGE_GUARD_ENABLE and _should_force_en(detected_lang, lang_prob, text_initial):
            logging.info("[lang-guard] Forcing English (detected=%s prob=%.2f)", detected_lang, lang_prob)
            if (model or "auto").lower() == "auto":
                if _HAS_CHOOSER:
                    out = _auto_transcribe(wav_path, language="en")
                    chosen_model = out.get("model") or "auto"
                else:
                    asr_engine = get_model_with_hints(ASR_DEFAULT_MODEL or "faster-whisper-tiny",
                                                      device=FW_DEVICE, compute_type=FW_COMPUTE)
                    out = asr_engine.transcribe(wav_path, language="en")
                    chosen_model = out.get("model") or (ASR_DEFAULT_MODEL or "faster-whisper-tiny")
            else:
                asr_engine = get_model_with_hints(model, device=FW_DEVICE, compute_type=FW_COMPUTE)
                out = asr_engine.transcribe(wav_path, language="en")

            # Refresh locals after re-run
            out["text"] = _normalize_case(out.get("text") or "")
            for seg in out.get("segments") or []:
                seg["text"] = _normalize_case(seg.get("text") or "")
            detected_lang = (out.get("language") or "").lower()
            lang_prob = float(out.get("language_prob") or out.get("language_probability") or 0.0)
            logging.info("[lang-guard] Re-run in English complete (now language=%s prob=%.2f)",
                         detected_lang, lang_prob)

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
            auto_num = None if (num_speakers in (None, 0, 2)) else int(num_speakers)
            spans = _run_any_diarization(wav_path, max_speakers=auto_num, prefer=diarizer)
            spans = _canonicalize_diar_spans(spans)
            logging.info("[diarization] requested=%s num_speakers_in=%s -> used=%s got_spans=%d",
                         diarizer, num_speakers, auto_num, len(spans))
            diar_spans = spans
            if spans and segs_tagged:
                segs_tagged = assign_speakers_to_segments_sticky(
                    segs_tagged, spans,
                    min_overlap=0.05,
                    stickiness_margin=0.12,
                    use_midpoint_fallback=True
                )
                segs_tagged, label_map = _remap_segments_by_first_appearance(segs_tagged)
                if spans:
                    diar_spans = [
                        {"start": float(sp["start"]), "end": float(sp["end"]),
                         "speaker": label_map.get(sp["speaker"], sp["speaker"])}
                        for sp in spans
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

        # --- Persist to DB (or memory stubs) ---
        rec = insert_transcription(
            filename=file.filename,
            language=out.get("language") or (language if language != "auto" else "auto"),
            duration_sec=duration_sec,
            text=speaker_text,
            summary=summary,
            model=chosen_model,
        )

        created = getattr(rec, "created_at", None)
        if isinstance(created, datetime):
            created = created.replace(tzinfo=timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        elif not created:
            created = _now_utc_iso()

        # --- Write downloadable artifacts ---
        base = os.path.splitext(file.filename)[0]
        stamp = time.strftime("%Y%m%d-%H%M%S")
        tag = f"{base}-{stamp}-{req_id[:8]}"

        # audio
        out_wav_name = f"{tag}.wav"
        out_wav_path = os.path.join(OUTPUT_DIR, out_wav_name)
        try:
            shutil.copyfile(wav_path, out_wav_path)
        except Exception as e:
            logging.warning("Failed to store WAV: %s", e)
            out_wav_name = None

        # srt
        segments_for_srt = segs_tagged if (segs_tagged and all(("start" in s and "end" in s) for s in segs_tagged)) \
                           else [{"start": 0.0, "end": duration_sec or 0.0, "text": text_plain}]
        srt_text = _segments_to_srt(segments_for_srt)
        out_srt_name = f"{tag}.srt"
        out_srt_path = os.path.join(OUTPUT_DIR, out_srt_name)
        try:
            with open(out_srt_path, "w", encoding="utf-8") as f:
                f.write(srt_text)
        except Exception as e:
            logging.warning("Failed to store SRT: %s", e)
            out_srt_name = None

        # txt (speaker-tagged or plain)
        out_txt_name = f"{tag}.txt"
        out_txt_path = os.path.join(OUTPUT_DIR, out_txt_name)
        try:
            with open(out_txt_path, "w", encoding="utf-8") as f:
                f.write(speaker_text or text_plain or "")
        except Exception as e:
            logging.warning("Failed to store TXT: %s", e)
            out_txt_name = None

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
            "text": speaker_text,
            "text_plain": text_plain,
            "summary": summary,
            "bullets": bullets,
            "summary_meta": summary_meta,
            "segments": segs_tagged,
            "speakers": diar_spans,
            "model": chosen_model,
            "latency_sec": round(time.time() - t0, 3),
            "created_at": created,
            "request_id": req_id,
            "speaker_map": SPEAKER_RENAME or {},
            "debug": debug,
            # download URLs exposed here:
            "audio_url": f"/files/{out_wav_name}" if out_wav_name else None,
            "srt_url": f"/files/{out_srt_name}" if out_srt_name else None,
            "txt_url": f"/files/{out_txt_name}" if out_txt_name else None,
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

# ---------------- SRT export (direct download) ----------------
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

        srt_text = _segments_to_srt(segments)
        srt_bytes = io.BytesIO(srt_text.encode("utf-8"))
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
