# asr_backend/asr/summarizer.py
"""
Robust local summarizer with token-aware chunking to avoid 1024-token limit.
"""

from __future__ import annotations
import re, difflib, threading
from typing import Dict, List, Tuple, Literal

from transformers import pipeline, AutoTokenizer

# Choose model (BART gives nicer prose; DistilBART is faster)
_MODEL_NAME = "facebook/bart-large-cnn"   # 1024 token max

# Token limits for safety (keep under model max)
_MAX_TOK = 1024
_IN_TOK = 900       # window size
_OVER_TOK = 150     # overlap between windows

_PIPE = None
_TOKENIZER = None
_LOCK = threading.Lock()

def _get_pipe():
    global _PIPE, _TOKENIZER
    if _PIPE is None or _TOKENIZER is None:
        with _LOCK:
            if _PIPE is None or _TOKENIZER is None:
                _PIPE = pipeline("summarization", model=_MODEL_NAME)
                _TOKENIZER = AutoTokenizer.from_pretrained(_MODEL_NAME, use_fast=True)
    return _PIPE, _TOKENIZER

def _clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def _similarity(a: str, b: str) -> float:
    a = _clean(a).lower(); b = _clean(b).lower()
    if not a or not b: return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()

def _ensure_period(s: str) -> str:
    return s if s.endswith((".", "!", "?")) else s + "."

def _tok_ids(text: str):
    _, tok = _get_pipe()
    return tok.encode(text, add_special_tokens=False)

def _decode(ids: List[int]) -> str:
    _, tok = _get_pipe()
    return tok.decode(ids, skip_special_tokens=True)

def _truncate_by_tokens(text: str, max_tokens: int) -> str:
    ids = _tok_ids(text)
    if len(ids) <= max_tokens:
        return text
    return _decode(ids[:max_tokens])

def _split_into_token_windows(text: str, win_tokens: int, overlap_tokens: int) -> List[str]:
    ids = _tok_ids(text)
    if len(ids) <= win_tokens:
        return [text]
    out = []
    i = 0
    n = len(ids)
    while i < n:
        j = min(n, i + win_tokens)
        out.append(_decode(ids[i:j]))
        if j == n:
            break
        i = max(j - overlap_tokens, i + 1)
    return out

def _auto_lengths(target_words: int) -> tuple[int, int]:
    mx = max(56, int(target_words * 1.2))
    mn = max(30, int(mx * 0.6))
    if mn >= mx: mn = max(24, mx - 10)
    return mx, mn

def _summarize_once(text: str, target_words: int) -> str:
    pipe, _ = _get_pipe()
    # hard-truncate any single call so BART never exceeds 1024
    text = _truncate_by_tokens(_clean(text), 950)
    mx, mn = _auto_lengths(target_words)
    out = pipe(text, max_length=mx, min_length=mn, do_sample=False)[0]["summary_text"]
    return _clean(out)

def _bullets_from_text(summary: str, k: int = 5) -> List[str]:
    if not summary: return []
    parts = [p.strip() for p in re.split(r"[.!?]\s+", summary) if len(p.strip().split()) >= 4]
    out: List[str] = []
    for p in parts:
        s = _ensure_period(p)
        if s not in out: out.append(s)
        if len(out) >= k: break
    return out

# ------------------- Public API -------------------

def summarize_text(
    text: str,
    *,
    target_words: int = 120,
) -> Tuple[str, List[str], Dict[str, object]]:
    """
    Token-aware, chunked summarization that never exceeds model limits.
    Returns (summary, bullets, meta) with meta['genre']="generic" for compatibility.
    """
    raw = _clean(text)
    if not raw:
        return "", [], {"genre": "generic", "action_items": []}

    meta = {"genre": "generic", "action_items": []}

    # If short enough: single pass
    ids = _tok_ids(raw)
    if len(ids) <= _IN_TOK:
        summary = _summarize_once(raw, target_words)
    else:
        # Chunk into overlapping token windows → summarize each → combine → summarize again
        windows = _split_into_token_windows(raw, _IN_TOK, _OVER_TOK)
        per = max(60, target_words // max(1, len(windows)))
        partials = [_summarize_once(w, per) for w in windows]
        combined = " ".join(partials)
        summary = _summarize_once(combined, target_words)

    # Anti-echo: if too similar to input, do a tighter pass on a truncated core
    if _similarity(summary, raw) > 0.70:
        core = _truncate_by_tokens(raw, 750)
        summary = _summarize_once(core, max(80, target_words))

    bullets = _bullets_from_text(summary, k=5)
    return summary, bullets, meta
