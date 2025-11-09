# asr_backend/asr/chooser.py
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import wave
import contextlib
from typing import Optional

from asr_backend.asr.registry import available_models, get_model

log = logging.getLogger(__name__)

# -------------------------
# Optional deps (psutil for RAM probe)
# -------------------------
def _ram_gb() -> float:
    try:
        import psutil  # type: ignore
        return psutil.virtual_memory().total / (1024 ** 3)
    except Exception:
        return 0.0


# -------------------------
# Duration probe (ffprobe -> wave header)
# -------------------------
def _ffprobe_duration(path: str) -> float:
    """Return duration (seconds). Falls back to wave header if ffprobe is missing."""
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        try:
            out = subprocess.check_output(
                [ffprobe, "-v", "error", "-show_entries", "format=duration", "-of", "json", path],
                stderr=subprocess.STDOUT,
            )
            data = json.loads(out.decode("utf-8", errors="ignore"))
            dur = float(data.get("format", {}).get("duration", 0.0) or 0.0)
            if dur > 0:
                return dur
        except Exception:
            pass
    try:
        with contextlib.closing(wave.open(path, "rb")) as wf:
            frames = wf.getnframes()
            rate = max(1, wf.getframerate() or 16000)
            return frames / float(rate)
    except Exception:
        return 0.0


# -------------------------
# GPU probe
# -------------------------
def _has_gpu() -> bool:
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


# -------------------------
# Config knobs (env-tunable)
# -------------------------
SHORT_SEC = float(os.getenv("CHOOSER_SHORT_SEC", "45"))     # short clip threshold
MID_SEC = float(os.getenv("CHOOSER_MID_SEC", str(8 * 60)))  # mid clip threshold
ALLOW_LARGE_ON_CPU = os.getenv("CHOOSER_ALLOW_LARGE_ON_CPU", "0") == "1"
RAM_LARGE_MIN_GB = float(os.getenv("CHOOSER_RAM_LARGE_MIN_GB", "12"))
RAM_MEDIUM_MIN_GB = float(os.getenv("CHOOSER_RAM_MEDIUM_MIN_GB", "6"))


# -------------------------
# Core decision
# -------------------------
def choose_model_for_file(path: str, *, language_hint: Optional[str] = None) -> str:
    """
    Auto-select the best model based on GPU, RAM, clip duration, and language.
    Never picks large on CPU unless CHOOSER_ALLOW_LARGE_ON_CPU=1.
    """
    avail = set(available_models())
    if not avail:
        raise RuntimeError("No ASR models available.")

    dur = _ffprobe_duration(path)
    lang = (language_hint or "").strip().lower()
    gpu = _has_gpu()
    ram_gb = _ram_gb()

    log.info(
        "[chooser] GPU=%s RAM≈%.1fGB dur=%.1fs lang=%s allow_large_cpu=%s",
        gpu, ram_gb, dur, lang or "auto", ALLOW_LARGE_ON_CPU,
    )

    # Preferred order candidates (we'll filter by availability later)
    order: list[str] = []

    if gpu:
        # Best accuracy first when GPU exists
        order = [
            "faster-whisper-large-v3",
            "faster-whisper-large-v2",
            "faster-whisper-medium",
            "faster-whisper-small",
            "wav2vec2-base-en",
            "vosk-small-en",
            "faster-whisper-tiny",
        ]
    else:
        # CPU path
        short = dur <= SHORT_SEC
        mid = dur <= MID_SEC
        is_en = lang in ("en", "eng", "english") or not lang or lang == "auto"

        # If multilingual likely, favor Whisper family
        if not is_en:
            if ram_gb >= RAM_MEDIUM_MIN_GB:
                order = [
                    # optionally allow large on CPU if explicitly enabled and RAM is high enough
                    *(
                        ["faster-whisper-large-v3", "faster-whisper-large-v2"]
                        if ALLOW_LARGE_ON_CPU and ram_gb >= RAM_LARGE_MIN_GB
                        else []
                    ),
                    "faster-whisper-medium",
                    "faster-whisper-small",
                    "faster-whisper-tiny",
                    "wav2vec2-base-en",   # last-resort English-capable fallback
                    "vosk-small-en",
                ]
            else:
                order = [
                    "faster-whisper-small",
                    "faster-whisper-tiny",
                    "wav2vec2-base-en",
                    "vosk-small-en",
                ]
        else:
            # English path: use wav2vec2/small for very short clips to be snappy
            if short:
                order = [
                    "wav2vec2-base-en",
                    "faster-whisper-small",
                    "vosk-small-en",
                    "faster-whisper-medium",
                ]
            elif mid:
                # Medium for balance if RAM allows, else small/w2v2
                if ram_gb >= RAM_MEDIUM_MIN_GB:
                    order = ["faster-whisper-medium", "wav2vec2-base-en", "faster-whisper-small"]
                else:
                    order = ["faster-whisper-small", "wav2vec2-base-en", "vosk-small-en"]
            else:
                # Long English clip on CPU: stick to medium; optionally allow large if enabled & RAM high
                order = [
                    *(
                        ["faster-whisper-large-v3", "faster-whisper-large-v2"]
                        if (ALLOW_LARGE_ON_CPU and ram_gb >= RAM_LARGE_MIN_GB)
                        else []
                    ),
                    "faster-whisper-medium",
                    "wav2vec2-base-en",
                    "faster-whisper-small",
                ]

    # Pick the first available model from the priority list
    for m in order:
        if m in avail:
            log.info("[chooser] Selected model=%s", m)
            return m

    # Last-resort fallback (whatever exists)
    chosen = next(iter(avail))
    log.warning("[chooser] Fallback to %s (no priority candidates available)", chosen)
    return chosen


# -------------------------
# Public API
# -------------------------
def transcribe_with_auto_choice(path: str, *, language: Optional[str] = None):
    """
    Choose the model automatically and run transcription.
    Adds 'auto_selected' to the result for transparency.
    """
    model = choose_model_for_file(path, language_hint=language)
    engine = get_model(model)
    out = engine.transcribe(path, language=language)
    out.setdefault("model", model)
    out["auto_selected"] = model
    return out
