# asr_backend/asr/diarize_basic.py
from __future__ import annotations

print(">>> diarize_basic: energy-VAD 16kHz + centroid labeling loaded")

import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import soundfile as sf
from resemblyzer import VoiceEncoder
from sklearn.cluster import AgglomerativeClustering
import librosa

# ---------------- Tunables (safe defaults) ----------------
_FRAME_MS = 30
_HOP_MS = 15
_SMOOTH_MS = 200

# Slightly stricter energy gate to reduce noise frames
_ENERGY_QUANTILE = 0.65
_MIN_REGION_MS = 350

# Merge gaps to reduce fragmentation
_MERGE_GAP_SEC = 0.5
_MERGE_GAP_LABEL_SEC = 0.5

# Embedding window size
_WINDOW_SEC = 1.2

# Labeling behavior (for overlap-only fallback)
_MIN_OVERLAP_SEC = 0.05
_USE_MIDPOINT_FALLBACK = True


# ---------------- Audio helpers ----------------
def _read_wav_mono16(path: str) -> tuple[np.ndarray, int]:
    """Read audio, convert to mono float32; resample to 16k if needed."""
    y, sr = sf.read(path, dtype="float32", always_2d=False)
    if hasattr(y, "ndim") and y.ndim == 2:
        y = y.mean(axis=1)
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        sr = 16000
    return y.astype(np.float32), sr


def _voiced_regions(
    y: np.ndarray,
    sr: int,
    frame_ms: int = _FRAME_MS,
    hop_ms: int = _HOP_MS,
    smooth_ms: int = _SMOOTH_MS,
    energy_quantile: float = _ENERGY_QUANTILE,
    min_region_ms: int = _MIN_REGION_MS
) -> List[tuple[int, int]]:
    """Energy-based VAD: returns voiced (start_sample, end_sample) regions."""
    frame_len = max(int(sr * frame_ms / 1000), 1)
    hop_len = max(int(sr * hop_ms / 1000), 1)

    frames = librosa.util.frame(y, frame_length=frame_len, hop_length=hop_len, axis=0)
    ste = (frames ** 2).mean(axis=1)

    thr = np.quantile(ste, energy_quantile)
    vad = (ste >= thr).astype(np.float32)

    # Smooth VAD with moving average
    smooth_len = max(1, int((smooth_ms / 1000) * (sr / hop_len)))
    if smooth_len > 1:
        k = np.ones(smooth_len, dtype=np.float32) / smooth_len
        vad = np.convolve(vad, k, mode="same")
        vad = (vad >= 0.5).astype(np.int32)
    else:
        vad = vad.astype(np.int32)

    regions: List[tuple[int, int]] = []
    in_seg, start_f = False, 0
    for i, v in enumerate(vad):
        if v and not in_seg:
            in_seg, start_f = True, i
        elif not v and in_seg:
            in_seg = False
            s = start_f * hop_len
            e = min(len(y), i * hop_len)
            if (e - s) >= int(min_region_ms * sr / 1000):
                regions.append((s, e))
    if in_seg:
        s = start_f * hop_len
        e = len(y)
        if (e - s) >= int(min_region_ms * sr / 1000):
            regions.append((s, e))

    # Merge gaps smaller than MERGE_GAP_SEC
    merged: List[tuple[int, int]] = []
    max_gap = int(_MERGE_GAP_SEC * sr)
    for s, e in regions:
        if merged and s - merged[-1][1] <= max_gap:
            merged[-1] = (merged[-1][0], e)
        else:
            merged.append((s, e))
    return merged


# ---------------- Quick multi-speaker check (optional) ----------------
def likely_multispeaker(
    wav_path: str,
    sample_windows: int = 6,
    window_sec: float = _WINDOW_SEC,
    cosine_threshold: float = 0.82
) -> bool:
    """Heuristic: sample windows and check min pairwise cosine similarity."""
    sig, sr = _read_wav_mono16(wav_path)
    if len(sig) < sr * 1.5:
        return False
    enc = VoiceEncoder()
    total = len(sig)
    step = max(int(total / (sample_windows + 1)), int(sr * window_sec))
    embs = []
    for k in range(1, sample_windows + 1):
        mid = min(total - 1, k * step)
        start = max(0, mid - int(sr * window_sec / 2))
        end = min(total, start + int(sr * window_sec))
        if end - start < int(sr * 0.8):
            continue
        chunk = sig[start:end]
        embs.append(enc.embed_utterance(chunk, rate=sr))
    if len(embs) < 3:
        return False

    E = np.vstack(embs)
    E = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    min_cos = 1.0
    for i in range(len(E)):
        for j in range(i + 1, len(E)):
            min_cos = min(min_cos, float(E[i] @ E[j]))
    return min_cos < cosine_threshold


# ---------------- Main diarization ----------------
def diarize_speakers(
    wav_path: str,
    max_speakers: Optional[int] = None,
    window_sec: float = _WINDOW_SEC
) -> List[tuple[float, float, int]]:
    """
    Returns a list of (start_sec, end_sec, speaker_idx) spans.
    Speaker indices are 0-based integers.
    """
    sig, sr = _read_wav_mono16(wav_path)
    return diarize_speakers_from_signal(sig, sr, max_speakers, window_sec)


def diarize_speakers_from_signal(
    sig: np.ndarray,
    sr: int,
    max_speakers: Optional[int] = None,
    window_sec: float = _WINDOW_SEC
) -> List[tuple[float, float, int]]:
    """Same as diarize_speakers, but takes raw signal and sr directly."""
    vr = _voiced_regions(sig, sr)
    if not vr:
        return []

    enc = VoiceEncoder()
    segs: List[Tuple[float, float]] = []
    embs: List[np.ndarray] = []

    ws = int(window_sec * sr)
    for s, e in vr:
        i = s
        while i < e:
            j = min(e, i + ws)
            chunk = sig[i:j]
            if len(chunk) < sr * 0.5:  # skip tiny windows
                i = j
                continue
            emb = enc.embed_utterance(chunk, rate=sr)
            embs.append(emb)
            segs.append((i / sr, j / sr))
            i = j

    if not embs:
        return []

    X = np.vstack(embs)
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)

    n_spk = int(max_speakers) if max_speakers else max(2, min(6, int(round(len(embs) ** 0.5))))
    labels = AgglomerativeClustering(n_clusters=n_spk).fit_predict(X)

    diar: List[tuple[float, float, int]] = [(segs[k][0], segs[k][1], int(labels[k])) for k in range(len(segs))]
    diar.sort()

    # Merge near-adjacent spans with same label
    merged: List[tuple[float, float, int]] = []
    for st, en, sp in diar:
        if merged and merged[-1][2] == sp and (st - merged[-1][1]) <= _MERGE_GAP_LABEL_SEC:
            merged[-1] = (merged[-1][0], en, sp)
        else:
            merged.append((st, en, sp))
    return merged


# ---------------- Segment labeling helpers ----------------
def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


def _speaker_name(idx: int) -> str:
    return f"SPEAKER_{idx:02d}"


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _build_speaker_centroids(
    sig: np.ndarray,
    sr: int,
    diar: List[Tuple[float, float, int]],
    enc: VoiceEncoder,
    max_samples_per_spk: int = 20,
) -> Dict[int, np.ndarray]:
    """Average a few embeddings per speaker to get a stable centroid."""
    buckets: Dict[int, list] = {}
    for (ds, de, sp) in diar:
        s = max(0, int(ds * sr))
        e = min(len(sig), int(de * sr))
        if e - s < int(0.4 * sr):
            continue
        emb = enc.embed_utterance(sig[s:e], rate=sr)
        buckets.setdefault(int(sp), []).append(emb)

    cents = {}
    for spk, arr in buckets.items():
        if not arr:
            continue
        if len(arr) > max_samples_per_spk:
            arr = arr[:max_samples_per_spk]
        cents[spk] = np.mean(np.vstack(arr), axis=0)
    return cents


# ---------------- Segment labeling (centroid-first, overlap-fallback) ----------------
def label_segments_with_speakers(
    segments: List[dict],
    diar: List[tuple[float, float, int]],
    *,
    # If sig & sr are provided we use centroid-based labeling (recommended).
    # If not provided, we fall back to the older overlap/midpoint logic.
    sig: np.ndarray | None = None,
    sr: int | None = None,
    min_overlap: float = _MIN_OVERLAP_SEC,
    use_midpoint_fallback: bool = _USE_MIDPOINT_FALLBACK,
    smooth_win: int = 3,
) -> List[dict]:
    """
    Assign speaker to each ASR segment.

    Preferred path (more stable):
      - If `sig` and `sr` are provided: build per-speaker centroids from diarization
        and pick nearest centroid by cosine similarity, then apply short temporal smoothing.

    Fallback path (backward compatible):
      - If `sig`/`sr` are missing: use max-overlap → midpoint → nearest-span logic.
    """
    if not segments:
        return segments

    # ---------- Preferred centroid path ----------
    if sig is not None and sr is not None and diar:
        enc = VoiceEncoder()
        cents = _build_speaker_centroids(sig, sr, diar, enc)
        if cents:
            spk_ids = sorted(cents.keys())
            cent_mat = np.vstack([cents[k] for k in spk_ids])

            labels: List[int] = []
            for s in segments:
                st = float(s.get("start", 0.0))
                en = float(s.get("end", st))
                a = max(0, int(st * sr))
                b = min(len(sig), int(en * sr))
                # ensure we have at least ~400 ms; if not, center-pad
                if b - a < int(0.4 * sr):
                    mid = int(((st + en) * 0.5) * sr)
                    half = int(0.4 * sr)
                    a = max(0, mid - half)
                    b = min(len(sig), mid + half)

                emb = enc.embed_utterance(sig[a:b], rate=sr)
                sims = [_cos_sim(emb, cent) for cent in cent_mat]
                sp = spk_ids[int(np.argmax(sims))]
                labels.append(int(sp))

            # temporal smoothing (simple mode)
            if smooth_win > 1:
                half = smooth_win // 2
                sm_labels = []
                for i in range(len(labels)):
                    lo = max(0, i - half)
                    hi = min(len(labels), i + half + 1)
                    window = labels[lo:hi]
                    vals, counts = np.unique(window, return_counts=True)
                    sm_labels.append(int(vals[int(np.argmax(counts))]))
                labels = sm_labels

            for s, sp in zip(segments, labels):
                s["speaker"] = _speaker_name(sp)
            return segments

    # ---------- Fallback: overlap / midpoint / nearest ----------
    if not diar:
        for s in segments:
            s["speaker"] = _speaker_name(0)
        return segments

    for s in segments:
        st = float(s.get("start", 0.0))
        en = float(s.get("end", st))
        best_sp, best_ov = None, 0.0

        for ds, de, sp in diar:
            ov = _overlap(st, en, ds, de)
            if ov > best_ov:
                best_ov, best_sp = ov, int(sp)

        # Midpoint fallback
        if (best_sp is None or best_ov < min_overlap) and use_midpoint_fallback:
            mid = 0.5 * (st + en)
            for ds, de, sp in diar:
                if ds <= mid <= de:
                    best_sp = int(sp)
                    break

        # Nearest span fallback
        if best_sp is None:
            best_sp = min(diar, key=lambda t: min(abs(t[0] - st), abs(t[1] - en)))[2]

        s["speaker"] = _speaker_name(int(best_sp))

    return segments
