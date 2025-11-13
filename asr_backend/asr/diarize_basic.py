# asr_backend/asr/diarize_basic.py
from __future__ import annotations

print(">>> diarize_basic: energy-VAD 16kHz + centroid labeling (stable, auto-K)")

from typing import List, Optional, Tuple, Dict
import numpy as np
import soundfile as sf
import librosa

# Speaker embeddings
from resemblyzer import VoiceEncoder

# Clustering & metrics
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score

# ========================== Tunables ==========================
# VAD framing
_FRAME_MS = 30
_HOP_MS = 15
_SMOOTH_MS = 200

# Energy gate
_ENERGY_QUANTILE = 0.65
_MIN_REGION_MS = 350

# Merging voiced islands
_MERGE_GAP_SEC = 0.50            # merge VAD islands closer than this
_MERGE_GAP_LABEL_SEC = 0.50      # merge diarized spans of same label within this gap

# Speaker windowing for embeddings
_WINDOW_SEC = 1.20
_MIN_EMB_WIN_SEC = 0.40          # ensure we embed at least 400 ms

# Turn-stability / anti-flip
_MIN_TURN_SEC = 1.00             # drop or relabel turns shorter than this
_HOLD_SEC = 0.60                 # require this hold before switching speakers

# Label smoothing (centroid path)
_LABEL_SMOOTH_WIN = 3            # majority filter width

# Silhouette-based auto K (when max_speakers is None)
_MIN_K = 1                       # allow auto-detecting single-speaker audio
_MAX_K = 6                       # raise to 8–10 for large meetings
_AUTO_SIL_FLOOR = 0.06           # if best silhouette < floor ⇒ use 1 speaker
_MIN_CLUSTER_SEC = 2.0           # reject K that creates clusters with <2s speech

# Overlap labeling fallback
_MIN_OVERLAP_SEC = 0.05
_USE_MIDPOINT_FALLBACK = True
# ===============================================================


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

    # Smooth with moving average in frame domain
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


# ---------------- Utilities ----------------
def _speaker_name(idx: int) -> str:
    return f"SPEAKER_{idx:02d}"


def _cos_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a, b))


def _overlap(a1: float, a2: float, b1: float, b2: float) -> float:
    return max(0.0, min(a2, b2) - max(a1, b1))


# ---------------- Embedding helpers ----------------
def _embed_chunks(
    sig: np.ndarray,
    sr: int,
    regions: List[tuple[int, int]],
    window_sec: float
) -> tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Slide a window across each voiced region and embed chunks.
    Returns (segment_times, embeddings_matrix).
    """
    enc = VoiceEncoder()
    segs: List[Tuple[float, float]] = []
    embs: List[np.ndarray] = []

    ws = int(window_sec * sr)
    minw = int(_MIN_EMB_WIN_SEC * sr)

    for s, e in regions:
        i = s
        while i < e:
            j = min(e, i + ws)
            chunk = sig[i:j]
            if len(chunk) < minw:
                i = j
                continue
            emb = enc.embed_utterance(chunk, rate=sr)
            embs.append(emb)
            segs.append((i / sr, j / sr))
            i = j

    if not embs:
        return [], np.zeros((0, 256), dtype=np.float32)

    X = np.vstack(embs).astype(np.float32)
    # L2 normalize
    X /= (np.linalg.norm(X, axis=1, keepdims=True) + 1e-9)
    return segs, X


# ---------------- Auto-K helpers ----------------
def _cluster_durations(labels: np.ndarray, segs: list[tuple[float, float]]) -> dict[int, float]:
    """Total seconds per cluster label."""
    dur: dict[int, float] = {}
    for lab, (s, e) in zip(labels, segs):
        dur[int(lab)] = dur.get(int(lab), 0.0) + max(0.0, float(e - s))
    return dur


def _choose_k_by_silhouette(
    X: np.ndarray,
    segs: list[tuple[float, float]],
    kmin: int,
    kmax: int,
    *,
    sil_floor: float = _AUTO_SIL_FLOOR,
    min_cluster_sec: float = _MIN_CLUSTER_SEC,
    random_state: int = 0
) -> int:
    """
    Pick K by maximizing cosine-silhouette using KMeans on normalized embeddings.
    Reject K that produce any tiny cluster (< min_cluster_sec).
    Fall back to K=1 if the best silhouette is weak.
    """
    n = X.shape[0]
    if n < 3:
        return 1

    lo = max(1, kmin)
    hi = min(kmax, n - 1)  # silhouette requires n_samples > n_clusters

    best_k, best_score = 1, -1.0
    for k in range(max(lo, 2), hi + 1):  # silhouette undefined for k=1
        try:
            km = KMeans(n_clusters=k, n_init=10, random_state=random_state)
            labels = km.fit_predict(X)

            # tiny cluster check
            durs = _cluster_durations(labels, segs)
            if any(v < min_cluster_sec for v in durs.values()):
                continue

            sc = silhouette_score(X, labels, metric="cosine")
            if sc > best_score:
                best_score, best_k = sc, k
        except Exception:
            continue

    if best_score < sil_floor:
        return 1
    return int(best_k)


# ---------------- Turn post-processing ----------------
def _postprocess_turns(
    diar: List[tuple[float, float, int]]
) -> List[tuple[float, float, int]]:
    """
    Stabilize diarization:
    - merge identical adjacent speakers across small gaps
    - drop or relabel very short turns (< _MIN_TURN_SEC) using neighbors
    - simple hold/hysteresis to reduce rapid flipping
    """
    if not diar:
        return diar

    diar = sorted(diar)
    out: List[tuple[float, float, int]] = []

    # 1) merge adjacency with small gaps
    for st, en, sp in diar:
        if out and out[-1][2] == sp and (st - out[-1][1]) <= _MERGE_GAP_LABEL_SEC:
            out[-1] = (out[-1][0], en, sp)
        else:
            out.append((st, en, sp))

    # 2) drop/relabel tiny turns
    i = 0
    while i < len(out):
        st, en, sp = out[i]
        dur = en - st
        if dur < _MIN_TURN_SEC and 0 < i < len(out) - 1:
            prev_sp = out[i - 1][2]
            next_sp = out[i + 1][2]
            if prev_sp == next_sp:
                # fold into neighbors
                out[i - 1] = (out[i - 1][0], out[i + 1][1], prev_sp)
                del out[i:i + 2]
                i = max(i - 1, 0)
                continue
            else:
                # relabel as longer neighbor
                left_d = out[i - 1][1] - out[i - 1][0]
                right_d = out[i + 1][1] - out[i + 1][0]
                new_sp = out[i - 1][2] if left_d >= right_d else out[i + 1][2]
                out[i] = (st, en, new_sp)
        i += 1

    # 3) hold/hysteresis: require HOLD_SEC presence before switching
    stable: List[tuple[float, float, int]] = []
    for seg in out:
        if not stable:
            stable.append(seg)
            continue
        pst, pen, psp = stable[-1]
        st, en, sp = seg
        if sp != psp:
            if (en - st) >= _HOLD_SEC:
                stable.append(seg)
            else:
                # attach short blip to previous
                stable[-1] = (pst, en, psp)
        else:
            # extend
            stable[-1] = (pst, en, psp)

    # final adjacent merge
    merged: List[tuple[float, float, int]] = []
    for st, en, sp in stable:
        if merged and merged[-1][2] == sp and (st - merged[-1][1]) <= _MERGE_GAP_LABEL_SEC:
            merged[-1] = (merged[-1][0], en, sp)
        else:
            merged.append((st, en, sp))
    return merged


# ---------------- Public diarization APIs ----------------
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

    # Embed chunks
    segs, X = _embed_chunks(sig, sr, vr, window_sec)
    if X.shape[0] == 0:
        return []

    # Decide number of speakers
    if isinstance(max_speakers, int) and max_speakers > 0:
        n_spk = max_speakers
    else:
        n_spk = _choose_k_by_silhouette(X, segs, _MIN_K, _MAX_K)

    # Cluster (or trivial 1-cluster)
    if n_spk == 1:
        labels = np.zeros((len(segs),), dtype=int)
    else:
        labels = AgglomerativeClustering(n_clusters=n_spk).fit_predict(X)

    # Build raw diar list
    diar: List[tuple[float, float, int]] = [(segs[k][0], segs[k][1], int(labels[k])) for k in range(len(segs))]
    diar.sort()

    # Stabilize turns
    diar = _postprocess_turns(diar)
    return diar


# ---------------- Centroid building & labeling ----------------
def _build_speaker_centroids(
    sig: np.ndarray,
    sr: int,
    diar: List[Tuple[float, float, int]],
    enc: VoiceEncoder | None = None,
    max_samples_per_spk: int = 20,
) -> Dict[int, np.ndarray]:
    """Average a few embeddings per speaker to get a stable centroid."""
    if enc is None:
        enc = VoiceEncoder()
    buckets: Dict[int, list] = {}
    for (ds, de, sp) in diar:
        s = max(0, int(ds * sr))
        e = min(len(sig), int(de * sr))
        if e - s < int(_MIN_EMB_WIN_SEC * sr):
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


def label_segments_with_speakers(
    segments: List[dict],
    diar: List[tuple[float, float, int]],
    * ,
    sig: np.ndarray | None = None,
    sr: int | None = None,
    min_overlap: float = _MIN_OVERLAP_SEC,
    use_midpoint_fallback: bool = _USE_MIDPOINT_FALLBACK,
    smooth_win: int = _LABEL_SMOOTH_WIN,
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
            minw = int(_MIN_EMB_WIN_SEC * sr)

            for s in segments:
                st = float(s.get("start", 0.0))
                en = float(s.get("end", st))
                a = max(0, int(st * sr))
                b = min(len(sig), int(en * sr))
                # ensure enough audio; if not, center-pad a small window
                if (b - a) < minw:
                    mid = int(((st + en) * 0.5) * sr)
                    half = minw // 2
                    a = max(0, mid - half)
                    b = min(len(sig), mid + half)

                emb = enc.embed_utterance(sig[a:b], rate=sr)
                sims = [_cos_sim(emb, cent) for cent in cent_mat]
                sp = spk_ids[int(np.argmax(sims))]
                labels.append(int(sp))

            # temporal smoothing (simple majority)
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
