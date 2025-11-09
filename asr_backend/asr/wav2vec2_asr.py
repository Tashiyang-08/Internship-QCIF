# asr/wav2vec2_asr.py
from typing import Optional, Tuple
import logging
import numpy as np

DEFAULT_MODEL_ID = "facebook/wav2vec2-base-960h"
_LOG = logging.getLogger(__name__)

def _fmt_time(t: float, pretty: bool, round_secs: int):
    if pretty:
        ms = int((t % 1) * 1000); s = int(t) % 60; m = (int(t) // 60) % 60; h = int(t) // 3600
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    return round(float(t), round_secs)

def _safe_load_audio(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, int, float]:
    """
    Load mono float32 audio at target_sr with robust fallbacks.
    Returns (audio, sr, duration_sec). Raises ValueError on failure.
    """
    # 1) librosa (works for many formats; ensure 0.10.2+ for numpy 2.x)
    try:
        import librosa
        audio, sr = librosa.load(path, sr=target_sr, mono=True)
        if audio is None or audio.size == 0:
            raise ValueError("Loaded audio is empty (librosa).")
        duration = float(audio.shape[-1]) / float(target_sr)
        return np.asarray(audio, dtype=np.float32), target_sr, duration
    except Exception as e:
        _LOG.warning("librosa.load failed: %s", e)

    # 2) soundfile (needs libsndfile)
    try:
        import soundfile as sf
        audio, sr = sf.read(path, dtype="float32", always_2d=False)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as e_rs:
                _LOG.warning("Resample fallback failed: %s", e_rs)
        if audio is None or audio.size == 0:
            raise ValueError("Loaded audio is empty (soundfile).")
        duration = float(audio.shape[-1]) / float(sr)
        return np.asarray(audio, dtype=np.float32), sr, duration
    except Exception as e:
        _LOG.warning("soundfile.read failed: %s", e)

    # 3) scipy (wav only)
    try:
        from scipy.io import wavfile
        sr, audio = wavfile.read(path)
        audio = audio.astype(np.float32, copy=False)
        if audio.dtype.kind in ("i", "u"):
            maxv = np.iinfo(audio.dtype).max
            audio = audio / float(maxv)
        if audio.ndim == 2:
            audio = np.mean(audio, axis=1)
        if sr != target_sr:
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
                sr = target_sr
            except Exception as e_rs2:
                _LOG.warning("Resample fallback 2 failed: %s", e_rs2)
        if audio is None or audio.size == 0:
            raise ValueError("Loaded audio is empty (scipy).")
        duration = float(audio.shape[-1]) / float(sr)
        return np.asarray(audio, dtype=np.float32), sr, duration
    except Exception as e:
        _LOG.warning("scipy.io.wavfile.read failed: %s", e)

    raise ValueError(
        "Audio load failed. Ensure compatible deps: librosa>=0.10.2.post1, soundfile, audioread, scipy; "
        "or provide a readable WAV file."
    )

class Wav2Vec2Engine:
    """Wav2Vec2 via HF pipeline; feed NumPy audio so ffmpeg isn’t required by the pipeline."""

    def __init__(self, model_id: str = DEFAULT_MODEL_ID, device: str = "cpu"):
        from transformers import pipeline

        dev = -1
        try:
            if isinstance(device, str):
                d = device.lower()
                if d.startswith("cuda"): dev = 0
                elif d.isdigit(): dev = int(d)
        except Exception:
            dev = -1

        self.model_id = model_id or DEFAULT_MODEL_ID
        self.asr = pipeline(task="automatic-speech-recognition", model=self.model_id, device=dev)
        _LOG.info("[Wav2Vec2Engine] Loaded model=%s device=%s", self.model_id, dev)

        # Some transformers versions don’t accept chunking args; we’ll probe by trying once later.
        self._supports_chunking = True

    def transcribe(
        self,
        path: str,
        *,
        language: Optional[str] = None,
        pretty_time: bool = False,
        round_secs: int = 2,
        max_minutes: float = 180.0,
    ):
        audio, sr, duration = _safe_load_audio(path, target_sr=16000)
        if duration <= 0:
            raise ValueError("Audio duration is zero.")
        if duration > max_minutes * 60:
            raise ValueError(f"Audio too long ({duration/60:.1f} min). Limit ~{max_minutes} min for wav2vec2.")

        sample = {"array": np.asarray(audio, dtype=np.float32), "sampling_rate": 16000}

        kwargs = {}
        if self._supports_chunking:
            kwargs.update(dict(chunk_length_s=15, stride_length_s=(2, 2)))  # safe chunking

        # Do NOT pass return_timestamps; many wav2vec2 pipelines don’t support it.
        try:
            result = self.asr(sample, **kwargs)
        except TypeError as e:
            # Retry without chunk args if this transformers version doesn’t support them
            _LOG.warning("ASR pipeline args not supported (%s). Retrying without chunking.", e)
            self._supports_chunking = False
            result = self.asr(sample)
        except Exception:
            _LOG.exception("ASR pipeline failed")
            raise

        text = ""
        if isinstance(result, dict):
            if "text" in result:
                text = (result["text"] or "").strip()
            elif "chunks" in result and isinstance(result["chunks"], list):
                text = " ".join((ch.get("text") or "").strip() for ch in result["chunks"]).strip()

        segments = [{
            "start": _fmt_time(0.0, pretty_time, round_secs),
            "end": _fmt_time(duration, pretty_time, round_secs),
            "text": text,
        }]

        return {
            "text": text,
            "segments": segments,
            "language": language or "en",
            "duration_sec": float(duration),
        }
