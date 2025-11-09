from __future__ import annotations
from typing import Optional
import os, json, subprocess, tempfile, shutil, wave, logging, shutil as _shutil

from vosk import Model, KaldiRecognizer

VOSK_MODEL_DIR = os.environ.get("VOSK_MODEL_DIR", "models/vosk-model-small-en-us-0.15")
TARGET_SR = 16000
TARGET_CH = 1
TARGET_SAMPWIDTH = 2  # 16-bit PCM
_LOG = logging.getLogger(__name__)

def _fmt_time(t: float, pretty: bool, round_secs: int):
    if pretty:
        ms = int((t % 1) * 1000); s = int(t) % 60; m = int(t / 60) % 60; h = int(t / 3600)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"
    return round(t, round_secs)

def _ffmpeg_bin() -> str:
    """Locate ffmpeg (FFMPEG_BIN env, PATH, or common winget path)."""
    env_bin = os.environ.get("FFMPEG_BIN")
    if env_bin and os.path.isfile(env_bin): return env_bin
    which = _shutil.which("ffmpeg") or _shutil.which("ffmpeg.exe")
    if which: return which
    winget = r"C:\Users\tashi\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0-full_build\bin\ffmpeg.exe"
    if os.path.isfile(winget): return winget
    raise RuntimeError("ffmpeg not found. Set FFMPEG_BIN or add ffmpeg to PATH.")

def _wav_props(path: str) -> tuple[int,int,int]:
    with wave.open(path, "rb") as wf:
        return wf.getframerate(), wf.getnchannels(), wf.getsampwidth()

def _needs_convert_wav(path: str) -> bool:
    """True if not WAV or not PCM16 mono 16k."""
    try:
        sr, ch, sw = _wav_props(path)
        return not (sr == TARGET_SR and ch == TARGET_CH and sw == TARGET_SAMPWIDTH)
    except wave.Error:
        return True  # not a WAV at all

def _convert_to_wav16k_mono(src: str) -> str:
    ffmpeg = _ffmpeg_bin()
    tmpdir = tempfile.mkdtemp(prefix="vosk_")
    dst = os.path.join(tmpdir, "audio_16k_mono.wav")
    cmd = [
        ffmpeg, "-v", "error", "-y", "-i", src,
        "-ac", str(TARGET_CH), "-ar", str(TARGET_SR),
        "-acodec", "pcm_s16le", "-f", "wav", dst
    ]
    try:
        subprocess.check_call(cmd)
    except subprocess.CalledProcessError as e:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(f"ffmpeg failed to convert audio (exit {e.returncode}).")
    except FileNotFoundError:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError("ffmpeg executable not found.")
    return dst

class VoskEngine:
    def __init__(self, model_path: str | None = None):
        model_path = model_path or VOSK_MODEL_DIR
        if not os.path.isdir(model_path):
            raise RuntimeError(
                f"Vosk model dir not found: {model_path}\n"
                "Download from https://alphacephei.com/vosk/models and set VOSK_MODEL_DIR."
            )
        try:
            self.model = Model(model_path)
        except Exception as e:
            raise RuntimeError(f"Failed to load Vosk model at {model_path}: {e}")

    def transcribe(self, path: str, *, language: Optional[str] = None,
                   pretty_time: bool = False, round_secs: int = 2):
        cleanup_dir = None
        use_path = path
        if _needs_convert_wav(path):
            use_path = _convert_to_wav16k_mono(path)
            cleanup_dir = os.path.dirname(use_path)

        try:
            with wave.open(use_path, "rb") as wf:
                sr = wf.getframerate() or TARGET_SR
                nframes = wf.getnframes()
                if nframes <= 0:
                    raise ValueError("Input audio has zero frames after conversion.")
                rec = KaldiRecognizer(self.model, sr)
                rec.SetWords(True)

                frame_size = 4000  # ~0.25s at 16k
                while True:
                    data = wf.readframes(frame_size)
                    if not data:
                        break
                    rec.AcceptWaveform(data)

                final = json.loads(rec.FinalResult() or "{}")
                text = (final.get("text") or "").strip()
                raw_words = final.get("result") or final.get("words") or []

                segs = []
                for w in raw_words:
                    start = float(w.get("start", 0.0)); end = float(w.get("end", start))
                    word = (w.get("word") or "").strip()
                    if not word: continue
                    segs.append({
                        "start": _fmt_time(start, pretty_time, round_secs),
                        "end": _fmt_time(end, pretty_time, round_secs),
                        "text": word,
                    })

                duration = nframes / float(sr)

            return {
                "text": text,
                "segments": segs,
                "language": language or "en",
                "duration_sec": duration,
            }
        finally:
            if cleanup_dir:
                shutil.rmtree(cleanup_dir, ignore_errors=True)
