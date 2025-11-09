# asr_backend/asr/faster_whisper.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

from faster_whisper import WhisperModel

# Env defaults
FW_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu").lower()      # "cpu" | "cuda"
FW_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8").lower()
MODEL_CACHE_DIR = os.getenv(
    "ASR_MODEL_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "asr_models")
)
CPU_THREADS = int(os.getenv("FASTER_WHISPER_CPU_THREADS", "0"))    # 0 = library default
NUM_WORKERS = int(os.getenv("FASTER_WHISPER_NUM_WORKERS", "1"))    # decoder workers

class FasterWhisperEngine:
    """
    Thin wrapper around faster-whisper's WhisperModel.

    Args:
      size:         "tiny" | "base" | "small" | "medium" | "large-v3" | HF repo id
      device:       "cpu" or "cuda" (defaults from env)
      compute_type: e.g., "int8" (CPU), "float16" (CUDA), "int8_float16", "int16", "float32"
    """

    def __init__(
        self,
        size: str,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.model_name = size
        self.device = (device or FW_DEVICE or "cpu").lower()
        self.compute_type = (compute_type or FW_COMPUTE or "int8").lower()

        os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

        # IMPORTANT: use download_root (NOT cache_directory)
        self._model = WhisperModel(
            model_size_or_path=self.model_name,
            device=self.device,
            compute_type=self.compute_type,
            cpu_threads=CPU_THREADS,
            num_workers=NUM_WORKERS,
            download_root=MODEL_CACHE_DIR,
            local_files_only=False,  # set True if you want to forbid downloads
        )

    def transcribe(
        self,
        audio_path: str,
        *,
        language: Optional[str] = None,
        beam_size: int = 5,
        vad_filter: bool = True,
        condition_on_previous_text: bool = True,
    ) -> Dict[str, Any]:
        seg_iter, info = self._model.transcribe(
            audio_path,
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            condition_on_previous_text=condition_on_previous_text,
        )

        segments: List[Dict[str, Any]] = []
        for seg in seg_iter:
            segments.append({
                "id": getattr(seg, "id", None),
                "start": float(getattr(seg, "start", 0.0) or 0.0),
                "end": float(getattr(seg, "end", 0.0) or 0.0),
                "text": (getattr(seg, "text", "") or "").strip(),
            })

        return {
            "text": " ".join(s["text"] for s in segments).strip(),
            "segments": segments,
            "duration_sec": float(getattr(info, "duration", 0.0) or 0.0),
            "language": getattr(info, "language", None),
            "language_prob": float(getattr(info, "language_probability", 0.0) or 0.0),
            "model": self.model_name,
        }
