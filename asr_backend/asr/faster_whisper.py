# asr_backend/asr/faster_whisper.py
from __future__ import annotations

import os
from typing import Optional, Dict, Any, List

try:
    # faster-whisper 1.x
    from faster_whisper import WhisperModel
except Exception as e:
    raise RuntimeError(
        f"Failed to import faster_whisper: {type(e).__name__}: {e}. "
        "Install with: pip install 'faster-whisper==1.0.3'"
    )


class FasterWhisperEngine:
    """
    Thin wrapper used by our registry/chooser.

    IMPORTANT: We turn OFF Silero VAD by default (vad_filter=False) to avoid
    Windows + onnxruntime + protobuf issues that throw:
        INVALID_PROTOBUF: silero_vad.onnx failed: Protobuf parsing failed
    You can re-enable it later by setting env FW_DISABLE_VAD=0.
    """

    def __init__(
        self,
        model_name: str,
        device: Optional[str] = None,
        compute_type: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.device = (device or "cpu")
        self.compute_type = (compute_type or "int8")
        self._model = WhisperModel(
            model_name,
            device=self.device,
            compute_type=self.compute_type,
        )

    def transcribe(self, path: str, language: Optional[str] = None) -> Dict[str, Any]:
        # If FW_DISABLE_VAD is "1" (default), don't use Silero VAD at all.
        # This completely bypasses loading silero_vad.onnx (the source of the crash).
        vad_disabled = os.getenv("FW_DISABLE_VAD", "1") == "1"

        # NOTE: You can pass additional knobs here if you like (beam_size, temperature, etc.)
        seg_iter, info = self._model.transcribe(
            path,
            language=language,
            vad_filter=not vad_disabled,  # False by default → no Silero VAD
        )

        segments: List[Dict[str, Any]] = []
        last_end = 0.0
        for s in seg_iter:
            st = float(getattr(s, "start", 0.0) or 0.0)
            en = float(getattr(s, "end", st) or st)
            last_end = max(last_end, en)
            segments.append(
                {
                    "start": st,
                    "end": en,
                    "text": (getattr(s, "text", "") or "").strip(),
                }
            )

        text = " ".join(seg.get("text", "") for seg in segments).strip()
        out: Dict[str, Any] = {
            "text": text,
            "segments": segments,
            "language": getattr(info, "language", None) or None,
            "language_prob": getattr(info, "language_probability", None),
            "duration_sec": float(getattr(info, "duration", last_end) or last_end),
            "model": self.model_name,
        }
        return out
