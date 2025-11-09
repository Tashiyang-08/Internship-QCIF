# asr_backend/asr/registry.py
from __future__ import annotations
from typing import Dict, Callable, Optional
import logging
import os
from pathlib import Path

log = logging.getLogger(__name__)

# -------------------------------------------------------------------
# Resolve paths relative to backend root (one level above /asr)
# -------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"

# -------------------------
# Global env defaults (for faster-whisper device/compute)
# -------------------------
FW_DEVICE = os.getenv("FASTER_WHISPER_DEVICE", "cpu").lower()
FW_COMPUTE = os.getenv("FASTER_WHISPER_COMPUTE_TYPE", "int8").lower()

# -------------------------------------------------------------------
# Factory helpers (lazy load actual engines only in get_model)
# -------------------------------------------------------------------
def _make_faster_whisper(size: str) -> Callable[[], object]:
    """Return a factory for FasterWhisperEngine(size, device, compute_type)."""
    def _factory():
        try:
            from asr_backend.asr.faster_whisper import FasterWhisperEngine
            return FasterWhisperEngine(
                size,
                device=FW_DEVICE,
                compute_type=FW_COMPUTE,
            )
        except Exception as e:
            raise RuntimeError(f"faster-whisper({size}) unavailable: {e}")
    return _factory


def _make_wav2vec2(hf_name: str) -> Callable[[], object]:
    """Return a factory for Wav2Vec2Engine using a Hugging Face model."""
    def _factory():
        try:
            from asr_backend.asr.wav2vec2_asr import Wav2Vec2Engine
            return Wav2Vec2Engine(hf_name)
        except Exception as e:
            raise RuntimeError(f"wav2vec2 '{hf_name}' unavailable: {e}")
    return _factory


def _make_vosk(model_dir: str | os.PathLike[str]) -> Callable[[], object]:
    """Return a factory that builds a VoskEngine for the given model_dir."""
    model_dir = str(model_dir)
    def _factory_resolved():
        try:
            from asr_backend.asr.vosk_asr import VoskEngine
            return VoskEngine(model_dir)
        except Exception as e:
            raise RuntimeError(f"vosk model at '{model_dir}' unavailable: {e}")
    return _factory_resolved

# -------------------------------------------------------------------
# Env resolution for Vosk paths
# -------------------------------------------------------------------
def _resolve_vosk_path(default_path: Path, specific_env: str) -> str:
    """
    Order:
      1) specific_env (e.g., VOSK_MODEL_DIR_SMALL)
      2) VOSK_MODEL_DIR
      3) default_path
    """
    p = os.environ.get(specific_env)
    if p:
        return p
    p = os.environ.get("VOSK_MODEL_DIR")
    if p:
        return p
    return str(default_path)

# -------------------------------------------------------------------
# Probes (must be LIGHT; no heavy model load)
# -------------------------------------------------------------------
def _probe_faster_whisper() -> bool:
    try:
        import asr_backend.asr.faster_whisper  # noqa
        return True
    except Exception as e:
        log.info(f"[registry] faster-whisper probe failed: {e}")
        return False


def _probe_wav2vec2() -> bool:
    try:
        import asr_backend.asr.wav2vec2_asr  # noqa
        import transformers  # noqa
        return True
    except Exception as e:
        log.info(f"[registry] wav2vec2 probe failed: {e}")
        return False


def _probe_vosk_dir(path: str) -> bool:
    if os.path.isdir(path):
        try:
            import vosk  # noqa
            return True
        except Exception as e:
            log.info(f"[registry] vosk import failed: {e}")
            return False
    else:
        log.info(f"[registry] vosk model dir missing: {path}")
        return False

# -------------------------------------------------------------------
# Resolve Vosk paths once
# -------------------------------------------------------------------
_VOSK_SMALL_PATH = _resolve_vosk_path(
    default_path=MODELS_DIR / "vosk-model-small-en-us-0.15",
    specific_env="VOSK_MODEL_DIR_SMALL",
)
_VOSK_LARGE_PATH = _resolve_vosk_path(
    default_path=MODELS_DIR / "vosk-model-en-us-0.22",
    specific_env="VOSK_MODEL_DIR_LARGE",
)

# -------------------------------------------------------------------
# Candidate model definitions
# -------------------------------------------------------------------
_CANDIDATES: Dict[str, tuple[Callable[[], bool], Callable[[], object]]] = {
    # 🔹 Faster-Whisper family (CPU/GPU)
    "faster-whisper-tiny":    (_probe_faster_whisper, _make_faster_whisper("tiny")),
    "faster-whisper-base":    (_probe_faster_whisper, _make_faster_whisper("base")),
    "faster-whisper-small":   (_probe_faster_whisper, _make_faster_whisper("small")),
    "faster-whisper-medium":  (_probe_faster_whisper, _make_faster_whisper("medium")),
    "faster-whisper-large-v2":(_probe_faster_whisper, _make_faster_whisper("large-v2")),
    "faster-whisper-large-v3":(_probe_faster_whisper, _make_faster_whisper("large-v3")),

    # 🔹 Wav2Vec2 (Hugging Face CTC models)
    "wav2vec2-base-en": (
        _probe_wav2vec2,
        _make_wav2vec2("facebook/wav2vec2-base-960h"),
    ),
    "wav2vec2-large-xlsr-en": (
        _probe_wav2vec2,
        _make_wav2vec2("jonatasgrosman/wav2vec2-large-xlsr-53-english"),
    ),
    "wav2vec2-large-xlsr": (
        _probe_wav2vec2,
        _make_wav2vec2("facebook/wav2vec2-large-xlsr-53"),
    ),

    # 🔹 Vosk (offline models)
    "vosk-small-en": (
        lambda: _probe_vosk_dir(_VOSK_SMALL_PATH),
        _make_vosk(_VOSK_SMALL_PATH),
    ),
    "vosk-large-en": (
        lambda: _probe_vosk_dir(_VOSK_LARGE_PATH),
        _make_vosk(_VOSK_LARGE_PATH),
    ),
}

# -------------------------------------------------------------------
# Lazy registry & model cache
# -------------------------------------------------------------------
_REGISTRY: Dict[str, Callable[[], object]] = {}
_MODELS: Dict[str, object] = {}

# -------------------------------------------------------------------
# Public API
# -------------------------------------------------------------------
def available_models() -> list[str]:
    """
    Return models that *appear available* using LIGHT probes.
    No heavy model instantiation here.
    """
    if _REGISTRY:
        return list(_REGISTRY.keys())

    for name, (probe, factory) in _CANDIDATES.items():
        try:
            if probe():
                _REGISTRY[name] = factory
            else:
                log.info(f"[registry] Skipping '{name}' (probe failed)")
        except Exception as e:
            log.info(f"[registry] Skipping '{name}': {e}")

    return list(_REGISTRY.keys())


def get_model(name: str, *, device: Optional[str] = None, compute_type: Optional[str] = None):
    """
    Instantiate and cache the requested model engine (heavy load happens here).
    """
    if not _REGISTRY:
        available_models()  # initialize registry via probes

    if name not in _REGISTRY:
        raise RuntimeError(f"Model not found: {name}")

    if name not in _MODELS:
        eng = _REGISTRY[name]()
        log.info(f"[registry] Loaded engine for '{name}': {type(eng).__name__}")
        _MODELS[name] = eng

    return _MODELS[name]
