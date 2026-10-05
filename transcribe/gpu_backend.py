"""whisper.cpp (Vulkan, AMD GPU) backend for Transcribe.

Drop-in for the two faster-whisper calls transcribe_gui.py makes:
    model = WhisperCppModel(name)            # like WhisperModel(name, ...)
    segments, info = model.transcribe(path, language=..., beam_size=..., initial_prompt=...)
Segments have .start / .end / .text (seconds), info has .duration - same as faster-whisper.

Models are ggml files (ggml-<name>.bin) in MODELS_DIR, downloaded on first use.
"""
import os
from pathlib import Path
from types import SimpleNamespace

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"      # shared with the other apps


def available():
    try:
        import pywhispercpp.model  # noqa: F401
        return True
    except Exception:
        return False


def model_is_cached(name):
    return (MODELS_DIR / f"ggml-{name}.bin").exists()


class WhisperCppModel:
    def __init__(self, name, gpu=True, n_threads=None, beam_size=5):
        from pywhispercpp.model import Model

        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.beam_size = beam_size
        self.model = Model(
            name,
            models_dir=str(MODELS_DIR),
            params_sampling_strategy=1 if beam_size > 1 else 0,
            redirect_whispercpp_logs_to=False,
            context_params={"use_gpu": bool(gpu), "flash_attn": bool(gpu)},
            n_threads=n_threads or min(8, os.cpu_count() or 4),
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            print_special=False,
            suppress_blank=True,
            suppress_nst=True,
        )
        self.device = "gpu" if gpu else "cpu"
        # First call compiles the Vulkan pipelines (~2 s): do it on a full 30 s window of silence now.
        import numpy as np
        self.model.transcribe(np.zeros(16000 * 30, dtype=np.float32), language="en")

    def transcribe(self, path, language=None, beam_size=None, initial_prompt=None,
                   task="transcribe", **_ignored):
        from faster_whisper.audio import decode_audio  # PyAV decoder, already bundled

        audio = decode_audio(str(path), sampling_rate=16000)
        duration = len(audio) / 16000.0
        params = {
            "language": language or "auto",
            "translate": task == "translate",
        }
        if initial_prompt:
            params["initial_prompt"] = initial_prompt
        bs = beam_size or self.beam_size
        if bs and bs > 1:
            params["beam_search"] = {"beam_size": int(bs), "patience": -1.0}
        raw = self.model.transcribe(audio, **params)
        segments = [
            SimpleNamespace(start=s.t0 / 100.0, end=s.t1 / 100.0, text=s.text)
            for s in raw if s.text and s.text.strip()
        ]
        info = SimpleNamespace(duration=duration, language=params["language"])
        return segments, info
