"""Speech recognition backends.

* WhisperCppBackend    - whisper.cpp via pywhispercpp, built with the Vulkan backend so it
                         runs on the AMD Radeon GPU. Preferred.
* FasterWhisperBackend - CTranslate2 on CPU. Fallback when the GPU build is missing.
* QwenAsrBackend       - Qwen3-ASR through llama.cpp's server (Vulkan). VoiceType's English
                         dictation; transcribe only.

All expose ``run(audio, task, language, prompt=None) -> Result`` where ``audio`` is
16 kHz float32 and ``task`` is "transcribe" (Turkish text) or "translate" (Whisper's own
Turkish -> English). ``Result.confidence`` is the mean token probability (0..1).
"""
import base64
import io
import json
import math
import os
import re
import subprocess
import sys
import time
import urllib.request
import wave
from dataclasses import dataclass

import numpy as np

from paths import app_root

ROOT = app_root()
MODELS_DIR = ROOT / "models"

# Whisper's encoder always sees a 30 s window (1500 positions). For short utterances we
# can shrink the encoder context proportionally (whisper.cpp ``audio_ctx``), which cuts
# encoder time a lot but costs accuracy. Only used with --audio-ctx auto.
AUDIO_CTX_FULL = 1500
AUDIO_CTX_MIN = 768


def dynamic_audio_ctx(seconds):
    ctx = int(math.ceil(seconds / 30.0 * AUDIO_CTX_FULL)) + 64
    return max(AUDIO_CTX_MIN, min(AUDIO_CTX_FULL, ctx))


@dataclass
class Result:
    text: str
    confidence: float = 1.0


def model_path(model):
    p = MODELS_DIR / f"ggml-{model}.bin"
    if not p.exists():
        raise FileNotFoundError(f"Whisper model not found: {p}  (run get_models.py)")
    return p


class WhisperCppBackend:
    def __init__(self, model="large-v3-turbo", threads=None, gpu=True, audio_ctx="full", beam=5,
                 fallback=True, **_):
        from pywhispercpp.model import Model  # noqa: WPS433 (optional dependency)

        self.threads = threads or min(8, os.cpu_count() or 4)
        self.audio_ctx = audio_ctx
        self.model_name = model
        params = dict(
            # hide whisper.cpp's model-loading chatter in the console; the windowed exe has no
            # stderr at all (sys.stderr is None), where redirecting would crash
            redirect_whispercpp_logs_to=None if sys.stderr is not None else False,
            context_params={"use_gpu": bool(gpu), "flash_attn": bool(gpu)},
            n_threads=self.threads,
            print_realtime=False,
            print_progress=False,
            print_timestamps=False,
            print_special=False,
            no_context=True,
            suppress_blank=True,
            suppress_nst=True,
            # Re-decode at a higher temperature when the output looks like a loop or
            # garbage (OpenAI's default). Costs time only on hard audio.
            temperature_inc=0.2 if fallback else 0.0,
        )
        if beam and beam > 1:
            params["params_sampling_strategy"] = 1
            params["beam_search"] = {"beam_size": int(beam), "patience": -1.0}
        self.model = Model(str(model_path(model)), **params)
        self.gpu = bool(gpu)
        self.name = f"whisper.cpp {'GPU/Vulkan' if gpu else 'CPU'}"
        # Warm up: the first call compiles the Vulkan pipelines (a few seconds); do it now
        # rather than on the first sentence of the meeting.
        self.model.transcribe(np.zeros(16000 * 5, dtype=np.float32), language="tr")

    def run(self, audio, task="transcribe", language="tr", prompt=None):
        seconds = len(audio) / 16000.0
        params = {"translate": task == "translate", "language": language,
                  "initial_prompt": prompt or "",   # params persist between calls: always set
                  # One utterance = one segment, and a token budget well above fast speech, so a
                  # decoding loop ("...the opposite, the opposite, ...") ends early instead of
                  # running on for seconds.
                  "single_segment": True, "max_tokens": int(20 + 12 * seconds)}
        if self.audio_ctx == "auto":
            params["audio_ctx"] = dynamic_audio_ctx(len(audio) / 16000.0)
        elif self.audio_ctx in (None, "full", 0, "0"):
            params["audio_ctx"] = 0
        else:
            params["audio_ctx"] = int(self.audio_ctx)
        segments = self.model.transcribe(audio, extract_probability=True, **params)
        texts = [s.text.strip() for s in segments if s.text and s.text.strip()]
        probs = [s.probability for s in segments if s.text and s.text.strip() and not math.isnan(s.probability)]
        return Result(" ".join(texts), float(np.mean(probs)) if probs else 1.0)


class FasterWhisperBackend:
    def __init__(self, model="small", threads=None, beam=5, fallback=True, **_):
        from faster_whisper import WhisperModel

        self.threads = threads or min(8, os.cpu_count() or 4)
        self.model_name = model
        self.beam = max(1, int(beam or 1))
        self.fallback = fallback
        self.model = WhisperModel(model, device="cpu", compute_type="int8", cpu_threads=self.threads)
        self.gpu = False
        self.name = "faster-whisper CPU"

    def run(self, audio, task="transcribe", language="tr", prompt=None):
        segments, _ = self.model.transcribe(
            audio,
            language=language,
            task=task,
            beam_size=self.beam,
            temperature=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0] if self.fallback else 0.0,
            vad_filter=False,
            condition_on_previous_text=False,
            initial_prompt=prompt or None,
        )
        segments = [s for s in segments if s.text.strip()]
        conf = float(np.mean([math.exp(s.avg_logprob) for s in segments])) if segments else 1.0
        return Result(" ".join(s.text.strip() for s in segments), conf)


def _kill_with_parent(proc):
    """Put ``proc`` in a job that Windows ends when this process exits, even by a crash, so a
    2 GB server is never left behind. Keep the returned handle open as long as proc should live."""
    import ctypes
    from ctypes import wintypes

    class BASIC(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class EXTENDED(ctypes.Structure):
        _fields_ = [("Basic", BASIC), ("Io", ctypes.c_uint64 * 6), ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t), ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.CreateJobObjectW.restype = wintypes.HANDLE
    k32.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    k32.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    job = k32.CreateJobObjectW(None, None)
    info = EXTENDED()
    info.Basic.LimitFlags = 0x2000                       # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    k32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info))   # extended limits
    k32.AssignProcessToJobObject(job, int(proc._handle))
    return job


class QwenAsrBackend:
    """Qwen3-ASR (Apache-2.0) run by llama.cpp's server on the GPU (Vulkan).

    Measured on this laptop against Whisper large-v3-turbo (2026-10-05, same clips): English
    WER 7.4% vs 8.1-9.5% (VoxPopuli, Earnings-22), ~0.5 s vs ~1.4 s per utterance, ready in
    ~2 s vs 8-18 s, same ~2.1 GB RAM. Spells the names given as ``prompt`` (its context) better
    than Whisper. Turkish is worse (TEDx 18.6% vs 9.0%), so the translator stays on Whisper.
    The 1.7B model was no better on English, twice as slow and needed 4.5 GB.
    """

    LANGUAGES = {"en": "English", "tr": "Turkish", "ar": "Arabic", "de": "German", "fr": "French",
                 "es": "Spanish", "it": "Italian", "ru": "Russian", "fa": "Persian", "hi": "Hindi",
                 "zh": "Chinese", "ja": "Japanese", "nl": "Dutch", "pt": "Portuguese", "ko": "Korean"}
    uses_style_prompt = False     # Whisper copies a sample sentence's punctuation; Qwen punctuates anyway

    def __init__(self, model="qwen3-asr-0.6b", gpu=True, **_):
        import socket

        self.model_name = model
        stem = "Qwen3-ASR-" + model[len("qwen3-asr-"):].upper()   # qwen3-asr-1.7b-tr -> Qwen3-ASR-1.7B-TR
        files = [MODELS_DIR / f"{stem}-Q8_0.gguf", MODELS_DIR / f"mmproj-{stem}-Q8_0.gguf"]
        exe = ROOT / "llama.cpp" / "llama-server.exe"
        for p in files + [exe]:
            if not p.exists():
                raise FileNotFoundError(f"Qwen3-ASR needs {p}")
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.url = f"http://127.0.0.1:{s.getsockname()[1]}"
        self.log_path = ROOT / "llama_server.log"
        self._log = self.log_path.open("wb")
        # one request at a time (-np 1), no prompt cache in RAM: the defaults cost ~1 GB more
        self.proc = subprocess.Popen(
            [str(exe), "-m", str(files[0]), "--mmproj", str(files[1]), "-ngl", "all" if gpu else "0",
             "-c", "2048", "-np", "1", "--cache-ram", "0", "--temp", "0", "--no-webui",
             "--host", "127.0.0.1", "--port", self.url.rsplit(":", 1)[1]],
            stdout=self._log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
        self._job = _kill_with_parent(self.proc)
        self.gpu = bool(gpu)
        self.name = f"Qwen3-ASR llama.cpp {'GPU/Vulkan' if gpu else 'CPU'}"
        deadline = time.monotonic() + 60
        while True:
            try:
                if b"ok" in urllib.request.urlopen(self.url + "/health", timeout=1).read():
                    break
            except OSError:
                pass
            if not self.alive() or time.monotonic() > deadline:
                self.close()
                raise RuntimeError(f"llama-server did not start (see {self.log_path})")
            time.sleep(0.1)
        self.run(np.zeros(16000, dtype=np.float32), language="en")   # warm-up: GPU pipelines

    def alive(self):
        return self.proc.poll() is None

    def run(self, audio, task="transcribe", language="en", prompt=None):
        if task != "transcribe":
            raise ValueError("Qwen3-ASR only transcribes")
        if not self.alive():
            raise RuntimeError(f"llama-server stopped (see {self.log_path})")
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(16000)
            w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
        messages = [{"role": "system", "content": prompt}] if prompt else []
        messages.append({"role": "user", "content": [{"type": "input_audio", "input_audio": {
            "data": base64.b64encode(buf.getvalue()).decode(), "format": "wav"}}]})
        if language in self.LANGUAGES:     # start its answer for it: forces the language
            messages.append({"role": "assistant", "content": f"language {self.LANGUAGES[language]}<asr_text>"})
        body = {"messages": messages, "temperature": 0, "cache_prompt": False, "logprobs": True,
                "max_tokens": int(20 + 12 * len(audio) / 16000.0)}     # a decoding loop ends early
        req = urllib.request.Request(self.url + "/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        choice = json.load(urllib.request.urlopen(req, timeout=120))["choices"][0]
        text = re.sub(r"^.*?<asr_text>", "", choice["message"]["content"], flags=re.S).strip()
        logprobs = [t["logprob"] for t in (choice.get("logprobs") or {}).get("content") or []]
        return Result(text, float(np.mean(np.exp(logprobs))) if logprobs else 1.0)

    def close(self):
        """Stop the server and free its memory."""
        if self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(5)
        if self._job:
            import ctypes

            ctypes.windll.kernel32.CloseHandle(ctypes.c_void_p(self._job))
            self._job = None
        self._log.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_backend(kind="auto", log=print, **kw):
    """kind: auto | whispercpp | faster"""
    if kind in ("auto", "whispercpp"):
        try:
            return WhisperCppBackend(**kw)
        except Exception as e:  # ImportError, missing model or a failed GPU init
            if kind == "whispercpp":
                raise
            log(f"[backend] whisper.cpp unavailable ({type(e).__name__}: {e}); using faster-whisper on CPU")
    kw["model"] = "medium" if kw.get("model", "").startswith("large-v3-turbo") else kw.get("model", "medium")
    return FasterWhisperBackend(**kw)
