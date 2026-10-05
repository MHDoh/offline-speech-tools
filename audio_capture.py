"""Audio capture for the meeting translator.

Default source is WASAPI *loopback* of the default speakers, i.e. exactly what you
hear coming out of Google Meet / Teams / Zoom / a browser. No virtual cable needed.
Audio is converted to 16 kHz mono float32, the format Whisper expects.
"""
import queue
from math import gcd

import numpy as np
import pyaudiowpatch as pyaudio
from scipy.signal import resample_poly

TARGET_SR = 16000


def list_devices():
    """Return (index, kind, name, sample_rate, channels) for every WASAPI device."""
    p = pyaudio.PyAudio()
    rows = []
    try:
        wasapi = p.get_host_api_info_by_type(pyaudio.paWASAPI)
        for i in range(p.get_device_count()):
            d = p.get_device_info_by_index(i)
            if d["hostApi"] != wasapi["index"]:
                continue
            if d.get("isLoopbackDevice"):
                kind = "loopback"
            elif d["maxInputChannels"] > 0:
                kind = "mic"
            else:
                kind = "output"
            rows.append((i, kind, d["name"], int(d["defaultSampleRate"]), int(d["maxInputChannels"])))
    finally:
        p.terminate()
    return rows


class Capture:
    """Pushes 16 kHz mono float32 numpy chunks onto ``self.q``."""

    def __init__(self, device_index=None, mic=False, chunk_ms=50):
        self.p = pyaudio.PyAudio()
        self.info = self._pick(device_index, mic)
        self.sr = int(self.info["defaultSampleRate"])
        self.ch = max(1, int(self.info["maxInputChannels"]))
        g = gcd(TARGET_SR, self.sr)
        self.up, self.down = TARGET_SR // g, self.sr // g
        self.chunk = int(self.sr * chunk_ms / 1000)
        self.q = queue.Queue()
        self._stream = None

    def _pick(self, idx, mic):
        if idx is not None:
            return self.p.get_device_info_by_index(idx)
        wasapi = self.p.get_host_api_info_by_type(pyaudio.paWASAPI)
        if mic:
            return self.p.get_device_info_by_index(wasapi["defaultInputDevice"])
        default_out = self.p.get_device_info_by_index(wasapi["defaultOutputDevice"])
        for lb in self.p.get_loopback_device_info_generator():
            if default_out["name"] in lb["name"]:
                return lb
        raise RuntimeError(
            "No loopback device found for the default speakers. "
            "Run with --list-devices and pick one with --device N."
        )

    def start(self):
        self._stream = self.p.open(
            format=pyaudio.paFloat32,
            channels=self.ch,
            rate=self.sr,
            input=True,
            input_device_index=self.info["index"],
            frames_per_buffer=self.chunk,
            stream_callback=self._cb,
        )
        self._stream.start_stream()

    def _cb(self, in_data, frame_count, time_info, status):
        x = np.frombuffer(in_data, dtype=np.float32)
        if self.ch > 1:
            x = x.reshape(-1, self.ch).mean(axis=1)
        if self.sr != TARGET_SR:
            x = resample_poly(x, self.up, self.down)
        self.q.put(np.ascontiguousarray(x, dtype=np.float32))
        return (None, pyaudio.paContinue)

    def stop(self):
        if self._stream is not None:
            try:
                self._stream.stop_stream()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        self.p.terminate()
