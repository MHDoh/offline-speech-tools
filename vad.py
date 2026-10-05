"""Voice activity detection (Silero VAD via onnxruntime) and utterance segmentation.

The segmenter turns a continuous 16 kHz stream into utterances: it starts a segment
when speech begins and closes it after a pause (``min_silence``), or cuts it when it
reaches ``max_len`` seconds. Each closed segment is sent to Whisper, so latency is
roughly: pause length + inference time.

Things that matter for translation quality:
  * Long speech is cut at the quietest moment of the last few seconds (usually a
    breath between phrases), not blindly at ``max_len`` in the middle of a word.
  * Very short pieces need a longer pause before they are closed, so a sentence
    with a dramatic pause is not translated as two fragments.
  * Trailing silence is trimmed; Whisper tends to invent text ("Thank you.") in it.
"""
from collections import deque

import numpy as np
import onnxruntime as ort

SR = 16000
FRAME = 512      # samples per VAD frame (32 ms)
CONTEXT = 64     # Silero v5 wants the last 64 samples of the previous frame too


def frames(seconds):
    return max(1, int(round(seconds * SR / FRAME)))


class SileroVAD:
    def __init__(self, model_path):
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(model_path), so, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.ctx = np.zeros((1, CONTEXT), dtype=np.float32)

    def prob(self, frame):
        """Speech probability for one 512-sample float32 frame."""
        x = np.concatenate([self.ctx, frame[None, :]], axis=1).astype(np.float32)
        out, self.state = self.sess.run(
            None, {"input": x, "state": self.state, "sr": np.array(SR, dtype=np.int64)}
        )
        self.ctx = x[:, -CONTEXT:]
        return float(out[0, 0])


class Segmenter:
    def __init__(self, vad, start_th=0.5, end_th=0.35, min_silence=0.6, max_len=12.0,
                 min_speech=0.4, pre_roll=0.3, short_len=2.0, short_silence=1.0,
                 tail=0.3, cut_window=3.0):
        self.vad = vad
        self.start_th, self.end_th = start_th, end_th
        self.min_silence_frames = frames(min_silence)
        # segments shorter than short_len wait for a longer pause before being closed
        self.short_frames = frames(short_len)
        self.short_silence_frames = max(self.min_silence_frames, frames(short_silence))
        self.max_frames = frames(max_len)
        self.min_speech_frames = frames(min_speech)
        self.tail_frames = frames(tail)
        self.cut_window = frames(cut_window)
        self.pre = deque(maxlen=frames(pre_roll))
        self.buf = np.zeros(0, dtype=np.float32)
        self.seg = []      # frames of the open segment
        self.probs = []    # VAD probability per frame of the open segment
        self.in_speech = False
        self.silence = 0
        self.level = 0.0   # last VAD probability, for the UI meter

    def _speech_frames(self, probs):
        return sum(1 for p in probs if p >= self.end_th)

    def _emit(self, out, n=None, trim_tail=False):
        """Close the first ``n`` frames of the open segment (all by default)."""
        n = len(self.seg) if n is None else n
        seg, probs = self.seg[:n], self.probs[:n]
        self.seg, self.probs = self.seg[n:], self.probs[n:]
        if trim_tail:
            last = max((i for i, p in enumerate(probs) if p >= self.end_th), default=len(probs) - 1)
            keep = min(len(seg), last + 1 + self.tail_frames)
            seg, probs = seg[:keep], probs[:keep]
        if seg and self._speech_frames(probs) >= self.min_speech_frames:
            out.append(np.concatenate(seg))
        self.silence = 0

    def _soft_cut(self):
        """Index to cut a too-long segment at: the quietest frame in the last few seconds."""
        lo = max(len(self.probs) // 2, len(self.probs) - self.cut_window)
        window = np.convolve(self.probs, np.ones(3) / 3, mode="same")[lo:]
        return lo + int(np.argmin(window)) + 1

    def feed(self, audio):
        """Feed float32 16 kHz samples; returns a list of finished utterances."""
        out = []
        self.buf = np.concatenate([self.buf, audio]) if len(self.buf) else audio
        n = len(self.buf) // FRAME
        for i in range(n):
            f = self.buf[i * FRAME:(i + 1) * FRAME]
            p = self.vad.prob(f)
            self.level = p
            if not self.in_speech:
                if p >= self.start_th:
                    self.in_speech = True
                    self.seg = list(self.pre) + [f]
                    self.probs = [0.0] * len(self.pre) + [p]
                    self.pre.clear()
                    self.silence = 0
                else:
                    self.pre.append(f)
                continue
            self.seg.append(f)
            self.probs.append(p)
            self.silence = self.silence + 1 if p < self.end_th else 0
            spoken = len(self.seg) - self.silence
            need = self.short_silence_frames if spoken < self.short_frames else self.min_silence_frames
            if self.silence >= need:
                # Pause detected: close the utterance, go back to idle.
                self._emit(out, trim_tail=True)
                self.in_speech = False
                self.seg, self.probs = [], []
            elif len(self.seg) >= self.max_frames:
                # Speaker keeps going: cut at the quietest recent moment, keep the rest.
                self._emit(out, self._soft_cut())
        self.buf = self.buf[n * FRAME:]
        return out

    def flush(self):
        out = []
        if self.in_speech:
            self._emit(out, trim_tail=True)
            self.in_speech = False
            self.seg, self.probs = [], []
        return out
