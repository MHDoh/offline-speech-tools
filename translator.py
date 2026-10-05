#!/usr/bin/env python
"""Live meeting translator (Turkish -> English), fully local.

Listens to what comes out of your speakers (Google Meet, Teams, Zoom, browser...),
cuts the audio into utterances, transcribes the Turkish with Whisper on the GPU
(whisper.cpp + Vulkan), translates it to English with OPUS-MT on the CPU, and shows
the English in a small always-on-top window. Every line (English + Turkish original)
is appended to transcripts/ as soon as it is translated.

    python translator.py                    # live, GUI
    python translator.py --cli              # live, terminal only
    python translator.py --file talk.mp3    # translate a recording
    python translator.py --list-devices     # pick another audio source
"""
import argparse
import datetime as dt
import queue
import re
import sys
import threading
import time
from pathlib import Path

import numpy as np

from paths import app_root  # noqa: E402

ROOT = app_root()

from audio_capture import TARGET_SR, Capture, list_devices  # noqa: E402
from backends import MODELS_DIR, load_backend  # noqa: E402
from textfilter import clean  # noqa: E402
from vad import Segmenter, SileroVAD  # noqa: E402

TRANSCRIPTS = ROOT / "transcripts"
RES = Path(getattr(sys, "_MEIPASS", ROOT))          # bundled assets when frozen
ERROR_LOG = ROOT / "error.log"
MT_MODEL_DIR = MODELS_DIR / "opus-mt-tc-big-tr-en-ct2"
# Whisper continues in the style of its prompt; this keeps it writing punctuated Turkish.
STYLE_PROMPT = "Merhaba, hoş geldiniz. Bugün neler konuşacağız?"

# Defaults per mode (measured on the Radeon 860M, see README). Live needs ~real-time speed:
# medium with a shortened encoder window takes ~1.5 s per line and copes well with music.
# Files have no deadline: large-v3-turbo on the full 30 s window recognises Turkish best.
PROFILES = {
    "live": {"model": "medium", "audio_ctx": "auto", "beam": 1},
    "file": {"model": "large-v3-turbo", "audio_ctx": "full", "beam": 5},
}


def log_error(where, exc):
    import traceback

    try:
        with ERROR_LOG.open("a", encoding="utf-8") as fh:
            stamp = dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            fh.write(chr(10) + "[" + stamp + "] " + where + chr(10))
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None,
                    help="whisper model: medium (live default) | large-v3-turbo (--file default) | small ...")
    ap.add_argument("--engine", default="auto", choices=["auto", "mt", "whisper"],
                    help="auto/mt = Whisper transcribes, OPUS-MT translates (best); "
                         "whisper = Whisper's own translation (old behaviour)")
    ap.add_argument("--backend", default="auto", choices=["auto", "whispercpp", "faster", "qwen"],
                    help="auto = whisper.cpp on GPU if available, else faster-whisper CPU; "
                         "qwen = Qwen3-ASR via llama.cpp (--model qwen3-asr-0.6b; transcribes only, OPUS-MT translates)")
    ap.add_argument("--cpu", action="store_true", help="force CPU even for whisper.cpp")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--language", default="tr", help="spoken language code (default: tr)")
    ap.add_argument("--beam", type=int, default=None,
                    help="whisper beam size (default: 1 live, 5 for --file; 1 = greedy, faster)")
    ap.add_argument("--no-fallback", action="store_true",
                    help="never re-decode hard audio at higher temperature (more predictable latency)")
    ap.add_argument("--no-context", action="store_true",
                    help="do not give Whisper the previous sentence as context")
    ap.add_argument("--show-original", action="store_true", help="also show the Turkish under each line")
    ap.add_argument("--prompt", default=None, help="vocabulary hint (names, jargon), e.g. \"Barış Bey, Bahar Hanım, Acme\"")
    ap.add_argument("--audio-ctx", default=None,
                    help="whisper.cpp encoder window: auto (live default, ~2x faster), full (--file default, "
                         "most accurate) or a number")
    ap.add_argument("--min-silence", type=float, default=0.6, help="pause (s) that ends an utterance")
    ap.add_argument("--max-segment", type=float, default=12.0, help="cut long speech after this many seconds")
    ap.add_argument("--device", type=int, default=None, help="audio device index (see --list-devices)")
    ap.add_argument("--mic", action="store_true", help="listen to the microphone instead of the speakers")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--file", default=None, help="translate an audio/video file instead of listening live")
    ap.add_argument("--log", default=None, help="transcript file (default: transcripts/meeting_<date>.txt)")
    ap.add_argument("--cli", action="store_true", help="no window, print to the terminal")
    ap.add_argument("--font-size", type=int, default=15)
    ap.add_argument("--seconds", type=float, default=None, help="(cli) stop automatically after N seconds")
    return ap.parse_args(argv)


def _words(s):
    return " ".join(re.sub(r"\W+", " ", s.lower()).split())


def is_echo(text, prompt):
    """True when Whisper just repeated (part of) its prompt, which it does on silence and music."""
    t = _words(text)
    if len(t.split()) >= 2 and t in _words(STYLE_PROMPT):      # e.g. a lone "Hoş geldiniz."
        return True
    return len(t.split()) >= 3 and t in _words(prompt)


def fmt_line(line):
    out = f"[{line['time']}] {line['en']}\n"
    if line["tr"]:
        out += f"    TR: {line['tr']}\n"
    return out


class TranscriptLog:
    """Appends every translated line to the transcript file right away (nothing is lost on a crash)."""

    def __init__(self, path, header):
        self.path = Path(path)
        self.header = header
        self.lock = threading.Lock()
        self.fh = None

    def write(self, line):
        with self.lock:
            if self.fh is None:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                new = not self.path.exists() or self.path.stat().st_size == 0
                self.fh = self.path.open("a", encoding="utf-8")
                if new:
                    self.fh.write(self.header + "\n")
            self.fh.write(fmt_line(line))
            self.fh.flush()

    def close(self):
        with self.lock:
            if self.fh:
                self.fh.close()
                self.fh = None


class Translator:
    """Capture -> segment -> whisper -> OPUS-MT -> lines. UI-agnostic; results go to ``self.events``."""

    def __init__(self, args):
        self.args = args
        self.events = queue.Queue()
        self.seg_q = queue.Queue()
        self.lines = []
        self.paused = False
        self.stop_ev = threading.Event()
        self.backend = None
        self.mt = None
        self.cap = None
        self.segmenter = None
        self.started_at = dt.datetime.now()
        self.prev = None          # (end time, Turkish text) of the last line, for context
        self.log = None

    # -- lifecycle -------------------------------------------------------
    def post(self, kind, **payload):
        self.events.put((kind, payload))

    def load(self):
        a = self.args
        for key, value in PROFILES["file" if a.file else "live"].items():
            if getattr(a, key) is None:
                setattr(a, key, value)
        model = a.model
        if a.backend == "qwen":
            from backends import QwenAsrBackend

            if not model.startswith("qwen3-asr"):
                model = a.model = "qwen3-asr-0.6b"
            a.engine = "mt"                       # Qwen3-ASR only transcribes
            self.post("status", text=f"Loading {model}...")
            self.backend = QwenAsrBackend(model=model, gpu=not a.cpu)
        else:
            if not (MODELS_DIR / f"ggml-{model}.bin").exists() and (MODELS_DIR / "ggml-medium.bin").exists():
                self.post("status", text=f"{model} not found in models/, using medium")
                model, a.model, a.audio_ctx, a.beam = "medium", "medium", "auto", 1
            self.post("status", text=f"Loading Whisper {model}...")
            self.backend = load_backend(a.backend, model=model, threads=a.threads, gpu=not a.cpu,
                                        audio_ctx=a.audio_ctx, beam=a.beam, fallback=not a.no_fallback,
                                        log=lambda m: self.post("status", text=m))
        if a.engine in ("auto", "mt"):
            self.post("status", text="Loading translation model...")
            try:
                from mt import OpusMT, names_from_prompt

                # names from --prompt are kept as names ("Barış Bey" -> "Mr. Barış", not "Mr. Peace")
                self.mt = OpusMT(MT_MODEL_DIR, threads=a.threads, names=names_from_prompt(a.prompt))
            except Exception as e:
                if a.engine == "mt":
                    raise
                log_error("translation model", e)
                self.post("status", text=f"OPUS-MT unavailable ({e}); using Whisper's own translation")
        self.segmenter = Segmenter(SileroVAD(MODELS_DIR / "silero_vad.onnx"),
                                   min_silence=a.min_silence, max_len=a.max_segment)
        self.log = TranscriptLog(self._log_path(), self._log_header())

    def describe(self):
        what = self.args.model if self.args.backend == "qwen" else f"whisper {self.args.model}"
        return what + (f" + {self.mt.name}" if self.mt else " (translate)")

    def _log_path(self):
        if self.args.log:
            return Path(self.args.log)
        stem = Path(self.args.file).stem if self.args.file else "meeting"
        return TRANSCRIPTS / f"{stem}_{self.started_at:%Y-%m-%d_%H%M}.txt"

    def _log_header(self):
        src = f"file {Path(self.args.file).name}" if self.args.file else "live audio"
        return (f"# Meeting transcript, {self.started_at:%Y-%m-%d %H:%M}, {src}\n"
                f"# Turkish speech, machine-translated to English ({self.describe()}).\n"
                f"# Each entry: [time] English translation, then TR: the Turkish transcript.\n"
                f"# Both are automatic and can contain errors; the Turkish is closer to what was said.\n")

    def start_live(self):
        self.cap = Capture(self.args.device, mic=self.args.mic)
        self.cap.start()
        threading.Thread(target=self._segment_loop, daemon=True).start()
        threading.Thread(target=self._infer_loop, daemon=True).start()
        self.post("status", text=f"Listening: {self.cap.info['name']}   |   {self.describe()}")

    def stop(self):
        self.stop_ev.set()
        if self.cap:
            self.cap.stop()
        if self.log:
            self.log.close()

    # -- workers ---------------------------------------------------------
    def _segment_loop(self):
        while not self.stop_ev.is_set():
            try:
                chunk = self.cap.q.get(timeout=0.2)
            except queue.Empty:
                continue
            if self.paused:
                continue
            for seg in self.segmenter.feed(chunk):
                self.seg_q.put((time.time(), seg))
            self.post("level", value=self.segmenter.level, pending=self.seg_q.qsize())

    def _infer_loop(self):
        while not self.stop_ev.is_set():
            try:
                t_end, seg = self.seg_q.get(timeout=0.2)
            except queue.Empty:
                continue
            self._process(seg, t_end)

    def _process(self, seg, t_end, when=None):
        try:
            self._process_inner(seg, t_end, when)
        except Exception as e:
            log_error("inference", e)
            self.post("status", text=f"ERROR: {type(e).__name__}: {e}  (details in error.log)")

    def _prompt(self, t_start):
        """Style anchor + vocabulary hint + the previous sentence.

        Whisper continues in the style of its prompt. A well-punctuated Turkish sentence keeps
        its output punctuated, which matters a lot: OPUS-MT translates unpunctuated Turkish
        far worse. The previous sentence keeps names and terms consistent.
        """
        parts = [STYLE_PROMPT] if getattr(self.backend, "uses_style_prompt", True) else []
        if self.args.prompt:
            parts.append(self.args.prompt)
        if not self.args.no_context and self.prev and t_start - self.prev[0] < 20:
            parts.append(self.prev[1][-200:])
        return " ".join(parts)

    def _process_inner(self, seg, t_end, when=None):
        a = self.args
        dur = len(seg) / TARGET_SR
        t_start = t_end - dur
        t = time.perf_counter()
        if self.mt:
            prompt = self._prompt(t_start)
            r = self.backend.run(seg, task="transcribe", language=a.language, prompt=prompt)
            tr = clean(r.text, r.confidence)
            if tr and is_echo(tr, prompt):
                tr = ""
            en = self.mt.translate(tr) if tr else ""
        else:
            r = self.backend.run(seg, task="translate", language=a.language, prompt=a.prompt)
            en = clean(r.text, r.confidence)
            tr = ""
            if en and a.show_original:
                tr = clean(self.backend.run(seg, task="transcribe", language=a.language, prompt=a.prompt).text)
        cost = time.perf_counter() - t
        if not en.strip():
            return
        # Only well-formed lines become context: Whisper copies the prompt's style, and one
        # unpunctuated line would otherwise make every following line unpunctuated too.
        if tr and re.search(r"[.!?…,]", tr) and not tr[0].islower():
            self.prev = (t_end, tr)
        stamp = when or dt.datetime.fromtimestamp(t_start).strftime("%H:%M:%S")
        line = {"time": stamp, "en": en.strip(), "tr": tr.strip(), "dur": dur, "cost": cost}
        self.lines.append(line)
        self.log.write(line)
        self.post("line", **line, pending=self.seg_q.qsize())

    # -- file mode -------------------------------------------------------
    def run_file(self, path):
        """Translate a whole file; always ends with a "done" event, so the window/CLI never waits forever."""
        try:
            self._run_file(path)
        except Exception as e:
            log_error("file", e)
            self.post("status", text=f"ERROR: {type(e).__name__}: {e}  (details in error.log)")
            raise
        finally:
            self.post("done")

    def _run_file(self, path):
        import shutil
        import subprocess

        if shutil.which("ffmpeg"):
            cmd = ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(TARGET_SR), "-"]
            audio = np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, dtype=np.float32)
        else:                                   # no ffmpeg installed: PyAV, which faster-whisper bundles
            from faster_whisper.audio import decode_audio

            audio = decode_audio(str(path), sampling_rate=TARGET_SR).astype(np.float32)
        total = len(audio) / TARGET_SR
        self.post("status", text=f"Translating {Path(path).name} ({total:.0f}s) with {self.describe()}")
        pos = 0
        step = TARGET_SR // 2
        t_start = time.perf_counter()

        def stamp(end_s, seg):
            s = max(0.0, end_s - len(seg) / TARGET_SR)
            return time.strftime("%H:%M:%S" if total >= 3600 else "%M:%S", time.gmtime(s))

        while pos < len(audio) and not self.stop_ev.is_set():
            chunk = audio[pos:pos + step]
            pos += len(chunk)
            for seg in self.segmenter.feed(chunk):
                end_s = (pos - len(self.segmenter.buf)) / TARGET_SR
                self._process(seg, end_s, when=stamp(end_s, seg))
        for seg in self.segmenter.flush():
            self._process(seg, total, when=stamp(total, seg))
        wall = time.perf_counter() - t_start
        self.post("status", text=f"Done: {total:.0f}s of audio in {wall:.1f}s (RTF {wall / max(total, 1e-6):.2f})"
                                 f"   |   saved to {self.log.path}")

    # -- transcript ------------------------------------------------------
    def transcript_text(self):
        return self._log_header() + "\n" + "".join(fmt_line(line) for line in self.lines)


# -- terminal front-end ----------------------------------------------------
def run_cli(tr, args):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    tr.load()
    if args.file:
        threading.Thread(target=tr.run_file, args=(args.file,), daemon=True).start()
    else:
        tr.start_live()
        print("Listening... Ctrl+C to stop.\n")
    deadline = time.time() + args.seconds if args.seconds else None
    try:
        while True:
            if deadline and time.time() > deadline:
                break
            try:
                kind, p = tr.events.get(timeout=0.5)
            except queue.Empty:
                continue
            if kind == "status":
                print(f"-- {p['text']}", flush=True)
            elif kind == "line":
                print(f"[{p['time']}] {p['en']}    ({p['dur']:.1f}s audio, {p['cost']:.1f}s)", flush=True)
                if p["tr"] and args.show_original:
                    print(f"           TR: {p['tr']}", flush=True)
            elif kind == "done":
                break
    except KeyboardInterrupt:
        pass
    finally:
        tr.stop()
        if tr.lines:
            print(f"\nTranscript: {tr.log.path}")


# -- window front-end ------------------------------------------------------
def run_gui(tr, args):
    import os
    import tkinter as tk
    from tkinter import scrolledtext

    BG, FG, DIM, ACC = "#111418", "#e8eaed", "#8a9099", "#7cc4ff"
    root = tk.Tk()
    root.title("Meeting Translator  (TR -> EN)")
    root.configure(bg=BG)
    root.geometry("560x420+40+40")
    root.attributes("-topmost", True)

    top = tk.Frame(root, bg=BG)
    top.pack(side="top", fill="x", padx=8, pady=(8, 4))
    status = tk.Label(root, text="Starting...", anchor="w", bg=BG, fg=DIM, font=("Segoe UI", 9))
    status.pack(side="bottom", fill="x", padx=10, pady=(2, 6))
    level = tk.Canvas(root, height=3, bg=BG, highlightthickness=0)
    level.pack(side="bottom", fill="x", padx=8)
    level_bar = level.create_rectangle(0, 0, 0, 3, fill=ACC, width=0)
    text = scrolledtext.ScrolledText(root, wrap="word", bg=BG, fg=FG, insertbackground=FG, height=6,
                                     font=("Segoe UI", args.font_size), relief="flat", padx=10, pady=8)
    text.pack(side="top", fill="both", expand=True, padx=8)
    text.tag_configure("time", foreground=DIM, font=("Consolas", max(8, args.font_size - 4)))
    text.tag_configure("en", foreground=FG)
    text.tag_configure("tr", foreground=DIM, font=("Segoe UI", max(8, args.font_size - 3), "italic"),
                       elide=not args.show_original)
    text.tag_configure("sys", foreground=ACC, font=("Segoe UI", max(8, args.font_size - 3)))
    text.configure(state="disabled")

    try:
        root.iconbitmap(str(RES / "assets" / "translator.ico"))
    except Exception:
        pass

    def append(s, tag):
        text.configure(state="normal")
        text.insert("end", s, tag)
        text.see("end")
        text.configure(state="disabled")

    def set_font(delta):
        args.font_size = max(8, args.font_size + delta)
        text.configure(font=("Segoe UI", args.font_size))
        text.tag_configure("tr", font=("Segoe UI", max(8, args.font_size - 3), "italic"))

    def toggle_pause():
        tr.paused = not tr.paused
        pause_btn.configure(text="Resume" if tr.paused else "Pause")
        append("-- paused --\n" if tr.paused else "-- resumed --\n", "sys")

    def clear():
        text.configure(state="normal")
        text.delete("1.0", "end")
        text.configure(state="disabled")

    def open_log():
        if tr.log and tr.log.path.exists():
            os.startfile(tr.log.path)
        else:
            append("-- nothing translated yet; the log is created with the first line\n", "sys")

    def copy_all():
        root.clipboard_clear()
        root.clipboard_append(tr.transcript_text())
        append("-- transcript copied to clipboard\n", "sys")

    def toggle_tr():
        text.tag_configure("tr", elide=not tr_var.get())
        text.see("end")

    def toggle_top():
        root.attributes("-topmost", bool(top_var.get()))

    btn = dict(bg="#22272e", fg=FG, activebackground="#2f3640", activeforeground=FG,
               relief="flat", padx=10, pady=3, font=("Segoe UI", 9))
    chk = dict(bg=BG, fg=DIM, selectcolor=BG, activebackground=BG, activeforeground=FG, font=("Segoe UI", 9))
    pause_btn = tk.Button(top, text="Pause", command=toggle_pause, **btn)
    pause_btn.pack(side="left", padx=(0, 4))
    tk.Button(top, text="Clear", command=clear, **btn).pack(side="left", padx=4)
    tk.Button(top, text="Open log", command=open_log, **btn).pack(side="left", padx=4)
    tk.Button(top, text="Copy", command=copy_all, **btn).pack(side="left", padx=4)
    tk.Button(top, text="A-", command=lambda: set_font(-1), **btn).pack(side="right")
    tk.Button(top, text="A+", command=lambda: set_font(+1), **btn).pack(side="right", padx=4)
    top_var = tk.IntVar(value=1)
    tk.Checkbutton(top, text="On top", variable=top_var, command=toggle_top, **chk).pack(side="right", padx=6)
    tr_var = tk.IntVar(value=1 if args.show_original else 0)
    tk.Checkbutton(top, text="TR", variable=tr_var, command=toggle_tr, **chk).pack(side="right")

    base_status = {"text": ""}

    def pump():
        try:
            while True:
                kind, p = tr.events.get_nowait()
                if kind == "status":
                    base_status["text"] = p["text"]
                    status.configure(text=p["text"])
                elif kind == "line":
                    append(f"{p['time']}  ", "time")
                    append(p["en"] + "\n", "en")
                    if p["tr"]:
                        append("    " + p["tr"] + "\n", "tr")
                    lag = f"   |   last: {p['dur']:.1f}s audio in {p['cost']:.1f}s"
                    if p.get("pending"):
                        lag += f"   |   queued: {p['pending']}"
                    status.configure(text=base_status["text"] + lag)
                elif kind == "level":
                    w = level.winfo_width()
                    level.coords(level_bar, 0, 0, int(w * p["value"]), 3)
        except queue.Empty:
            pass
        root.after(80, pump)

    def boot():
        try:
            tr.load()
            if args.file:
                tr.run_file(args.file)
            else:
                tr.start_live()
        except Exception as e:
            log_error("startup", e)
            tr.post("status", text=f"ERROR: {type(e).__name__}: {e}  (details in error.log)")

    def on_close():
        tr.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    threading.Thread(target=boot, daemon=True).start()
    root.after(80, pump)
    root.mainloop()


def main(argv=None):
    args = parse_args(argv)
    if args.list_devices:
        for i, kind, name, sr, ch in list_devices():
            print(f"{i:3d}  {kind:8s} {sr:6d} Hz  {ch}ch  {name}")
        return
    tr = Translator(args)
    if args.cli:
        run_cli(tr, args)
    else:
        run_gui(tr, args)


if __name__ == "__main__":
    main()
