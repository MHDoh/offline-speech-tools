r"""
Transcribe - drag & drop speech-to-text. Runs fully offline (faster-whisper).

Drop audio/video files or folders onto the window (or onto Transcribe.exe itself).
A .txt transcript is written next to each file.
"""
import json
import os
import queue
import subprocess
import sys
import threading
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")  # plain HTTP model downloads are more reliable

APP_NAME = "Transcribe"
APP_DIR = Path(os.environ.get("LOCALAPPDATA", Path.home())) / APP_NAME
LOG_PATH = APP_DIR / "transcribe.log"
SETTINGS_PATH = APP_DIR / "settings.json"
ASSETS = Path(getattr(sys, "_MEIPASS", Path(__file__).parent)) / "assets"

AUDIO_EXTS = {
    ".ogg", ".oga", ".opus", ".mp3", ".m4a", ".aac", ".wav", ".flac", ".wma",
    ".mp4", ".mkv", ".mov", ".webm", ".avi", ".mpeg", ".mpg", ".3gp", ".amr",
}
MODELS = ["small", "medium", "large-v3"]
MODEL_HINT = {"small": "~460 MB, fastest", "medium": "~1.5 GB, best balance", "large-v3": "~3 GB, slow on CPU"}
DEFAULT_PROMPT = ""

# All 100 languages Whisper supports (code -> name). Shown in the dropdown, stored as the code.
LANGUAGES = {
    "af": "Afrikaans", "am": "Amharic", "ar": "Arabic", "as": "Assamese", "az": "Azerbaijani", "ba": "Bashkir",
    "be": "Belarusian", "bg": "Bulgarian", "bn": "Bengali", "bo": "Tibetan", "br": "Breton", "bs": "Bosnian",
    "ca": "Catalan", "cs": "Czech", "cy": "Welsh", "da": "Danish", "de": "German", "el": "Greek", "en": "English",
    "es": "Spanish", "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish", "fo": "Faroese",
    "fr": "French", "gl": "Galician", "gu": "Gujarati", "ha": "Hausa", "haw": "Hawaiian", "he": "Hebrew",
    "hi": "Hindi", "hr": "Croatian", "ht": "Haitian Creole", "hu": "Hungarian", "hy": "Armenian",
    "id": "Indonesian", "is": "Icelandic", "it": "Italian", "ja": "Japanese", "jw": "Javanese", "ka": "Georgian",
    "kk": "Kazakh", "km": "Khmer", "kn": "Kannada", "ko": "Korean", "la": "Latin", "lb": "Luxembourgish",
    "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian", "lv": "Latvian", "mg": "Malagasy", "mi": "Maori",
    "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian", "mr": "Marathi", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "no": "Norwegian", "oc": "Occitan",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese", "ro": "Romanian", "ru": "Russian",
    "sa": "Sanskrit", "sd": "Sindhi", "si": "Sinhala", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
    "so": "Somali", "sq": "Albanian", "sr": "Serbian", "su": "Sundanese", "sv": "Swedish", "sw": "Swahili",
    "ta": "Tamil", "te": "Telugu", "tg": "Tajik", "th": "Thai", "tk": "Turkmen", "tl": "Tagalog", "tr": "Turkish",
    "tt": "Tatar", "uk": "Ukrainian", "ur": "Urdu", "uz": "Uzbek", "vi": "Vietnamese", "yi": "Yiddish",
    "yo": "Yoruba", "yue": "Cantonese", "zh": "Chinese",
}
AUTO_LABEL = "Auto-detect"
# Frequently used ones first, then everything alphabetically
TOP_LANGS = ["en", "ar", "fr", "es", "de", "hi", "ur", "tr", "zh", "ja", "ru", "pt", "it"]
LANG_CHOICES = [AUTO_LABEL] + [LANGUAGES[c] for c in TOP_LANGS] + sorted(
    n for c, n in LANGUAGES.items() if c not in TOP_LANGS
)
NAME_TO_CODE = {n: c for c, n in LANGUAGES.items()}


def lang_code_to_label(code):
    return LANGUAGES.get((code or "").strip().lower(), AUTO_LABEL)


def lang_label_to_code(label):
    label = (label or "").strip()
    if label in NAME_TO_CODE:
        return NAME_TO_CODE[label]
    if label.lower() in LANGUAGES:  # someone typed a code
        return label.lower()
    return "auto"

ACCENT = "#3B82F6"
ZONE_BG = ("#F3F6FB", "#1C2430")
ZONE_BORDER = ("#C9D3E3", "#34405A")
ZONE_ACTIVE_BG = ("#E6EFFF", "#20304A")
MUTED = ("#6B7280", "#9AA3B2")


def _ensure_streams():
    """In a windowed (no-console) build stdout/stderr may be None; give libraries a file to write to."""
    if sys.stdout is None or sys.stderr is None:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        f = open(LOG_PATH, "a", encoding="utf-8", buffering=1)
        if sys.stdout is None:
            sys.stdout = f
        if sys.stderr is None:
            sys.stderr = f


_ensure_streams()

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox  # noqa: E402

import customtkinter as ctk  # noqa: E402
from PIL import Image  # noqa: E402

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD  # noqa: E402
    HAVE_DND = True
except Exception:  # fallback if the dnd binaries are missing
    HAVE_DND = False


# ----------------------------------------------------------------------------- helpers
def fmt_ts(seconds: float, srt: bool = False) -> str:
    ms = int(round(seconds * 1000))
    h, rem = divmod(ms, 3_600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    if srt:
        return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def expand_inputs(paths):
    files, seen = [], set()
    for p in paths:
        p = Path(p)
        if p.is_dir():
            cands = sorted(f for f in p.iterdir() if f.suffix.lower() in AUDIO_EXTS)
        else:
            cands = [p]
        for f in cands:
            if not f.is_file():
                continue
            r = f.resolve()
            if r not in seen:
                seen.add(r)
                files.append(f)
    return files


def model_is_cached(name: str) -> bool:
    hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    return any(
        d.name.endswith(f"faster-whisper-{name}") and (d / "snapshots").exists()
        for d in hub.glob("models--*")
    )


def load_settings():
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def save_settings(d):
    try:
        APP_DIR.mkdir(parents=True, exist_ok=True)
        SETTINGS_PATH.write_text(json.dumps(d, indent=2), encoding="utf-8")
    except Exception:
        pass


# ----------------------------------------------------------------------------- worker
class Worker(threading.Thread):
    """Background thread: loads models on demand (kept in memory) and transcribes queued jobs."""

    def __init__(self, out_q):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.out = out_q
        self.models = {}

    def post(self, kind, payload=None):
        self.out.put((kind, payload))

    def get_model(self, name):
        if name in self.models:
            return self.models[name]
        hint = MODEL_HINT.get(name, "")

        # Preferred: whisper.cpp on the AMD GPU (Vulkan). Falls back to faster-whisper below.
        try:
            import gpu_backend
        except Exception:
            gpu_backend = None
        if gpu_backend is not None and gpu_backend.available():
            if gpu_backend.model_is_cached(name):
                self.post("status", f"Loading {name} model (GPU)…")
            else:
                self.post("status", f"Downloading {name} model - first time only ({hint})…")
                self.post("sys", f"Downloading the {name} model (first time only, {hint}). Please wait…")
            t0 = time.time()
            try:
                m = gpu_backend.WhisperCppModel(name)
                self.post("sys", f"{name} model ready (GPU / whisper.cpp Vulkan, {time.time() - t0:.0f}s)")
                self.models[name] = m
                return m
            except Exception as e:  # no Vulkan device, bad model file, ... -> CPU path
                self.post("sys", f"GPU backend unavailable ({e}); using CPU")

        from faster_whisper import WhisperModel
        import ctranslate2

        if ctranslate2.get_cuda_device_count() > 0:
            device, compute = "cuda", "float16"
        else:
            device, compute = "cpu", "int8"
        if model_is_cached(name):
            self.post("status", f"Loading {name} model…")
        else:
            self.post("status", f"Downloading {name} model - first time only ({hint})…")
            self.post("sys", f"Downloading the {name} model (first time only, {hint}). Please wait…")
        t0 = time.time()
        m = WhisperModel(name, device=device, compute_type=compute)
        self.post("sys", f"{name} model ready ({device}, {time.time() - t0:.0f}s)")
        self.models[name] = m
        return m

    def run(self):
        while True:
            files, opts = self.jobs.get()
            self.post("busy", True)
            try:
                self.process(files, opts)
            except Exception as e:  # keep the app alive on any failure
                self.post("sys", f"Error: {e}")
                self.post("status", f"Error: {e}")
            self.post("busy", False)

    def process(self, files, opts):
        model = self.get_model(opts["model"])
        lang = opts["language"].strip().lower()
        language = None if lang in ("", "auto") else lang
        results = []
        for i, f in enumerate(files, 1):
            self.post("status", f"Transcribing {i} of {len(files)}: {f.name}")
            self.post("file", f.name)
            t0 = time.time()
            segments, info = model.transcribe(
                str(f),
                language=language,
                beam_size=5,
                initial_prompt=opts["prompt"].strip() or None,
                vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
            )
            seg_list, lines = [], []
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                seg_list.append({"start": seg.start, "end": seg.end, "text": text})
                line = f"[{fmt_ts(seg.start)}] {text}" if opts["timestamps"] else text
                lines.append(line)
                self.post("seg", line)

            body = ("\n".join(lines) if opts["timestamps"] else " ".join(lines)) + "\n"
            txt_path = f.with_suffix(".txt")
            txt_path.write_text(body, encoding="utf-8")
            written = [txt_path.name]
            if opts["srt"]:
                srt_path = f.with_suffix(".srt")
                with srt_path.open("w", encoding="utf-8") as fh:
                    for n, s in enumerate(seg_list, 1):
                        fh.write(f"{n}\n{fmt_ts(s['start'], True)} --> {fmt_ts(s['end'], True)}\n{s['text']}\n\n")
                written.append(srt_path.name)
            results.append((f, body))
            self.post(
                "meta",
                f"Saved {' and '.join(written)}  ·  {info.duration:.0f}s of audio in {time.time() - t0:.1f}s",
            )
        self.post("done", results)


# ----------------------------------------------------------------------------- GUI
if HAVE_DND:
    class Root(ctk.CTk, TkinterDnD.DnDWrapper):
        def __init__(self, *a, **k):
            super().__init__(*a, **k)
            self.TkdndVersion = TkinterDnD._require(self)
else:
    Root = ctk.CTk


class App:
    def __init__(self, root, initial_files):
        self.root = root
        self.q = queue.Queue()
        self.worker = Worker(self.q)
        self.worker.start()
        self.results = []
        self.last_folder = None
        self.exit_when_done = False
        self.busy = False

        s = load_settings()
        self.var_model = ctk.StringVar(value=s.get("model", "medium"))
        self.var_lang = ctk.StringVar(value=lang_code_to_label(s.get("language", "en")))
        self.var_ts = ctk.BooleanVar(value=s.get("timestamps", False))
        self.var_srt = ctk.BooleanVar(value=s.get("srt", False))
        self.var_prompt = ctk.StringVar(value=s.get("prompt", DEFAULT_PROMPT))

        root.title(APP_NAME)
        root.geometry("900x700")
        root.minsize(680, 520)
        self.fonts = {
            "title": ctk.CTkFont("Segoe UI", 24, "bold"),
            "sub": ctk.CTkFont("Segoe UI", 13),
            "zone": ctk.CTkFont("Segoe UI", 17, "bold"),
            "body": ctk.CTkFont("Segoe UI", 13),
            "label": ctk.CTkFont("Segoe UI", 12, "bold"),
            "text": ctk.CTkFont("Segoe UI", 14),
        }
        self.build()
        self.set_icon()
        root.after(100, self.poll)
        root.bind("<Control-o>", lambda e: self.browse())
        if initial_files:
            root.after(300, lambda: self.enqueue(initial_files))

    # -- layout
    def build(self):
        r = self.root
        r.grid_columnconfigure(0, weight=1)
        r.grid_rowconfigure(3, weight=1)
        PAD = 22

        # header ---------------------------------------------------------------
        header = ctk.CTkFrame(r, fg_color="transparent")
        header.grid(row=0, column=0, sticky="ew", padx=PAD, pady=(18, 10))
        logo = self.image("logo.png", 40)
        if logo:
            ctk.CTkLabel(header, image=logo, text="").pack(side="left", padx=(0, 12))
        titles = ctk.CTkFrame(header, fg_color="transparent")
        titles.pack(side="left")
        ctk.CTkLabel(titles, text=APP_NAME, font=self.fonts["title"], anchor="w").pack(anchor="w")
        ctk.CTkLabel(
            titles, text="Offline speech-to-text · drop audio in, get text out",
            font=self.fonts["sub"], text_color=MUTED, anchor="w",
        ).pack(anchor="w")
        self.theme_btn = ctk.CTkSegmentedButton(
            header, values=["Light", "Dark"], width=140, command=self.set_theme, font=self.fonts["sub"],
        )
        self.theme_btn.set("Dark" if ctk.get_appearance_mode() == "Dark" else "Light")
        self.theme_btn.pack(side="right")

        # drop zone -------------------------------------------------------------
        self.zone = ctk.CTkFrame(r, corner_radius=16, border_width=2, fg_color=ZONE_BG, border_color=ZONE_BORDER)
        self.zone.grid(row=1, column=0, sticky="ew", padx=PAD, pady=(4, 12))
        inner = ctk.CTkFrame(self.zone, fg_color="transparent")
        inner.pack(pady=26)
        zicon = self.image("logo.png", 52)
        if zicon:
            ctk.CTkLabel(inner, image=zicon, text="").pack(pady=(0, 8))
        self.zone_title = ctk.CTkLabel(inner, text="Drop audio or video files here", font=self.fonts["zone"])
        self.zone_title.pack()
        self.zone_sub = ctk.CTkLabel(
            inner, text="or click to browse  ·  folders work too  ·  transcripts are saved next to the originals",
            font=self.fonts["body"], text_color=MUTED,
        )
        self.zone_sub.pack(pady=(2, 0))
        for w in self.walk(self.zone):
            w.configure(cursor="hand2")
            w.bind("<Button-1>", lambda e: self.browse())

        # options card ----------------------------------------------------------
        card = ctk.CTkFrame(r, corner_radius=14)
        card.grid(row=2, column=0, sticky="ew", padx=PAD, pady=(0, 12))
        card.grid_columnconfigure(7, weight=1)
        px = dict(padx=(0, 10), pady=(14, 6))
        ctk.CTkLabel(card, text="Model", font=self.fonts["label"]).grid(row=0, column=0, sticky="w", padx=(16, 8), pady=(14, 6))
        self.model_btn = ctk.CTkSegmentedButton(card, values=MODELS, variable=self.var_model, font=self.fonts["body"])
        self.model_btn.grid(row=0, column=1, sticky="w", **px)
        ctk.CTkLabel(card, text="Language", font=self.fonts["label"]).grid(row=0, column=2, sticky="w", padx=(12, 8), pady=(14, 6))
        self.lang_menu = ctk.CTkComboBox(
            card, values=LANG_CHOICES, variable=self.var_lang, width=170, state="readonly",
            font=self.fonts["body"], dropdown_font=self.fonts["body"],
        )
        self.lang_menu.grid(row=0, column=3, sticky="w", **px)
        ctk.CTkSwitch(card, text="Timestamps", variable=self.var_ts, font=self.fonts["body"]).grid(row=0, column=4, sticky="w", padx=(12, 14), pady=(14, 6))
        ctk.CTkSwitch(card, text="Subtitles (.srt)", variable=self.var_srt, font=self.fonts["body"]).grid(row=0, column=5, sticky="w", padx=(0, 10), pady=(14, 6))

        ctk.CTkLabel(card, text="Vocabulary hints", font=self.fonts["label"]).grid(row=1, column=0, sticky="w", padx=(16, 8), pady=(4, 14))
        self.prompt_entry = ctk.CTkEntry(
            card, textvariable=self.var_prompt, font=self.fonts["body"],
            placeholder_text="Names and jargon the speaker uses, e.g. Kubernetes, Dr. Yılmaz, Acme",
        )
        self.prompt_entry.grid(row=1, column=1, columnspan=7, sticky="ew", padx=(0, 16), pady=(4, 14))

        # transcript ------------------------------------------------------------
        self.text = ctk.CTkTextbox(r, corner_radius=14, font=self.fonts["text"], wrap="word", state="disabled", padx=8, pady=8)
        self.text.grid(row=3, column=0, sticky="nsew", padx=PAD, pady=(0, 12))
        self.style_tags()
        self.sys("Ready. Drop files above to get started.")

        # bottom bar ------------------------------------------------------------
        bar = ctk.CTkFrame(r, fg_color="transparent")
        bar.grid(row=4, column=0, sticky="ew", padx=PAD, pady=(0, 18))
        bar.grid_columnconfigure(0, weight=1)
        self.status = ctk.CTkLabel(bar, text="Ready", font=self.fonts["body"], text_color=MUTED, anchor="w")
        self.status.grid(row=0, column=0, sticky="ew")
        self.prog = ctk.CTkProgressBar(bar, mode="indeterminate", width=150, height=8)
        self.prog.grid(row=0, column=1, padx=(10, 14))
        self.prog.grid_remove()
        self.btn_clear = ctk.CTkButton(
            bar, text="Clear", width=70, fg_color="transparent", border_width=1, text_color=MUTED,
            border_color=ZONE_BORDER, hover_color=ZONE_ACTIVE_BG, font=self.fonts["body"], command=self.clear,
        )
        self.btn_clear.grid(row=0, column=2, padx=(0, 8))
        self.btn_copy = ctk.CTkButton(
            bar, text="Copy transcript", width=130, fg_color="transparent", border_width=1,
            border_color=ACCENT, text_color=(ACCENT, "#8AB4FF"), hover_color=ZONE_ACTIVE_BG,
            font=self.fonts["body"], command=self.copy_transcript, state="disabled",
        )
        self.btn_copy.grid(row=0, column=3, padx=(0, 8))
        self.btn_open = ctk.CTkButton(bar, text="Open folder", width=110, font=self.fonts["body"], command=self.open_folder, state="disabled")
        self.btn_open.grid(row=0, column=4)

        # drag & drop anywhere on the window ------------------------------------
        if HAVE_DND:
            for w in [r] + self.walk(r):
                try:
                    w.drop_target_register(DND_FILES)
                    w.dnd_bind("<<Drop>>", self.on_drop)
                    w.dnd_bind("<<DragEnter>>", lambda e: self.highlight(True))
                    w.dnd_bind("<<DragLeave>>", lambda e: self.highlight(False))
                except Exception:
                    pass
        else:
            self.zone_title.configure(text="Click to choose audio or video files")
            self.zone_sub.configure(text="drag & drop is unavailable on this system")

    def walk(self, widget):
        out = []
        for c in widget.winfo_children():
            out.append(c)
            out.extend(self.walk(c))
        return out

    def image(self, name, size):
        p = ASSETS / name
        if not p.exists():
            return None
        return ctk.CTkImage(Image.open(p), size=(size, size))

    def set_icon(self):
        ico = ASSETS / "transcribe.ico"
        if ico.exists():
            # CTk resets the icon shortly after startup, so apply ours a moment later
            self.root.after(250, lambda: self.root.iconbitmap(str(ico)))

    def style_tags(self):
        dark = ctk.get_appearance_mode() == "Dark"
        grey = "#8B93A3" if dark else "#7A8290"
        # CTkTextbox refuses fonts on tags, so style the underlying Tk text widget with DPI scaling applied by hand
        tb = self.text._textbox
        sc = ctk.ScalingTracker.get_widget_scaling(self.text)
        f = lambda size, *style: ("Segoe UI", -int(round(size * sc * 4 / 3)), *style)  # negative = pixels
        tb.tag_config("file", font=f(14, "bold"), spacing1=14, spacing3=4)
        tb.tag_config("seg", spacing1=2, spacing3=2)
        tb.tag_config("meta", foreground=grey, font=f(12), spacing1=4, spacing3=6)
        tb.tag_config("sys", foreground=grey, font=f(12, "italic"), spacing3=4)

    def set_theme(self, value):
        ctk.set_appearance_mode(value.lower())
        self.root.after(50, self.style_tags)

    # -- text area helpers
    def write(self, text, tag):
        self.text.configure(state="normal")
        self.text.insert("end", text + "\n", tag)
        self.text.see("end")
        self.text.configure(state="disabled")

    def sys(self, text):
        self.write(text, "sys")

    def clear(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        self.results = []
        self.btn_copy.configure(state="disabled")
        self.sys("Ready. Drop files above to get started.")

    def highlight(self, on):
        if self.busy:
            return
        self.zone.configure(border_color=ACCENT if on else ZONE_BORDER, fg_color=ZONE_ACTIVE_BG if on else ZONE_BG)

    # -- actions
    def browse(self):
        exts = " ".join(f"*{e}" for e in sorted(AUDIO_EXTS))
        paths = filedialog.askopenfilenames(
            title="Choose audio or video files",
            filetypes=[("Audio / video", exts), ("All files", "*.*")],
        )
        if paths:
            self.enqueue(paths)

    def on_drop(self, event):
        self.highlight(False)
        self.enqueue(self.root.tk.splitlist(event.data))

    def enqueue(self, paths):
        files = expand_inputs(paths)
        if not files:
            messagebox.showinfo(APP_NAME, "No audio or video files found in what you dropped.")
            return
        opts = {
            "model": self.var_model.get(),
            "language": lang_label_to_code(self.var_lang.get()),
            "timestamps": self.var_ts.get(),
            "srt": self.var_srt.get(),
            "prompt": self.var_prompt.get(),
        }
        save_settings(opts)
        self.last_folder = files[0].parent
        self.btn_open.configure(state="normal")
        n = len(files)
        self.sys(f"Queued {n} file{'s' if n != 1 else ''} · {opts['model']} model")
        self.worker.jobs.put((files, opts))

    def copy_transcript(self):
        text = "\n\n".join(f"{f.name}\n{body.strip()}" for f, body in self.results)
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status.configure(text="Transcript copied to clipboard")

    def open_folder(self):
        if self.last_folder:
            subprocess.Popen(["explorer", str(self.last_folder)])

    def set_busy(self, on):
        self.busy = on
        if on:
            self.prog.grid()
            self.prog.start()
            self.zone.configure(border_color=ZONE_BORDER, fg_color=ZONE_BG)
            self.zone_title.configure(text="Working…")
            self.zone_sub.configure(text="you can keep dropping files - they will be queued")
        else:
            self.prog.stop()
            self.prog.grid_remove()
            self.zone_title.configure(text="Drop audio or video files here")
            self.zone_sub.configure(text="or click to browse  ·  folders work too  ·  transcripts are saved next to the originals")

    def poll(self):
        try:
            while True:
                kind, payload = self.q.get_nowait()
                if kind in ("seg", "meta", "sys", "file"):
                    self.write(payload, kind)
                elif kind == "status":
                    self.status.configure(text=payload)
                elif kind == "busy":
                    self.set_busy(payload)
                elif kind == "done":
                    self.results.extend(payload)
                    self.btn_copy.configure(state="normal")
                    n = len(payload)
                    self.status.configure(text=f"Done · {n} file{'s' if n != 1 else ''} transcribed")
                    if self.exit_when_done:
                        self.root.after(200, self.root.destroy)
        except queue.Empty:
            pass
        self.root.after(100, self.poll)


def main():
    args = sys.argv[1:]
    exit_when_done = "--exit-when-done" in args  # used for automated testing
    files = [a for a in args if not a.startswith("--")]
    theme = next((a.split("=", 1)[1] for a in args if a.startswith("--theme=")), "system")
    ctk.set_appearance_mode(theme)
    ctk.set_default_color_theme("blue")
    root = Root()
    app = App(root, files)
    app.exit_when_done = exit_when_done
    root.mainloop()


if __name__ == "__main__":
    main()
