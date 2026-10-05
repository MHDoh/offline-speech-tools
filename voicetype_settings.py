"""VoiceType settings window, opened from the tray icon.

It writes voicetype.json; the running VoiceType notices the change and applies it right
away (the hotkey at once, a new model on the next key press).
"""
import ctypes
import json
import os
import re
import tkinter as tk
from tkinter import messagebox, ttk

from paths import app_root

ROOT = app_root()
CONFIG = ROOT / "voicetype.json"
MODELS_DIR = ROOT / "models"
HISTORY = ROOT / "transcripts"

# Spare keys only: a typing key would stop working as itself.
KEYS = ([("Copilot key", 0x86)] + [(f"F{n}", 0x6F + n) for n in range(1, 13)]
        + [("Insert", 0x2D), ("Menu key", 0x5D)])
MODIFIERS = [("Alt", "alt"), ("Ctrl", "ctrl"), ("Nothing (the key alone)", "none")]
LANGUAGES = [("English", "en"), ("Turkish", "tr"), ("Detect automatically", "auto"), ("Arabic", "ar"),
             ("German", "de"), ("French", "fr"), ("Spanish", "es"), ("Italian", "it"), ("Russian", "ru"),
             ("Persian", "fa"), ("Urdu", "ur"), ("Hindi", "hi"), ("Chinese", "zh"), ("Japanese", "ja")]
MODEL_NAMES = {"qwen3-asr-0.6b": "Qwen3-ASR 0.6B (fastest, best English)",
               "large-v3-turbo": "Whisper Large v3 Turbo (best for Turkish)", "medium": "Whisper Medium",
               "medium-q5_0": "Whisper Medium, compressed", "small": "Whisper Small (least accurate)"}
SILENCE = [0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
UNLOAD = [("1 minute", 1), ("5 minutes", 5), ("10 minutes", 10), ("30 minutes", 30), ("Never (keep it loaded)", 0)]

_window = None


def key_name(vk):
    for name, code in KEYS:
        if code == vk:
            return name
    buf = ctypes.create_unicode_buffer(64)
    scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    if scan and ctypes.windll.user32.GetKeyNameTextW(scan << 16, buf, 64):
        return buf.value
    return f"key {vk}"


def hotkey_text(cfg):
    """'Alt+Copilot key', or just the key's name when no modifier is used."""
    mod = str(cfg.get("modifier", "alt")).lower()
    key = key_name(int(cfg["hotkey_vk"]))
    return key if mod == "none" else f"{mod.capitalize()}+{key}"


def open_settings(master, cfg, icon=None):
    """Show the settings window (or bring it to the front if it is already open)."""
    global _window
    if _window is None or not _window.win.winfo_exists():
        _window = SettingsWindow(master, cfg, icon)
    _window.show()


class SettingsWindow:
    def __init__(self, master, cfg, icon=None):
        self.cfg = cfg
        w = self.win = tk.Toplevel(master)
        w.withdraw()
        w.title("VoiceType settings")
        w.resizable(False, False)
        if icon is not None:
            from PIL import ImageTk

            self._icon = ImageTk.PhotoImage(icon)     # keep a reference, or Tk shows a blank icon
            w.iconphoto(False, self._icon)
        w.bind("<Escape>", lambda _e: w.destroy())
        self.body = ttk.Frame(w, padding=(20, 16, 20, 16))
        self.body.pack(fill="both", expand=True)
        self.body.columnconfigure(1, weight=1)
        self.row = 0

        self.summary = ttk.Label(self.body, foreground="#444")
        self.summary.grid(row=self.row, column=0, columnspan=2, sticky="w")
        self.row += 1

        self._heading("Hotkey")
        self.key = self._combo(KEYS, int(cfg["hotkey_vk"]))
        self._field("Key", self.key)
        self.modifier = self._combo(MODIFIERS, str(cfg.get("modifier", "alt")).lower())
        self._field("Together with", self.modifier)
        self.plain = self._combo([("Does nothing", "block"), ("Works as usual", "copilot")],
                                 str(cfg.get("plain_key", "block")).lower())
        self.plain_label = self._field("", self.plain)
        for cb in (self.key, self.modifier):
            cb.bind("<<ComboboxSelected>>", lambda _e: self._update_hotkey(), add="+")

        self._heading("Speech")
        self.language = self._combo(LANGUAGES, cfg.get("language", "en"))
        self._field("Language", self.language)
        order = list(MODEL_NAMES)
        models = [p.name[5:-4] for p in MODELS_DIR.glob("ggml-*.bin")]             # ggml-<name>.bin
        if (ROOT / "llama.cpp" / "llama-server.exe").exists():                     # Qwen3-ASR-<size>-Q8_0.gguf
            models += ["qwen3-asr-" + p.name[10:-10].lower() for p in MODELS_DIR.glob("Qwen3-ASR-*-Q8_0.gguf")]
        models.sort(key=lambda m: (order.index(m) if m in order else len(order), m))
        self.model = self._combo([(MODEL_NAMES.get(m, m), m) for m in models], cfg.get("model", "large-v3-turbo"))
        self._field("Model", self.model)
        self.vocabulary = tk.Text(self.body, width=36, height=3, wrap="word", font=("Segoe UI", 9),
                                  relief="flat", borderwidth=0, highlightthickness=1, padx=4, pady=3,
                                  highlightbackground="#a0a0a0", highlightcolor="#0067c0")
        self.vocabulary.insert("1.0", cfg.get("vocabulary", ""))
        self._field("Names and terms", self.vocabulary,
                    hint="Separate with commas. Whisper will spell these the way you write them here.")

        self._heading("Listening")
        silence = float(cfg.get("stop_after_silence", 1.5))
        self.silence = self._combo([(f"{s:g} seconds", s) for s in SILENCE], silence)
        self._field("Stop after a pause of", self.silence, hint="Used when the key is tapped rather than held.")
        self.unload = self._combo(UNLOAD, int(cfg.get("unload_after_minutes", 10) or 0))
        self._field("Free the model's memory after", self.unload,
                    hint="This long without use. Loading it again takes a few seconds.")

        self._heading("Typing")
        self.trailing_space = tk.BooleanVar(value=bool(cfg.get("trailing_space", True)))
        ttk.Checkbutton(self.body, text="Add a space after the text", variable=self.trailing_space).grid(
            row=self.row, column=0, columnspan=2, sticky="w", pady=2)
        self.row += 1
        self.save_history = tk.BooleanVar(value=bool(cfg.get("save_history", True)))
        history = ttk.Frame(self.body)
        ttk.Checkbutton(history, text="Keep a history of everything you dictate", variable=self.save_history).pack(
            side="left")
        ttk.Button(history, text="Open folder", command=self._open_history).pack(side="left", padx=(12, 0))
        history.grid(row=self.row, column=0, columnspan=2, sticky="w", pady=2)
        self.row += 1

        ttk.Separator(self.body).grid(row=self.row, column=0, columnspan=2, sticky="ew", pady=(16, 12))
        self.row += 1
        buttons = ttk.Frame(self.body)
        ttk.Button(buttons, text="Cancel", command=w.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", default="active", command=self.save).pack(side="right", padx=(0, 8))
        ttk.Label(buttons, text="Changes apply as soon as you save.", foreground="#666").pack(side="left")
        buttons.grid(row=self.row, column=0, columnspan=2, sticky="ew")
        w.bind("<Return>", lambda e: None if e.widget is self.vocabulary else self.save())
        self._update_hotkey()

    def _heading(self, text):
        ttk.Label(self.body, text=text, font=("Segoe UI Semibold", 10)).grid(
            row=self.row, column=0, columnspan=2, sticky="w", pady=(14, 4))
        self.row += 1

    def _field(self, label, widget, hint=None):
        lbl = ttk.Label(self.body, text=label)
        lbl.grid(row=self.row, column=0, sticky="nw", padx=(0, 16), pady=3)
        widget.grid(row=self.row, column=1, sticky="w", pady=3)
        self.row += 1
        if hint:
            ttk.Label(self.body, text=hint, foreground="#666", font=("Segoe UI", 8)).grid(
                row=self.row, column=1, sticky="w", pady=(0, 4))
            self.row += 1
        return lbl

    def _combo(self, options, value):
        """A drop-down of (label, value) pairs; a value set by hand in the JSON is kept as an extra entry."""
        options = list(options)
        if value not in [v for _, v in options]:
            options.append((str(value), value))
        values = [v for _, v in options]
        cb = ttk.Combobox(self.body, values=[label for label, _ in options], state="readonly", width=34)
        cb.current(values.index(value))
        cb.value = lambda: values[cb.current()]
        cb.bind("<<ComboboxSelected>>", lambda _e: cb.selection_clear())
        return cb

    def _update_hotkey(self):
        cfg = {"hotkey_vk": self.key.value(), "modifier": self.modifier.value()}
        key = key_name(cfg["hotkey_vk"])
        self.summary.configure(text=f"Press {hotkey_text(cfg)} and speak. "
                                    "The text is typed wherever your cursor is.")
        self.plain_label.configure(text=f"{key} on its own")
        usual = "Opens Copilot" if cfg["hotkey_vk"] == 0x86 else "Works as usual"
        current = self.plain.current()
        self.plain.configure(values=["Does nothing", usual])
        self.plain.current(current)
        # with no modifier the key itself is the hotkey, so there is no "on its own"
        self.plain.configure(state="disabled" if cfg["modifier"] == "none" else "readonly")

    def _open_history(self):
        HISTORY.mkdir(exist_ok=True)
        os.startfile(HISTORY)

    def show(self):
        w = self.win
        w.update_idletasks()
        x = (w.winfo_screenwidth() - w.winfo_reqwidth()) // 2
        y = (w.winfo_screenheight() - w.winfo_reqheight()) // 3
        if not w.winfo_viewable():
            w.geometry(f"+{x}+{y}")
        w.deiconify()
        w.lift()
        w.attributes("-topmost", True)                  # come to the front over the app you were in
        w.after(300, lambda: w.winfo_exists() and w.attributes("-topmost", False))
        w.focus_force()

    def save(self):
        vocab = re.split(r"[,\n]+", self.vocabulary.get("1.0", "end"))
        new = dict(self.cfg)                             # keeps settings this window doesn't show
        new.update(
            hotkey_vk=self.key.value(),
            modifier=self.modifier.value(),
            plain_key=self.plain.value(),
            language=self.language.value(),
            model=self.model.value(),
            vocabulary=", ".join(v.strip() for v in vocab if v.strip()),
            stop_after_silence=self.silence.value(),
            unload_after_minutes=self.unload.value(),
            trailing_space=self.trailing_space.get(),
            save_history=self.save_history.get(),
        )
        try:
            tmp = CONFIG.with_suffix(".json.tmp")         # replace in one step: VoiceType never reads half a file
            tmp.write_text(json.dumps(new, indent=2), encoding="utf-8")
            os.replace(tmp, CONFIG)
        except OSError as e:
            messagebox.showerror("VoiceType", f"Couldn't save the settings:\n{e}", parent=self.win)
            return
        self.win.destroy()
