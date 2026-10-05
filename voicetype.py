#!/usr/bin/env python
"""VoiceType: Alt+Copilot key = local dictation (English) into the focused text box.

Press Alt+Copilot and speak; the text is typed into whatever box has the focus. Most
keyboards send the Copilot key as a short tap, so it stops by itself after a pause (or
press again). If the key reports a real hold, releasing it ends the message instead.
The plain Copilot key does nothing (the settings can bring Copilot back).

Everything runs locally on the GPU: Qwen3-ASR 0.6B (llama.cpp + Vulkan) by default, which
measured more accurate and ~3x faster than Whisper on English; Whisper large-v3-turbo (the
Meeting Translator's engine, better for Turkish) can be picked in the settings. Lives in the tray: click the icon for the settings
(saved in voicetype.json); "Turn off" in its menu frees the model's memory until turned on.

    python voicetype.py            # run (normally started from the tray exe at logon)
    python voicetype.py --keylog   # print the key codes you press, to find another hotkey
"""
import argparse
import ctypes
import datetime as dt
import json
import os
import queue
import re
import subprocess
import sys
import threading
import time
from ctypes import wintypes

from paths import app_root

ROOT = app_root()
CONFIG = ROOT / "voicetype.json"
ERROR_LOG = ROOT / "error.log"
CRASH_LOG = ROOT / "voicetype_crash.log"
HISTORY = ROOT / "transcripts"
# Whisper writes in the style of its prompt; without one it sometimes drops all punctuation.
STYLE_PROMPT = "Hi, here's a quick update. I'll send the details later, okay?"
DEFAULTS = {
    "model": "qwen3-asr-0.6b",  # or a Whisper model: "large-v3-turbo" (best for Turkish), "medium", ...
    "language": "en",
    "vocabulary": "",           # names/jargon Whisper should spell right, e.g. "Acme, Kubernetes, Dr. Yılmaz"
    "hotkey_vk": 0x86,          # F23: the Copilot key sends Win+Shift+F23
    "modifier": "alt",          # dictate on Alt+Copilot ("alt", "ctrl", or "none" = the plain key)
    "plain_key": "block",       # plain Copilot key: "block" = does nothing, "copilot" = opens Copilot
    "stop_after_silence": 1.5,  # tap mode: seconds of silence that end the message
    "trailing_space": True,
    "save_history": True,       # also append each dictation to transcripts/dictation_<month>.txt
    "unload_after_minutes": 10, # free the model's ~2 GB after this long unused (0 = keep it loaded)
}

user32 = ctypes.WinDLL("user32", use_last_error=True)
kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

WH_KEYBOARD_LL, WM_KEYDOWN, WM_KEYUP, WM_SYSKEYDOWN, WM_SYSKEYUP = 13, 0x100, 0x101, 0x104, 0x105
WM_TIMER = 0x113
LLKHF_INJECTED = 0x10
VK_MASK = 0xE8        # unassigned key: pressed so Windows doesn't open Start when Win is released
INPUT_KEYBOARD, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE = 1, 0x2, 0x4
GWL_EXSTYLE, WS_EX_NOACTIVATE, WS_EX_TOOLWINDOW, WS_EX_TOPMOST = -20, 0x08000000, 0x80, 0x8
SW_HIDE = 0
HWND_TOPMOST, SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, SWP_SHOWWINDOW = -1, 0x1, 0x2, 0x10, 0x40
MONITOR_DEFAULTTOPRIMARY = 1


class KBDLLHOOKSTRUCT(ctypes.Structure):
    _fields_ = [("vkCode", wintypes.DWORD), ("scanCode", wintypes.DWORD), ("flags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class KEYBDINPUT(ctypes.Structure):
    _fields_ = [("wVk", wintypes.WORD), ("wScan", wintypes.WORD), ("dwFlags", wintypes.DWORD),
                ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MOUSEINPUT(ctypes.Structure):
    _fields_ = [("dx", wintypes.LONG), ("dy", wintypes.LONG), ("mouseData", wintypes.DWORD),
                ("dwFlags", wintypes.DWORD), ("time", wintypes.DWORD), ("dwExtraInfo", ctypes.c_size_t)]


class MONITORINFO(ctypes.Structure):
    _fields_ = [("cbSize", wintypes.DWORD), ("rcMonitor", wintypes.RECT), ("rcWork", wintypes.RECT),
                ("dwFlags", wintypes.DWORD)]


class INPUT(ctypes.Structure):
    class _U(ctypes.Union):
        _fields_ = [("ki", KEYBDINPUT), ("mi", MOUSEINPUT)]
    _anonymous_ = ("u",)
    _fields_ = [("type", wintypes.DWORD), ("u", _U)]


HOOKPROC = ctypes.WINFUNCTYPE(ctypes.c_ssize_t, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
user32.SetWindowsHookExW.argtypes = [ctypes.c_int, HOOKPROC, wintypes.HINSTANCE, wintypes.DWORD]
user32.SetWindowsHookExW.restype = wintypes.HHOOK
user32.CallNextHookEx.argtypes = [wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM]
user32.CallNextHookEx.restype = ctypes.c_ssize_t
user32.UnhookWindowsHookEx.argtypes = [wintypes.HHOOK]
user32.SetTimer.argtypes = [wintypes.HWND, ctypes.c_size_t, wintypes.UINT, ctypes.c_void_p]
user32.SetTimer.restype = ctypes.c_size_t
user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(INPUT), ctypes.c_int]
user32.GetAsyncKeyState.argtypes = [ctypes.c_int]
user32.GetAsyncKeyState.restype = ctypes.c_short
user32.GetParent.restype = wintypes.HWND
user32.GetParent.argtypes = [wintypes.HWND]
user32.GetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int]
user32.SetWindowLongW.argtypes = [wintypes.HWND, ctypes.c_int, wintypes.LONG]
user32.ShowWindow.argtypes = [wintypes.HWND, ctypes.c_int]
user32.IsWindowVisible.argtypes = [wintypes.HWND]
user32.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                wintypes.UINT]
user32.GetForegroundWindow.restype = wintypes.HWND
user32.MonitorFromWindow.argtypes = [wintypes.HWND, wintypes.DWORD]
user32.MonitorFromWindow.restype = wintypes.HMONITOR
user32.GetMonitorInfoW.argtypes = [wintypes.HMONITOR, ctypes.POINTER(MONITORINFO)]
kernel32.GetModuleHandleW.restype = wintypes.HMODULE


def log_error(where, exc):
    import traceback

    try:
        with ERROR_LOG.open("a", encoding="utf-8") as fh:
            fh.write(f"\n[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] voicetype {where}\n")
            fh.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    except Exception:
        pass


DEBUG_LOG = ROOT / "voicetype_debug.log"
_debug_q = queue.Queue()


def debug(msg):
    """Key/state events for diagnosing missed presses; cheap enough to call from the hook."""
    _debug_q.put(f"{dt.datetime.now():%Y-%m-%d %H:%M:%S.%f}"[:-3] + " " + msg)


def _debug_writer():
    while True:
        lines = [_debug_q.get()]
        while not _debug_q.empty():
            lines.append(_debug_q.get_nowait())
        try:
            if DEBUG_LOG.exists() and DEBUG_LOG.stat().st_size > 1_000_000:
                DEBUG_LOG.replace(DEBUG_LOG.with_suffix(".old.log"))
            with DEBUG_LOG.open("a", encoding="utf-8") as fh:
                fh.write("".join(line + chr(10) for line in lines))
        except Exception:
            pass


threading.Thread(target=_debug_writer, daemon=True).start()

_crash_log = None


def capture_crashes():
    """pythonw has no stderr, so a native abort (whisper.cpp / the GPU driver) left no trace.
    Send stderr - whisper.cpp's messages, including the reason it gives before aborting - and
    Python's fault dump (which thread, which line) to voicetype_crash.log instead."""
    global _crash_log
    import faulthandler

    _crash_log = CRASH_LOG.open("a", encoding="utf-8")
    trim_crash_log()
    faulthandler.enable(file=_crash_log, all_threads=True)
    # whisper.cpp prints to the C runtime's stderr, which goes nowhere in a windowed app
    ucrt = ctypes.CDLL("ucrtbase")
    ucrt.__acrt_iob_func.argtypes = [ctypes.c_uint]
    ucrt.__acrt_iob_func.restype = ctypes.c_void_p
    ucrt._wfreopen.argtypes = [ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_void_p]
    ucrt._wfreopen.restype = ctypes.c_void_p
    ucrt.setvbuf.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t]
    stderr = ucrt.__acrt_iob_func(2)
    if ucrt._wfreopen(str(CRASH_LOG), "a", stderr):
        ucrt.setvbuf(stderr, None, 4, 0)                 # _IONBF: an abort doesn't flush buffers
    _crash_log.write(f"\n=== VoiceType started {dt.datetime.now():%Y-%m-%d %H:%M:%S} ===\n")
    _crash_log.flush()


def trim_crash_log():
    """whisper.cpp prints ~80 lines per model load: start over past 2 MB."""
    if _crash_log is not None and os.fstat(_crash_log.fileno()).st_size > 2_000_000:
        os.ftruncate(_crash_log.fileno(), 0)


def load_config():
    cfg = dict(DEFAULTS)
    try:
        cfg.update(json.loads(CONFIG.read_text(encoding="utf-8-sig")))   # utf-8-sig: Notepad may add a BOM
    except FileNotFoundError:
        CONFIG.write_text(json.dumps(DEFAULTS, indent=2), encoding="utf-8")
    except Exception as e:
        log_error("config", e)
    return cfg


def work_area():
    """The screen you are working on (the one with the focused window), minus the taskbar."""
    info = MONITORINFO(cbSize=ctypes.sizeof(MONITORINFO))
    user32.GetMonitorInfoW(user32.MonitorFromWindow(user32.GetForegroundWindow(), MONITOR_DEFAULTTOPRIMARY),
                           ctypes.byref(info))
    return info.rcWork


def raise_overlay(hwnd):
    """Put the overlay above all other always-on-top windows, without taking the focus. Being
    always-on-top only puts it in that group; within it, whichever window came up last covers the rest."""
    user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE | SWP_SHOWWINDOW)


def type_text(text):
    """Type text into the focused window as Unicode key events (the clipboard is untouched)."""
    units = text.encode("utf-16-le")
    codes = [int.from_bytes(units[i:i + 2], "little") for i in range(0, len(units), 2)]
    inputs = (INPUT * (2 * len(codes)))()
    for i, c in enumerate(codes):
        for j, flags in enumerate((KEYEVENTF_UNICODE, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP)):
            inp = inputs[2 * i + j]
            inp.type = INPUT_KEYBOARD
            inp.ki = KEYBDINPUT(0, c, flags, 0, 0)
    user32.SendInput(len(inputs), inputs, ctypes.sizeof(INPUT))


MODIFIER_VK = {"alt": 0x12, "ctrl": 0x11, "none": None}


class KeyHook:
    """Low-level keyboard hook: reports the hotkey's press/release and swallows it.

    With a modifier (default Alt), only Alt+hotkey is taken over; the plain key is passed on.
    """

    def __init__(self, vk, on_event, keylog=False, modifier="alt", block_plain=True):
        self.on_event, self.keylog = on_event, keylog
        self.set_hotkey(vk, modifier, block_plain)
        self.held = False
        self.blocked = False      # a plain press we swallowed: swallow its release too
        self.LATE_MODIFIER = 0.4  # Alt pressed up to this long after the Copilot key still counts
        self.plain_at = 0.0       # time of the last blocked plain press (Alt may follow a moment later)
        self._proc = HOOKPROC(self._callback)     # keep a reference: ctypes callbacks get GC'd
        threading.Thread(target=self._run, daemon=True).start()

    def set_hotkey(self, vk, modifier="alt", block_plain=True):
        """Also called on a running hook when the settings change."""
        self.vk = vk
        self.modifier = MODIFIER_VK.get(str(modifier).lower(), 0x12)
        self.block_plain = block_plain

    def _callback(self, code, wparam, lparam):
        if code == 0:
            k = ctypes.cast(lparam, ctypes.POINTER(KBDLLHOOKSTRUCT)).contents
            down = wparam in (WM_KEYDOWN, WM_SYSKEYDOWN)
            if self.keylog and not k.flags & LLKHF_INJECTED:
                print(f"{'down' if down else 'up  '} vk=0x{k.vkCode:02X} ({k.vkCode}) scan={k.scanCode}", flush=True)
            if (down and k.vkCode in (0x12, 0xA4, 0xA5) and self.modifier == 0x12 and self.plain_at
                    and not k.flags & LLKHF_INJECTED and time.monotonic() - self.plain_at < self.LATE_MODIFIER):
                # Alt went down just after the Copilot key: still counts as Alt+Copilot
                self.plain_at = 0.0
                debug("late Alt after Copilot -> dictation (tap)")
                self.on_event("down")
                self.on_event("up")
            if k.vkCode == self.vk and not k.flags & LLKHF_INJECTED and not self.keylog:
                mod_down = bool(self.modifier and user32.GetAsyncKeyState(self.modifier) & 0x8000)
                if down and not self.held and self.modifier and not mod_down:
                    if not self.block_plain:
                        debug("Copilot without Alt -> passed to Copilot")
                        return user32.CallNextHookEx(None, code, wparam, lparam)    # plain key: Copilot
                    if not self.blocked:                                          # plain key: nothing
                        self.blocked = True
                        self.plain_at = time.monotonic()
                        debug("Copilot without Alt -> blocked (Alt within 0.4 s still triggers)")
                        user32.keybd_event(VK_MASK, 0, 0, 0)
                        user32.keybd_event(VK_MASK, 0, KEYEVENTF_KEYUP, 0)
                    return 1
                if not down and not self.held:
                    if self.blocked:
                        self.blocked = False
                        return 1
                    return user32.CallNextHookEx(None, code, wparam, lparam)
                if down and not self.held:
                    self.held = True
                    self.plain_at = 0.0
                    debug("Alt+Copilot down")
                    # the Copilot key also presses Win: tap an unused key so Start doesn't open
                    user32.keybd_event(VK_MASK, 0, 0, 0)
                    user32.keybd_event(VK_MASK, 0, KEYEVENTF_KEYUP, 0)
                    self.on_event("down")
                elif not down:
                    self.held = False
                    debug("Alt+Copilot up")
                    self.on_event("up")
                return 1                            # swallow: Copilot doesn't open
        return user32.CallNextHookEx(None, code, wparam, lparam)

    def _install(self):
        return user32.SetWindowsHookExW(WH_KEYBOARD_LL, self._proc, kernel32.GetModuleHandleW(None), 0)

    def _run(self):
        hook = self._install()
        if not hook:
            log_error("hook", OSError(ctypes.get_last_error(), "SetWindowsHookEx failed"))
            return
        # Windows silently removes a low-level hook that ever answers too slowly (e.g. when the
        # PC stalls); re-register it every 30 s so the key can't quietly go back to Copilot.
        user32.SetTimer(None, 0, 30000, None)
        msg = wintypes.MSG()
        while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) != 0:
            if msg.message == WM_TIMER and not self.held:
                fresh = self._install()
                if fresh:
                    user32.UnhookWindowsHookEx(hook)
                    hook = fresh
                continue
            user32.TranslateMessage(ctypes.byref(msg))
            user32.DispatchMessageW(ctypes.byref(msg))


class Dictation:
    """Records while the key is held (or until a pause), transcribes, types the text."""

    HOLD_THRESHOLD = 0.4      # a key-up sooner than this means the keyboard only sends taps

    def __init__(self, cfg, ui):
        from vad import SileroVAD

        self.cfg, self.ui = cfg, ui
        self.events = queue.Queue()
        self.vad = SileroVAD(ROOT / "models" / "silero_vad.onnx")
        # The Whisper model (~2 GB) is only loaded when the key is pressed, in parallel with the
        # recording, so most of its ~8 s load is hidden behind your speech. It is freed again
        # after unload_after_minutes without use.
        self.backend, self.load_error, self.loaded = None, None, threading.Event()
        self.last_used = time.monotonic()
        self.busy = False
        self.enabled = True       # "Turn off" in the tray menu: free the model, ignore the key
        threading.Thread(target=self._loop, daemon=True).start()

    def set_enabled(self, on):
        self.enabled = on
        if not on:
            self.events.put(("off", time.monotonic()))     # the dictation thread frees the model

    def key(self, what):
        if self.busy:
            if what == "down":
                debug("press while still transcribing the previous message -> ignored")
                self.ui("error", "Still typing the last message, press again in a second")
            return
        self.events.put((what, time.monotonic()))

    def _start_loading(self):
        if self.backend is not None or getattr(self, "_loader", None) and self._loader.is_alive():
            return
        self.loaded.clear()
        self.load_error = None

        def load():
            from backends import QwenAsrBackend, WhisperCppBackend
            trim_crash_log()
            t0 = time.monotonic()
            try:
                if self.cfg["model"].startswith("qwen3-asr"):
                    self.backend = QwenAsrBackend(model=self.cfg["model"])
                else:
                    # Full audio context: the shortened "auto" one saved only ~0.3 s per message here
                    # but made turbo less sure and invent words where the audio window was cut.
                    self.backend = WhisperCppBackend(model=self.cfg["model"], audio_ctx="full", beam=5)
                debug(f"model loaded in {time.monotonic() - t0:.1f} s")
            except Exception as e:
                self.load_error = e
            self.loaded.set()

        self._loader = threading.Thread(target=load, daemon=True)
        self._loader.start()

    def _drop_model(self):
        if hasattr(self.backend, "close"):
            self.backend.close()                # Qwen: stop its llama-server process
        self.backend = None
        self.loaded.clear()

    def _unload_if_unused(self):
        minutes = float(self.cfg.get("unload_after_minutes", 10) or 0)
        idle = minutes and time.monotonic() - self.last_used > minutes * 60
        if self.backend is not None and (idle or not self.enabled):
            import gc

            self._drop_model()
            gc.collect()
            debug("model unloaded")

    def _loop(self):
        while True:
            try:
                what, t = self.events.get(timeout=30)
            except queue.Empty:
                self._unload_if_unused()
                continue
            if what == "off":
                self._unload_if_unused()
                continue
            if what != "down":
                continue
            if not self.enabled:
                self.ui("error", "VoiceType is off. Turn it on from its tray icon.")
                time.sleep(2)
                self.ui("idle", None)
                continue
            if self.backend is not None and (self.backend.model_name != self.cfg["model"]
                                             or not getattr(self.backend, "alive", lambda: True)()):
                self._drop_model()                  # another model was picked, or its server died
            debug("recording")
            self._start_loading()
            try:
                audio = self._record(t)
                if audio is not None:
                    self._transcribe(audio)
            except Exception as e:
                log_error("dictation", e)
                self.ui("error", f"Error: {e}")
                time.sleep(2)
            self.last_used = time.monotonic()
            self.ui("idle", None)
            while not self.events.empty():     # drop presses made while we were busy
                self.events.get_nowait()
            self._unload_if_unused()           # in case it was turned off mid-dictation

    def _record(self, t_down):
        import numpy as np

        from audio_capture import Capture

        self.vad.reset()
        cap = Capture(mic=True)
        cap.start()
        self.ui("listening", "Listening...")
        chunks, tap_mode = [], False
        speech_seen, silent_for, last_vad = False, 0.0, np.zeros(0, np.float32)
        silence_needed = float(self.cfg["stop_after_silence"])
        try:
            while True:
                try:
                    what, t = self.events.get(timeout=0.05)
                except queue.Empty:
                    what = None
                if what == "up" and not tap_mode:
                    if t - t_down >= self.HOLD_THRESHOLD:
                        break                                # released after holding: done
                    tap_mode = True                          # keyboard only sends taps
                    self.ui("listening", "Listening... (stops when you pause)")
                elif what == "down" and tap_mode:
                    break                                    # second tap: done
                while not cap.q.empty():
                    c = cap.q.get_nowait()
                    chunks.append(c)
                    last_vad = np.concatenate([last_vad, c])
                    while len(last_vad) >= 512:          # voice detection, for the tap mode stop
                        p = self.vad.prob(last_vad[:512])
                        last_vad = last_vad[512:]
                        if p >= 0.5:
                            speech_seen, silent_for = True, 0.0
                        else:
                            silent_for += 512 / 16000
                elapsed = time.monotonic() - t_down
                if tap_mode and speech_seen and silent_for >= silence_needed:
                    break
                if tap_mode and not speech_seen and elapsed > 8:
                    self.ui("error", "Didn't hear anything")
                    time.sleep(1)
                    return None
                if elapsed > 120:
                    break
        finally:
            cap.stop()
        if not chunks:
            return None
        return np.concatenate(chunks)

    def _trim(self, audio):
        """Cut leading/trailing silence (Whisper invents words in it); None if no speech."""
        self.vad.reset()
        n = len(audio) // 512
        probs = [self.vad.prob(audio[i * 512:(i + 1) * 512]) for i in range(n)]
        speech = [i for i, p in enumerate(probs) if p >= 0.5]
        if len(speech) < 8:                           # < 0.25 s of speech
            return None
        a = max(0, (speech[0] - 10) * 512)            # keep 0.3 s around the speech
        b = min(len(audio), (speech[-1] + 11) * 512)
        return audio[a:b]

    def _transcribe(self, audio):
        self.busy = True
        try:
            self._transcribe_inner(audio)
        finally:
            self.busy = False

    def _transcribe_inner(self, audio):
        import numpy as np

        from textfilter import clean

        audio = self._trim(audio)
        if audio is None:
            debug("no speech heard")
            self.ui("error", "Didn't hear anything")
            time.sleep(1)
            return
        if not self.loaded.is_set():
            self.ui("busy", "Loading the model (first use in a while)...")
            self.loaded.wait()
        if self.load_error:
            raise self.load_error
        self.ui("busy", "Transcribing...")
        vocabulary = self.cfg.get("vocabulary") or ""
        style = STYLE_PROMPT if getattr(self.backend, "uses_style_prompt", True) else ""
        prompt = " ".join(p for p in (style, vocabulary) if p)
        t0 = time.monotonic()
        r = self.backend.run(audio, task="transcribe", language=self.cfg["language"], prompt=prompt)
        # how loud the mic was and how sure the model is: for telling a bad mic from a bad model
        rms = float(np.sqrt(np.mean(np.square(audio)))) + 1e-9
        debug(f"heard {len(audio) / 16000:.1f} s at {20 * np.log10(rms):.0f} dBFS, "
              f"confidence {r.confidence:.2f}, transcribed in {time.monotonic() - t0:.1f} s")
        text = clean(r.text, r.confidence).strip()
        words = " ".join(re.sub(r"\W+", " ", text.lower()).split())
        if len(words.split()) >= 2 and (words in " ".join(re.sub(r"\W+", " ", prompt.lower()).split())
                                        or set(words.split()) <= set(re.sub(r"\W+", " ", vocabulary.lower()).split())):
            text = ""          # it just repeated its prompt, or (Qwen, on noise) names from the list
        if not text:
            self.ui("error", "Didn't catch that")
            time.sleep(1)
            return
        type_text(text + (" " if self.cfg.get("trailing_space", True) else ""))
        if self.cfg.get("save_history", True):
            HISTORY.mkdir(exist_ok=True)
            with (HISTORY / f"dictation_{dt.datetime.now():%Y-%m}.txt").open("a", encoding="utf-8") as fh:
                fh.write(f"[{dt.datetime.now():%Y-%m-%d %H:%M:%S}] {text}\n")


def hotkey_settings(cfg):
    return dict(vk=int(cfg["hotkey_vk"]), modifier=cfg.get("modifier", "alt"),
                block_plain=str(cfg.get("plain_key", "block")).lower() != "copilot")


def config_mtime():
    try:
        return CONFIG.stat().st_mtime_ns
    except OSError:
        return None


def mic_icon(color, crossed=False):
    from PIL import Image, ImageDraw

    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle((22, 6, 42, 38), radius=10, fill=color)
    d.arc((14, 18, 50, 48), 0, 180, fill=color, width=4)
    d.line((32, 48, 32, 58), fill=color, width=4)
    if crossed:
        d.line((10, 6, 54, 58), fill=(229, 83, 75), width=6)
    return img


def run(cfg, restarted=False):
    import tkinter as tk

    import pystray

    from voicetype_settings import hotkey_text, open_settings

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.configure(bg="#1f2328")
    label = tk.Label(root, text="", fg="#e8eaed", bg="#1f2328", font=("Segoe UI", 11), padx=16, pady=8)
    label.pack()
    root.update_idletasks()
    hwnd = user32.GetParent(root.winfo_id())
    # never take the keyboard focus: the text must go to the window you were typing in
    user32.SetWindowLongW(hwnd, GWL_EXSTYLE, user32.GetWindowLongW(hwnd, GWL_EXSTYLE)
                          | WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST)
    user32.ShowWindow(hwnd, SW_HIDE)
    ui_q = queue.Queue()
    colors = {"listening": "#e5534b", "busy": "#7cc4ff", "error": "#d4a72c", "notice": "#d4a72c", "loading": "#8a9099"}

    app = {}                                # the Dictation and KeyHook, once boot() has made them
    seen_mtime, checked_at = config_mtime(), time.monotonic()
    raised_at = 0.0

    def ui(state, text):
        ui_q.put((state, text))

    def pump():
        nonlocal seen_mtime, checked_at, raised_at
        try:
            while True:
                state, text = ui_q.get_nowait()
                if state == "settings":
                    try:
                        open_settings(root, cfg, img)
                    except Exception as e:
                        log_error("settings", e)
                    continue
                if state == "idle" or not text:
                    user32.ShowWindow(hwnd, SW_HIDE)
                    continue
                dot = "● " if state == "listening" else ""
                label.configure(text=dot + text, fg=colors.get(state, "#e8eaed"))
                root.update_idletasks()
                w, h = root.winfo_reqwidth(), root.winfo_reqheight()
                area = work_area()
                root.geometry(f"{w}x{h}+{area.left + (area.right - area.left - w) // 2}+{area.bottom - h - 40}")
                root.update_idletasks()
                raise_overlay(hwnd)
                raised_at = time.monotonic()
                if state == "notice":            # hide it after a while, unless something else is shown by then
                    root.after(6000, lambda t=text: label.cget("text") == t and user32.ShowWindow(hwnd, SW_HIDE))
        except queue.Empty:
            pass
        if user32.IsWindowVisible(hwnd) and time.monotonic() - raised_at > 0.5:
            raise_overlay(hwnd)                  # stay on top of windows that came up after it
            raised_at = time.monotonic()
        if time.monotonic() - checked_at > 1:
            checked_at = time.monotonic()
            mtime = config_mtime()
            if mtime != seen_mtime:            # saved by the settings window (or an editor)
                seen_mtime = mtime
                try:
                    reload_config()
                except Exception as e:
                    log_error("reload settings", e)
        root.after(40, pump)

    def reload_config():
        cfg.update(load_config())
        if "hook" in app:
            app["hook"].set_hotkey(**hotkey_settings(cfg))
        debug("settings reloaded")
        show_state()

    img = mic_icon((124, 196, 255))
    img_off = mic_icon((138, 144, 153), crossed=True)

    def is_on():
        return "dictation" not in app or app["dictation"].enabled

    def show_state():
        if "hook" not in app:
            return                              # still starting (or failed to)
        icon.icon = img if is_on() else img_off
        icon.title = f"VoiceType: {hotkey_text(cfg)}, then speak" if is_on() else "VoiceType: off"
        icon.update_menu()

    def toggle(_icon, _item):
        if "dictation" in app:
            app["dictation"].set_enabled(not is_on())
            debug("turned " + ("on" if is_on() else "off"))
            show_state()

    def quit_app(icon, _item):
        icon.stop()
        root.after(0, root.destroy)

    icon = pystray.Icon("VoiceType", img, "VoiceType: loading...", menu=pystray.Menu(
        pystray.MenuItem(lambda _i: f"VoiceType: press {hotkey_text(cfg)} and speak" if is_on()
                         else "VoiceType is off", None, enabled=False),
        pystray.MenuItem("Settings...", lambda *_: ui("settings", None), default=True),
        pystray.MenuItem(lambda _i: "Turn off (frees memory)" if is_on() else "Turn on", toggle),
        pystray.MenuItem("Quit", quit_app)))
    icon.run_detached()

    def boot():
        try:
            app["dictation"] = Dictation(cfg, ui)      # the model itself loads on the first key press
            app["hook"] = KeyHook(on_event=app["dictation"].key, **hotkey_settings(cfg))
            show_state()
            if restarted:
                ui("notice", "VoiceType had a problem and restarted itself. Please press the key again.")
        except Exception as e:
            log_error("startup", e)
            ui("error", f"VoiceType failed to start: {e}")
            icon.title = "VoiceType: failed to start (see error.log)"

    threading.Thread(target=boot, daemon=True).start()
    root.after(40, pump)
    root.mainloop()


def supervise():
    """Run the app in a child process and start it again if it crashes - e.g. whisper.cpp
    aborting on the first key press after the PC wakes from standby. Quit ends both."""
    cmd = [sys.executable] + ([] if getattr(sys, "frozen", False) else [os.path.abspath(__file__)])
    crashes, restarted = [], False
    while True:
        code = subprocess.call(cmd + ["--child"] + (["--restarted"] if restarted else []))
        if code == 0:
            return                                       # Quit from the tray
        debug(f"crashed (exit code 0x{code & 0xFFFFFFFF:08X}), restarting; see voicetype_crash.log")
        crashes = [t for t in crashes if time.monotonic() - t < 600] + [time.monotonic()]
        if len(crashes) >= 5:
            log_error("supervisor", RuntimeError("VoiceType crashed 5 times in 10 minutes; not restarting it "
                                                 "(see voicetype_crash.log)"))
            return
        restarted = True
        time.sleep(2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keylog", action="store_true", help="print key codes for 30 s (to pick another hotkey)")
    ap.add_argument("--child", action="store_true", help=argparse.SUPPRESS)       # started by supervise()
    ap.add_argument("--restarted", action="store_true", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a.keylog:
        print("Press keys (e.g. the Copilot key); vk codes are printed for 30 s. Nothing is blocked.")
        KeyHook(0, lambda _e: None, keylog=True)
        time.sleep(30)
        return
    if a.child:
        capture_crashes()
        run(load_config(), restarted=a.restarted)
        os._exit(0)      # Quit: exit code 0 tells supervise() not to restart; don't risk the hook threads' teardown
    mutex = kernel32.CreateMutexW(None, False, "Local\\VoiceTypeSingleInstance")
    if ctypes.get_last_error() == 183:        # ERROR_ALREADY_EXISTS: already running
        return
    supervise()
    del mutex


if __name__ == "__main__":
    main()
