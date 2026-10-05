# Offline Speech Tools

Three small Windows apps for speech, running entirely on your own PC. Nothing is uploaded,
nothing needs an account or an API key. They use the GPU (AMD, NVIDIA or Intel, through
Vulkan) when there is one and fall back to the CPU when there isn't.

| App | What it does |
|---|---|
| **Meeting Translator** | Live Turkish → English translation of whatever your speakers play (Meet, Teams, Zoom, a video), in a small always-on-top window. Every line is saved with the Turkish original. Also translates audio/video files. |
| **VoiceType** | Dictation into any text box: hold **Alt + Copilot key**, speak, release; the text is typed where the cursor is. Lives in the tray. |
| **Transcribe** | Drag and drop audio or video files (WhatsApp voice notes, recordings, videos) and get a `.txt` and `.srt` next to each one. 100 languages. |

## Install

1. [Download this repository](../../archive/refs/heads/main.zip) and unzip it (or `git clone` it).
2. Double-click **`setup.bat`**. It installs Python 3.12 if needed (via winget), the Python
   packages into a local `.venv` folder, and downloads the models (~4.4 GB, once).
   Only need one app? `setup.bat voicetype` or `setup.bat translator` downloads less.
3. Start an app with **`MeetingTranslator.bat`**, **`VoiceType.bat`** or **`Transcribe.bat`**.

Requirements: Windows 10/11 (64-bit), ~6 GB of disk, 8 GB of RAM or more. A GPU is optional
but makes everything several times faster. To uninstall, delete the folder
(and run `VoiceType.bat noautostart` first if you turned autostart on).

## Meeting Translator

```
MeetingTranslator.bat                                   the window; listens to the speakers
MeetingTranslator.bat --prompt "Barış Bey, Bahar Hanım, Acme"  names and jargon: big win
MeetingTranslator.bat --file recording.mp3              translate a file
MeetingTranslator.bat --mic                             listen to the microphone instead
MeetingTranslator.bat --show-original                   show the Turkish under each line
MeetingTranslator.bat --model large-v3-turbo --audio-ctx full   live, more accurate, slower
MeetingTranslator.bat --list-devices                    then --device N
```

**Pass names with `--prompt`** (people, places, products). Whisper then spells them right and
the translator keeps them as names: many Turkish names are ordinary words, so without it
"Barış Bey geç kaldı" becomes "Mr. Peace is late"; with it, "Mr. Barış is late".

Window buttons: **Pause**, **Clear**, **Open log**, **Copy** (the whole transcript, e.g. to
paste into an assistant for a summary), **TR** (show the Turkish), **A-/A+**, **On top**.

Each session is written line by line to `transcripts\meeting_<date>_<time>.txt`, so the log
is complete even if the app is closed abruptly:

```
[14:03:05] Hello, welcome to today's meeting.
    TR: Merhaba, bugünkü toplantıya hoş geldiniz.
[14:03:09] First, let's talk about last week's test results.
    TR: Öncelikle geçen haftaki test sonuçlarını konuşalım.
```

How it works: `audio_capture.py` records the speakers through WASAPI loopback (no virtual
cable) → `vad.py` (Silero VAD) cuts the audio into sentences at pauses → `backends.py` runs
Whisper with whisper.cpp to transcribe the Turkish (live: medium, ~1.5 s per line; files:
large-v3-turbo, beam 5) → `textfilter.py` drops what Whisper invents on music and silence →
`mt.py` translates with OPUS-MT (CPU, ~0.25 s per line). Whisper is much better at
transcribing Turkish than at translating it, so a dedicated translation model does step two.

Measured on a TEDx talk (17.5 min, 168 lines) against the human subtitles, with
`eval\evaluate.py`:

| | BLEU | chrF++ | Turkish WER | s per line |
|---|---|---|---|---|
| Whisper medium translating directly (the usual approach) | 14.4 | 39.6 | | 1.3 |
| Live default (medium → OPUS-MT) | **16.0** | 42.4 | 19.5% | 1.6 |
| Files (large-v3-turbo → OPUS-MT) | 14.7 | 42.3 | **17.1%** | 2.3 |

Other languages: `--language de` etc. transcribes them, but the translation model is
Turkish → English only; `--engine whisper` uses Whisper's own translation for any language.

## VoiceType

Start `VoiceType.bat`; a microphone icon appears in the tray. Hold **Alt + Copilot key**
(the key next to the right Alt on newer laptops), speak, and release: the text is typed into
whatever has focus. A short tap starts hands-free mode, which stops after 1.5 s of silence.

**No Copilot key?** Left-click the tray icon → **Settings** and pick F1-F12, Insert or the
Menu key, with Alt, Ctrl or nothing. Settings also has the language, the model, a **names
and terms** list (spelled the way you write them), and how long the model stays in memory.
**Turn off** in the tray menu frees the memory until you turn it on again.
`VoiceType.bat autostart` starts it with Windows.

It uses **Qwen3-ASR 0.6B** by default (through llama.cpp): measured against Whisper
large-v3-turbo on English (VoxPopuli, Earnings-22) it had fewer errors (7.4% vs 8.1-9.5% WER),
was ~3x faster per sentence and is ready in ~2 s instead of 8-18 s. For Turkish dictation
pick **Whisper Large v3 Turbo** in Settings: on Turkish it made about half as many errors.

Files: settings in `voicetype.json`, dictation history in `transcripts\dictation_<month>.txt`
(can be turned off in Settings), problems in `error.log`.

## Transcribe

Start `Transcribe.bat` and drop files or folders on the window (or on `Transcribe.bat`
itself). It reads any audio or video format (ogg, opus, mp3, m4a, wav, mp4, mkv, ...),
writes `<name>.txt` next to each file, and `<name>.srt` subtitles if the box is ticked.
Pick the language (or auto-detect), the model (small / medium / large-v3; downloaded on first
use), and optionally names and terms the speaker uses. Runs on the GPU when available.

## Models

`get_models.py` (run by `setup.bat`) downloads everything from the original publishers into
`models\` and `llama.cpp\`:

| File | Used by | Source | License |
|---|---|---|---|
| `ggml-medium.bin`, `ggml-large-v3-turbo.bin` | Meeting Translator, VoiceType option, Transcribe | [ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp) (OpenAI Whisper) | MIT |
| `Qwen3-ASR-0.6B-Q8_0.gguf` + `mmproj-…` | VoiceType | [ggml-org/Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) (Alibaba Qwen) | Apache-2.0 |
| `opus-mt-tc-big-tr-en-ct2\` | Meeting Translator | [Helsinki-NLP/opus-mt-tc-big-tr-en](https://huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en), converted to CTranslate2 int8 (attached to this repo's release; no official conversion exists) | CC-BY-4.0 |
| `silero_vad.onnx` | Meeting Translator, VoiceType | [snakers4/silero-vad](https://github.com/snakers4/silero-vad) v5.1.2 | MIT |
| `llama.cpp\` | VoiceType (Qwen3-ASR) | [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp/releases/tag/b11415) b11415, Windows Vulkan build | MIT |

The GPU build of [pywhispercpp](https://github.com/absadiki/pywhispercpp) (MIT) is compiled
with Vulkan and attached to the release; if it can't be installed, `setup.bat` installs the
regular CPU build from PyPI.

## Files

| File | Purpose |
|---|---|
| `setup.bat`, `get_models.py` | install packages, download models |
| `translator.py` | Meeting Translator: window, pipeline, transcript log |
| `voicetype.py`, `voicetype_settings.py` | VoiceType: hotkey, recording, typing, tray, settings window |
| `transcribe\transcribe_gui.py`, `transcribe\gpu_backend.py` | Transcribe |
| `backends.py` | speech recognition: whisper.cpp (GPU), faster-whisper (CPU), Qwen3-ASR (llama.cpp) |
| `audio_capture.py`, `vad.py`, `textfilter.py`, `mt.py` | speaker capture, speech detection, clean-up, translation |
| `eval\` | translation quality benchmark (see `eval\README.md`) |

## License

MIT for the code in this repository (see `LICENSE`). The models keep their own licenses,
listed above.
