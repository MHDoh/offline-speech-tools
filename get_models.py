"""Download the models and the llama.cpp runtime the apps need (setup.bat runs this).

    .venv\\Scripts\\python get_models.py              # everything (~4.4 GB)
    .venv\\Scripts\\python get_models.py translator   # only what one app needs
    .venv\\Scripts\\python get_models.py voicetype
    .venv\\Scripts\\python get_models.py small        # an extra Whisper model (ggml-small.bin)

Every file comes from its original publisher; nothing is re-hosted except the CTranslate2
conversion of OPUS-MT (there is no official one), which is attached to this repo's release.
Transcribe downloads its own models on first use into the same models\\ folder.
Files that are already there are skipped, so running it again is cheap.
"""
import os
import shutil
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_XET", "1")     # plain HTTP downloads resume after a network hiccup
from huggingface_hub import hf_hub_download  # noqa: E402

ROOT = Path(__file__).resolve().parent
MODELS = ROOT / "models"
LLAMA = ROOT / "llama.cpp"
REPO_RELEASE = "https://github.com/MHDoh/offline-speech-tools/releases/download/v1.0"

# name: (what, size, source, license)
SOURCES = {
    "ggml-medium.bin": ("Whisper medium, Meeting Translator live", "1.5 GB",
                        "huggingface.co/ggerganov/whisper.cpp", "MIT"),
    "ggml-large-v3-turbo.bin": ("Whisper large-v3-turbo, Meeting Translator files + VoiceType option", "1.6 GB",
                                "huggingface.co/ggerganov/whisper.cpp", "MIT"),
    "Qwen3-ASR-0.6B-Q8_0.gguf": ("Qwen3-ASR 0.6B, VoiceType default", "0.8 GB",
                                 "huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF", "Apache-2.0"),
    "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf": ("Qwen3-ASR 0.6B audio encoder", "0.2 GB",
                                        "huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF", "Apache-2.0"),
    "silero_vad.onnx": ("Silero VAD v5.1.2, speech detection", "2 MB",
                        "github.com/snakers4/silero-vad", "MIT"),
    "opus-mt-tc-big-tr-en-ct2": ("Helsinki-NLP/opus-mt-tc-big-tr-en, CTranslate2 int8, Turkish->English", "0.2 GB",
                                 "huggingface.co/Helsinki-NLP/opus-mt-tc-big-tr-en", "CC-BY-4.0"),
    "llama.cpp": ("llama.cpp b11415 Windows Vulkan build, runs Qwen3-ASR", "0.03 GB",
                  "github.com/ggml-org/llama.cpp/releases/tag/b11415", "MIT"),
}
APPS = {
    "translator": ["silero_vad.onnx", "ggml-medium.bin", "ggml-large-v3-turbo.bin", "opus-mt-tc-big-tr-en-ct2"],
    "voicetype": ["silero_vad.onnx", "Qwen3-ASR-0.6B-Q8_0.gguf", "mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", "llama.cpp"],
}


def retry(fn, what, tries=20):
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            print(f"  {what}: {type(e).__name__}: {str(e)[:120]} (retry {i + 1}/{tries})", flush=True)
            time.sleep(5)
    raise SystemExit(f"failed to download {what}")


def fetch(url, dst):
    """Download url to dst via a .part file, so an interrupted download is never mistaken for a finished one."""
    part = dst.with_name(dst.name + ".part")
    with urllib.request.urlopen(url, timeout=60) as r, part.open("wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    part.replace(dst)


def unzip_url(url, dst):
    tmp = dst.with_name(dst.name + ".zip")
    retry(lambda: fetch(url, tmp), tmp.name)
    with zipfile.ZipFile(tmp) as z:
        z.extractall(dst.with_name(dst.name + ".tmp"))
    tmp.unlink()
    src = dst.with_name(dst.name + ".tmp")
    inner = [p for p in src.iterdir()]
    if len(inner) == 1 and inner[0].is_dir():          # zip with a single top folder
        src = inner[0]
    shutil.rmtree(dst, ignore_errors=True)             # a half-finished earlier attempt
    shutil.move(str(src), str(dst))
    shutil.rmtree(dst.with_name(dst.name + ".tmp"), ignore_errors=True)


def get(name):
    if name == "llama.cpp":
        done = (LLAMA / "llama-server.exe").exists()
    else:
        dst = MODELS / name
        done = (dst / "model.bin").exists() if name.endswith("-ct2") else dst.exists()
    what, size, source, lic = SOURCES.get(name, (f"Whisper {name}", "?", "huggingface.co/ggerganov/whisper.cpp", "MIT"))
    if done:
        print(f"ok   {name}")
        return
    print(f"get  {name} ({size}) - {what}\n     from {source} ({lic})", flush=True)
    if name.startswith("ggml-"):
        retry(lambda: hf_hub_download("ggerganov/whisper.cpp", name, local_dir=MODELS), name)
    elif name.endswith(".gguf"):
        retry(lambda: hf_hub_download("ggml-org/Qwen3-ASR-0.6B-GGUF", name, local_dir=MODELS), name)
    elif name == "silero_vad.onnx":
        url = "https://github.com/snakers4/silero-vad/raw/v5.1.2/src/silero_vad/data/silero_vad.onnx"
        retry(lambda: fetch(url, MODELS / name), name)
    elif name == "opus-mt-tc-big-tr-en-ct2":
        unzip_url(f"{REPO_RELEASE}/opus-mt-tc-big-tr-en-ct2.zip", MODELS / name)
    elif name == "llama.cpp":
        unzip_url("https://github.com/ggml-org/llama.cpp/releases/download/b11415/llama-b11415-bin-win-vulkan-x64.zip",
                  LLAMA)
    shutil.rmtree(MODELS / ".cache", ignore_errors=True)      # hf_hub_download bookkeeping


def main():
    MODELS.mkdir(exist_ok=True)
    wanted = []
    for arg in sys.argv[1:] or ["all"]:
        if arg == "all":
            wanted += APPS["translator"] + APPS["voicetype"]
        elif arg in APPS:
            wanted += APPS[arg]
        else:
            wanted.append(arg if arg.startswith("ggml-") else f"ggml-{arg}.bin")
    for name in dict.fromkeys(wanted):
        get(name)
    print("models ready")


if __name__ == "__main__":
    main()
