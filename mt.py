"""Turkish -> English text translation (OPUS-MT on CTranslate2, CPU).

Whisper is much better at *transcribing* Turkish than at translating it, so by default
the app transcribes with Whisper and translates the Turkish text with
Helsinki-NLP/opus-mt-tc-big-tr-en, a dedicated translation model trained largely on
film/TV subtitles. It runs int8 on the CPU (a few hundred ms per line), which leaves
the GPU to Whisper.

Names are protected: many Turkish names are ordinary words (Barış = peace, Bahar = spring,
Çağrı = call), so names given with --prompt are swapped for placeholder surnames the
model copies verbatim, then put back ("Barış Bey geç kaldı" -> "Mr. Barış is late", not
"Mr. Peace is late").

The model lives in models/opus-mt-tc-big-tr-en-ct2 (see get_models.py).
"""
import os
import re
from pathlib import Path

# Split after . ! ? … when the next sentence starts (capital letter, digit, quote, dash).
SENT_SPLIT = re.compile(r"(?<=[.!?…])\s+(?=[\"'“‘(\[\-–—]?[A-ZÇĞİÖŞÜ0-9])")
# Surnames OPUS-MT copies unchanged; they stand in for protected names during translation.
PLACEHOLDERS = ["Harrison", "Fitzgerald", "Morrison", "Whitaker", "Pemberton", "Ellington", "Donovan",
                "Sinclair", "Hawthorne", "Lancaster"]


def split_sentences(text):
    parts = [s.strip() for s in SENT_SPLIT.split(text.strip()) if s.strip()]
    return parts or [text.strip()]


def names_from_prompt(prompt):
    """Capitalised entries of the --prompt vocabulary list ("Barış Bey, Pasinler, Acme").

    Honorifics are dropped so the model still translates them ("Barış Bey" -> "Mr. Barış").
    """
    items = [t.strip(" .!?\"'") for t in re.split(r"[,;\n]", prompt or "")]
    items = [re.sub(r"\s+(Bey|Hanım|Hoca|Hocam|Efendi|Paşa)$", "", t) for t in items]
    return [t for t in items if t[:1].isupper()]


class OpusMT:
    name = "opus-mt-tc-big-tr-en"

    def __init__(self, model_dir, threads=None, beam=4, names=()):
        import ctranslate2
        import sentencepiece as spm

        d = Path(model_dir)
        if not (d / "model.bin").exists():
            raise FileNotFoundError(f"translation model not found in {d}")
        self.src = spm.SentencePieceProcessor(model_file=str(d / "source.spm"))
        self.tgt = spm.SentencePieceProcessor(model_file=str(d / "target.spm"))
        self.beam = beam
        self.model = ctranslate2.Translator(str(d), device="cpu", compute_type="int8", inter_threads=1,
                                            intra_threads=threads or min(8, os.cpu_count() or 4))
        self.names = []
        for n in sorted(set(names), key=len, reverse=True):       # longest first: "Çağrı Bey" before "Çağrı"
            # Short names must stand alone or take an apostrophe suffix (Can'ın), so "Canım"
            # (my dear) is not mistaken for "Can"; longer ones may take a bare suffix (Pasinlerde).
            tail = r"(?!\w)" if len(n) < 5 else ""
            self.names.append((n, re.compile(rf"(?<!\w){re.escape(n)}{tail}")))
        self.translate("Merhaba.")   # warm-up

    def _protect(self, sentence):
        swaps = []
        for name, pattern in self.names:
            if not pattern.search(sentence):
                continue
            free = [p for p in PLACEHOLDERS if p not in sentence and p not in dict(swaps)]
            if not free:
                break
            sentence = pattern.sub(free[0], sentence)
            swaps.append((free[0], name))
        return sentence, swaps

    def translate(self, text):
        """Translate Turkish text (one or more sentences) to English."""
        if not text or not text.strip():
            return ""
        sentences, swaps = [], []
        for s in split_sentences(text):
            s, sw = self._protect(s)
            sentences.append(s)
            swaps.append(sw)
        tokens = [self.src.encode(s, out_type=str) + ["</s>"] for s in sentences]
        results = self.model.translate_batch(tokens, beam_size=self.beam, max_decoding_length=256)
        out = []
        for r, sw in zip(results, swaps):
            en = self.tgt.decode(r.hypotheses[0]).strip()
            for placeholder, name in sw:
                en = en.replace(placeholder, name)
            out.append(en)
        return " ".join(out).strip()
