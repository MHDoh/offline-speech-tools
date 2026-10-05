"""Clean-up of Whisper output before it is translated, shown and logged.

Whisper was trained on a lot of subtitled video, so on music, applause or silence it
tends to "hear" subtitle credits and YouTube outros ("Altyazı M.K.", "İzlediğiniz için
teşekkürler", "Thanks for watching"), and on hard audio it can loop ("Ne? Ne? Ne? ...").
"""
import re

from mt import split_sentences

# Sentences that are almost always invented by Whisper, not spoken.
JUNK = re.compile("|".join([
    r"altyaz[ıi]",                                   # "Altyazı M.K." and friends
    r"izlediğiniz için teşekkür",
    r"abone olmay[ıi] unutma",
    r"kanal[ıi]m[ıi]za? abone",
    r"beğenmeyi unutma",
    r"bir sonraki videoda görüşmek",
    r"thanks? (you )?for watching",
    r"don'?t forget to (like|subscribe)",
    r"please subscribe",
    r"subtitles? (by|created by|made by)",
    r"amara\.org",
    r"www\.|https?://",
    r"^\W*(müzik|music|alkış(lar)?|applause|kahkaha|laughter|sessizlik|silence|gülüşmeler)\W*$",
    r"^[\W_]*$",                                     # only punctuation / symbols / ♪
]), re.IGNORECASE)

# Short lines Whisper often produces from noise; dropped only when it is unsure.
WEAK = re.compile(r"^\W*(teşekkürler|teşekkür ederim|hoşça ?kalın|görüşürüz|hmm+|ııı+|thank you|thanks|bye|you)\W*$",
                  re.IGNORECASE)


def _norm(word):
    return re.sub(r"\W+", "", word.lower())


def collapse_repeats(text, keep=3):
    """Cut runs of the same 1..8-word phrase repeated more than ``keep`` times."""
    words = text.split()
    for n in range(1, 9):
        out, i = [], 0
        while i < len(words):
            gram = [_norm(w) for w in words[i:i + n]]
            j, reps = i + n, 1
            while len(gram) == n and any(gram) and [_norm(w) for w in words[j:j + n]] == gram:
                j += n
                reps += 1
            if reps > keep:
                out.extend(words[i:i + n * keep])
                i = j
            else:
                out.append(words[i])
                i += 1
        words = out
    return " ".join(words)


def clean(text, confidence=1.0, weak_below=0.5):
    """Return the cleaned text, or "" if nothing real is left."""
    text = text.replace("�", "").strip()        # partial UTF-8 bytes from the decoder
    if not text:
        return ""
    text = collapse_repeats(text)
    kept = []
    for s in split_sentences(text):
        if JUNK.search(s) or (kept and _norm(s) == _norm(kept[-1])):   # invented, or a looped sentence
            continue
        kept.append(s)
    text = " ".join(kept).strip()
    if WEAK.match(text) and confidence < weak_below:
        return ""
    return text
