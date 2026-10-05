"""Translate a test clip with the app's own file mode and score it against human subtitles.

    .venv\\Scripts\\python eval\\evaluate.py tedx                              # app defaults for --file
    .venv\\Scripts\\python eval\\evaluate.py alp --tag live -- --model medium --audio-ctx auto --beam 1
    .venv\\Scripts\\python eval\\evaluate.py --score                           # table of everything in eval\\results

Clips (see get_clips.cmd): clips\\<name>.wav with English references in clips\\<name>.en.vtt
or refs\\<name>.en.vtt, and optionally Turkish ones (<name>.tr.vtt) for the Turkish word error rate.
Scores: BLEU / chrF++ over 30 s windows (robust to different line splits), chrF++ over the
whole document, TR WER, and seconds per line.
"""
import argparse
import json
import math
import re
import sys
import time
from pathlib import Path

EVAL = Path(__file__).resolve().parent
APP = EVAL.parent
CLIPS, REFS, RESULTS = EVAL / "clips", EVAL / "refs", EVAL / "results"
sys.path.insert(0, str(APP))


def translate(clip, tag, app_argv):
    import translator as T

    audio = CLIPS / f"{clip}.wav"
    RESULTS.mkdir(exist_ok=True)
    log = RESULTS / f"{clip}_{tag}.txt"
    log.unlink(missing_ok=True)
    tr = T.Translator(T.parse_args(["--cli", "--file", str(audio), "--log", str(log)] + app_argv))
    tr.load()
    t0 = time.perf_counter()
    tr.run_file(audio)
    wall = time.perf_counter() - t0
    rows = []
    for ln in tr.lines:
        parts = [int(x) for x in ln["time"].split(":")]
        start = parts[-1] + 60 * parts[-2] + (3600 * parts[0] if len(parts) == 3 else 0)
        rows.append({"start": float(start), "end": start + ln["dur"], "en": ln["en"], "tr": ln["tr"],
                     "cost": ln["cost"]})
    out = {"clip": clip, "tag": tag, "argv": app_argv, "desc": tr.describe(), "wall": wall, "lines": rows}
    (RESULTS / f"{clip}_{tag}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{clip}_{tag}: {len(rows)} lines, {wall:.0f}s")


def parse_vtt(path):
    cues, lines = [], Path(path).read_text(encoding="utf-8").splitlines()
    ts = re.compile(r"(\d+):(\d+):(\d+)\.(\d+)\s+-->\s+(\d+):(\d+):(\d+)\.(\d+)")
    i = 0
    while i < len(lines):
        m = ts.match(lines[i])
        i += 1
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        text = []
        while i < len(lines) and lines[i].strip():
            text.append(lines[i].strip())
            i += 1
        t = " ".join(text)
        if not re.match(r"^(Translation|Translator|Reviewer|Çeviri|Çevirmen|Gözden geçirme)\s*:", t, re.I):
            cues.append((g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000, g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000, t))
    return cues


def ref_file(clip, lang):
    for d in (CLIPS, REFS):
        if (d / f"{clip}.{lang}.vtt").exists():
            return d / f"{clip}.{lang}.vtt"
    return None


def windows(items, width, n):
    buckets = [[] for _ in range(n)]
    for s, e, t in items:
        buckets[min(n - 1, int(((s + e) / 2) // width))].append(t)
    return [" ".join(b) for b in buckets]


def norm_tr(s):
    s = s.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", s)).strip()


def score(width=30.0):
    import jiwer
    import sacrebleu

    print(f"{'result':34s} {'BLEU':>5s} {'chrF++':>7s} {'doc-chrF++':>10s} {'TR WER':>7s} {'s/line':>7s}")
    for p in sorted(RESULTS.glob("*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        en_ref = parse_vtt(ref_file(d["clip"], "en"))
        hyp = [(r["start"], r["end"], r["en"]) for r in d["lines"]]
        n = math.ceil(max([e for _, e, _ in en_ref] + [e for _, e, _ in hyp]) / width)
        R, H = windows(en_ref, width, n), windows(hyp, width, n)
        keep = [i for i in range(n) if R[i] or H[i]]
        R, H = [R[i] for i in keep], [H[i] for i in keep]
        bleu = sacrebleu.corpus_bleu(H, [R]).score
        chrf = sacrebleu.corpus_chrf(H, [R], word_order=2).score
        doc = sacrebleu.corpus_chrf([" ".join(H)], [[" ".join(R)]], word_order=2).score
        wer = ""
        tr_ref = ref_file(d["clip"], "tr")
        if tr_ref and any(r["tr"] for r in d["lines"]):
            tref = parse_vtt(tr_ref)
            Rt = windows(tref, width, n)
            Ht = windows([(r["start"], r["end"], r["tr"]) for r in d["lines"]], width, n)
            pairs = [(norm_tr(a), norm_tr(b) or "x") for a, b in zip(Rt, Ht) if a.strip()]
            wer = f"{jiwer.wer([a for a, _ in pairs], [b for _, b in pairs]) * 100:6.1f}%"
        cost = sum(r["cost"] for r in d["lines"]) / max(1, len(d["lines"]))
        print(f"{p.stem:34s} {bleu:5.1f} {chrf:7.1f} {doc:10.1f} {wer:>7s} {cost:6.2f}s")


def main():
    argv = sys.argv[1:]
    app_argv = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args(argv)
    if a.clip:
        translate(a.clip, a.tag, app_argv)
    if a.score or not a.clip:
        score()


if __name__ == "__main__":
    main()
