# Translation quality benchmark

`evaluate.py` runs the app's own file mode (`translator.py --file`) on a test clip and scores
the English against human subtitles, so every number reflects exactly what the app does.

```
eval\get_clips.cmd                                   # once: downloads the clips into eval\clips
.venv\Scripts\python eval\evaluate.py tedx --tag file            # app defaults for --file
.venv\Scripts\python eval\evaluate.py tedx --tag live -- --model medium --audio-ctx auto --beam 1
.venv\Scripts\python eval\evaluate.py --score                    # table of everything in eval\results
```

Needs `.venv\Scripts\python -m pip install sacrebleu jiwer`. The `alp` English reference
(a transcription of the clip's burned-in subtitles) is not included, so only `tedx` scores out of the box.

* **BLEU / chrF++** are computed over 30 s windows, so a system is not penalised for splitting
  lines differently from the subtitles; **doc-chrF++** compares the whole transcript at once.
  Both reward wording close to the reference, which favours literal, subtitle-sized
  translations: small differences (< 1 point) are noise, read the lines too.
* **TR WER** is the Turkish word error rate against the human Turkish subtitles (tedx only).
  TED subtitles are lightly edited, so even a perfect recogniser would not reach 0.
* `alp` is a hard case (shouting over music and battle noise, 41 subtitle lines); `tedx` is
  closer to a meeting (one speaker, 168 lines).
