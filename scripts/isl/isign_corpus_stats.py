"""
What is actually in iSign's text, and how big a subset you need.

Runs on your laptop against iSign_v1.1.csv. No GPU, no downloads, ~1 minute.
It answers the two questions that decide the next run:

  1. How much of the corpus is junk, boilerplate or duplicate - i.e. what is
     there for a quality filter to remove?

  2. How does the VOCABULARY grow with the number of clips? This is the number
     that governs whether a run can learn anything, and it is usually ignored.
     Stage 1 is a multi-label classification over every distinct word stem in
     the training set. Add clips and you add examples, but you also add
     classes. If vocabulary grows as fast as data, the task never gets easier.

     PHOENIX-2014T reaches BLEU-4 13 with ~7,000 training pairs because it is
     weather forecasts: ~3,000 word types, one topic, nine signers. iSign at
     full 127,000 pairs scores BLEU-4 0.24-0.56 in its own paper, because it is
     open-domain YouTube with ~40,000 word types. More data did not save it.
     So the useful question is not "how many clips" but "how many clips at what
     vocabulary size".

  It then reports, for a range of vocabulary caps, how many clips are fully
  covered by that vocabulary - which is how you build a PHOENIX-shaped subset
  out of an open-domain corpus.

Usage:
    python scripts/isl/isign_corpus_stats.py --csv iSign_v1.1.csv
    python scripts/isl/isign_corpus_stats.py --csv iSign_v1.1.csv \
        --write_keeplist coherent_8k.txt --vocab_cap 2000 --min_coverage 1.0
"""

import argparse
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isign_build_subset import clean_text, video_id_of      # noqa: E402

# Rows that are not transcriptions of a signed utterance. These come from
# educational material where the on-screen text was captured verbatim.
JUNK_PATTERNS = [
    (re.compile(r"^(page|chapter|lesson|unit|exercise|figure|table|slide)\s*\d*$"), "heading"),
    (re.compile(r"^\W*\d[\d\s\.\-/]*$"), "numbers only"),
    (re.compile(r"^(introduction|conclusion|index|contents|summary|the end|thank you)$"), "boilerplate"),
    (re.compile(r"^\W*$"), "empty after cleaning"),
]

STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being do does did have has had will would shall should can could may might must not no i you he
she it we they this that these those there here what which who whom whose when where why how""".split())


def content_words(text):
    return [w for w in text.split() if w not in STOP and len(w) > 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--min_words", type=int, default=3)
    ap.add_argument("--max_words", type=int, default=15)
    ap.add_argument("--vocab_cap", type=int, default=None,
                    help="With --write_keeplist: restrict to this many most-frequent content words")
    ap.add_argument("--min_coverage", type=float, default=1.0,
                    help="Fraction of a clip's content words that must be inside the capped "
                         "vocabulary for the clip to be kept (1.0 = all of them)")
    ap.add_argument("--write_keeplist", default=None,
                    help="Write the surviving clip ids here, one per line")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    uid = next(c for c in df.columns if c.lower() in ("uid", "id", "clip_id", "name"))
    txt = next(c for c in df.columns if c.lower() in ("text", "translation", "sentence"))
    n0 = len(df)
    print(f"\n{'='*66}\n  iSign corpus: {n0:,} rows\n{'='*66}")

    df = df.dropna(subset=[uid, txt]).copy()
    df["clean"] = df[txt].map(clean_text)
    df["video_id"] = df[uid].astype(str).map(video_id_of)
    df["nw"] = df["clean"].str.split().str.len()

    # ---------------------------------------------------------------- junk
    print("\n1. What a text filter could remove\n")
    reasons = Counter()
    flags = np.zeros(len(df), dtype=bool)
    for i, t in enumerate(df["clean"].values):
        for pat, why in JUNK_PATTERNS:
            if pat.match(t):
                reasons[why] += 1
                flags[i] = True
                break
    dupes = df["clean"].duplicated(keep="first")
    rep = df["clean"].value_counts()
    boiler = set(rep[rep >= 20].index)                 # same sentence 20+ times
    boiler_n = int(df["clean"].isin(boiler).sum())

    short = int((df["nw"] < a.min_words).sum())
    long_ = int((df["nw"] > a.max_words).sum())
    for why, n in reasons.most_common():
        print(f"   {why:24s} {n:7,}  ({100*n/n0:.1f}%)")
    print(f"   {'exact duplicate text':24s} {int(dupes.sum()):7,}  ({100*dupes.sum()/n0:.1f}%)")
    print(f"   {'repeated >=20x (boiler)':24s} {boiler_n:7,}  ({100*boiler_n/n0:.1f}%)")
    print(f"   {f'under {a.min_words} words':24s} {short:7,}  ({100*short/n0:.1f}%)")
    print(f"   {f'over {a.max_words} words':24s} {long_:7,}  ({100*long_/n0:.1f}%)")

    keep = (~flags) & (~dupes.values) & (df["nw"] >= a.min_words).values & (df["nw"] <= a.max_words).values
    work = df[keep].copy()
    print(f"\n   survives all of the above: {len(work):,}  ({100*len(work)/n0:.1f}% of the corpus)")
    print(f"   distinct source videos:    {work['video_id'].nunique():,}")
    print(f"   clips per video: median {int(work.groupby('video_id').size().median())}, "
          f"max {int(work.groupby('video_id').size().max())}")

    # ------------------------------------------------- vocabulary growth
    print("\n2. Vocabulary growth - the number that decides how hard the task is\n")
    rng = np.random.default_rng(a.seed)
    order = rng.permutation(len(work))
    texts = work["clean"].values[order]
    print(f"   {'clips':>8s}  {'word types':>11s}  {'new types/clip':>15s}")
    seen, rows = set(), []
    marks = [500, 1000, 2000, 5000, 10000, 20000, 50000, len(texts)]
    marks = sorted({m for m in marks if m <= len(texts)})
    mi, cnt = 0, 0
    for t in texts:
        cnt += 1
        seen.update(content_words(t))
        if mi < len(marks) and cnt == marks[mi]:
            rows.append((cnt, len(seen)))
            prev = rows[-2] if len(rows) > 1 else (0, 0)
            rate = (len(seen) - prev[1]) / max(1, cnt - prev[0])
            print(f"   {cnt:8,}  {len(seen):11,}  {rate:15.2f}")
            mi += 1

    print("\n   For comparison: PHOENIX-2014T is ~3,000 word types over ~7,000 training")
    print("   pairs - about 0.4 new types per clip - and reaches BLEU-4 13.")

    # -------------------------------------------- coherent-subset curve
    print("\n3. Restricting the vocabulary: how many clips stay fully covered\n")
    freq = Counter()
    for t in work["clean"].values:
        freq.update(content_words(t))
    print(f"   total content-word types in the filtered corpus: {len(freq):,}")
    print(f"\n   {'vocab cap':>10s}  {'clips fully covered':>20s}  {'% of filtered':>14s}")
    cw = [content_words(t) for t in work["clean"].values]
    for cap in [500, 1000, 1500, 2000, 3000, 5000, 8000]:
        vocab = {w for w, _ in freq.most_common(cap)}
        n = sum(1 for words in cw if words and all(w in vocab for w in words))
        print(f"   {cap:10,}  {n:20,}  {100*n/len(work):13.1f}%")
    print("\n   Read this as: 'a subset of N clips whose entire content vocabulary is")
    print("   V words'. A PHOENIX-shaped problem is roughly V=3,000 with N=7,000.")

    # ----------------------------------------------------------- keeplist
    if a.write_keeplist:
        if not a.vocab_cap:
            sys.exit("[err] --write_keeplist needs --vocab_cap")
        vocab = {w for w, _ in freq.most_common(a.vocab_cap)}
        sel = []
        for (i, words) in enumerate(cw):
            if not words:
                continue
            cov = sum(1 for w in words if w in vocab) / len(words)
            if cov >= a.min_coverage:
                sel.append(work[uid].values[i])
        Path(a.write_keeplist).write_text("\n".join(map(str, sel)) + "\n")
        vids = pd.Series([video_id_of(str(s)) for s in sel]).nunique()
        print(f"\n4. Wrote {len(sel):,} clip ids to {a.write_keeplist}")
        print(f"   vocabulary cap {a.vocab_cap:,} | coverage >= {a.min_coverage:.0%} | "
              f"{vids:,} distinct videos")
        print("   Feed it to the subset builder:")
        print(f"     python scripts/isl/isign_build_subset.py --csv {a.csv} \\")
        print(f"         --keeplist {a.write_keeplist} --output_dir data/isl \\")
        print(f"         --max_clips 8000 --max_clips_per_video 5")
    print()


if __name__ == "__main__":
    main()
