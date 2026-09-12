"""
iSign (Exploration-Lab/iSign v1.1) -> Sign2GPT corpus CSVs, 10k-pair subset.

WHAT THIS DOES
--------------
Reads the master `iSign_v1.1.csv`, cleans the English text, drops rows that
would poison training, samples a subset of N pairs (default 10,000) *grouped by
video_id*, splits 80/10/10 (again by video_id), and writes the pipe-separated
corpus CSVs that `dataloaders/phoenix_video_dataset.py` expects:

    name | speaker | orth | translation | start | end

It also writes PHOENIX-named copies (`PHOENIX-2014-T.{split}.corpus.csv`)
because `scripts/phoenix2014t/image_lmdb_creator.py` looks for that filename,
and a `needed_videos.txt` manifest consumed by `isign_zip_to_lmdb.py`.

WHY GROUP BY video_id
---------------------
iSign uids are `<video_id>-<segment_number>`; consecutive segments of the same
source YouTube video share signer, lighting, background and topic. Splitting
segment-wise leaks the test signer into train and inflates BLEU. The iSign
authors recommend splitting on video_id; this script enforces it.

WHY CLEAN TEXT HERE AND NOWHERE ELSE
------------------------------------
`scripts/pseudo_gloss_en.py` keys `dict_sentence` by the *exact* translation
string, and the dataloader looks the sentence up with that same string. If the
text is cleaned at any later point the keys stop matching and every clip
silently gets an EMPTY pseudo-gloss target -> stage 1 trains on nothing and
`class_f1` sits at 0. So: clean once, here, before the CSV is written.

USAGE
-----
    # 1. see what columns the release actually has
    python scripts/isl/isign_build_subset.py --csv /workspace/data/isign/iSign_v1.1.csv --inspect

    # 2. build the 10k subset
    python scripts/isl/isign_build_subset.py \
        --csv        /workspace/data/isign/iSign_v1.1.csv \
        --output_dir /workspace/Sign2GPT/data/isl \
        --max_clips  10000 \
        --seed       1
"""

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column auto-detection. The public release has been re-cut a few times, so
# rather than hard-coding names we score candidates and let --uid-col /
# --text-col override.
# ---------------------------------------------------------------------------
UID_CANDIDATES = ["uid", "id", "video_id", "clip_id", "name", "file", "filename", "video"]
TEXT_CANDIDATES = ["text", "translation", "sentence", "english", "english_text",
                   "caption", "transcript", "gloss_text"]

# Sign2GPT reads the CSV with sep="|", so a literal pipe in the text splits the
# row and shifts every later column. Same for CR/LF.
_PIPE_RE = re.compile(r"[|\r\n\t]+")
_PUNCT_RE = re.compile(r"[.,!?;:\"'`()\[\]{}<>*_~/\\]")
_SPACE_RE = re.compile(r"\s+")
# Expand contractions to real words; only the possessive 's is dropped, since
# "'s" is ambiguous (Rohit's / it is) and inventing the wrong one is worse than
# losing it. Truncating instead - "don't" -> "don" - just moves the junk class.
_CONTRACTIONS = [
    (re.compile(r"\bcan['\u2019]t\b"), "cannot"),
    (re.compile(r"\bwon['\u2019]t\b"), "will not"),
    (re.compile(r"\bshan['\u2019]t\b"), "shall not"),
    (re.compile(r"n['\u2019]t\b"), " not"),
    (re.compile(r"['\u2019]re\b"), " are"),
    (re.compile(r"['\u2019]ve\b"), " have"),
    (re.compile(r"['\u2019]ll\b"), " will"),
    (re.compile(r"['\u2019]m\b"), " am"),
    (re.compile(r"['\u2019]d\b"), " would"),
    (re.compile(r"['\u2019]s\b"), ""),
]


def clean_text(s: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Mirrors what `scripts/pseudo_gloss_en.py` will see, so pseudo-gloss keys
    line up with the dataloader's `item["translation"]` byte for byte.
    """
    s = str(s)
    s = _PIPE_RE.sub(" ", s)
    s = s.lower()
    # Resolve contractions BEFORE stripping punctuation. Otherwise "it's"
    # becomes "it s" and the orphan "s" is lemmatised into its own pseudo-gloss
    # class - "s" was the 4th most common lemma in the first trial.
    for _re, _rep in _CONTRACTIONS:
        s = _re.sub(_rep, s)
    s = _PUNCT_RE.sub(" ", s)
    s = _SPACE_RE.sub(" ", s).strip()
    return s


def pick_column(df, explicit, candidates, what):
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"[err] --{what}-col '{explicit}' not in CSV. Columns: {list(df.columns)}")
        return explicit
    lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    sys.exit(
        f"[err] Could not auto-detect the {what} column.\n"
        f"      Columns present: {list(df.columns)}\n"
        f"      Re-run with --{what}-col <name>."
    )


def video_id_of(uid: str) -> str:
    """`<video_id>-<segment>` -> `<video_id>`.

    iSign video ids are YouTube ids, which may themselves contain '-', so we
    split on the LAST hyphen and only when the tail is numeric.
    """
    uid = str(uid)
    if "-" in uid:
        head, _, tail = uid.rpartition("-")
        if tail.isdigit() and head:
            return head
    return uid


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="Path to iSign_v1.1.csv")
    p.add_argument("--output_dir", help="Where to write ISL.{train,dev,test}.corpus.csv")
    p.add_argument("--inspect", action="store_true",
                   help="Print columns + 5 sample rows and exit (do this first)")
    p.add_argument("--uid-col", dest="uid_col", default=None)
    p.add_argument("--text-col", dest="text_col", default=None)
    p.add_argument("--source-col", dest="source_col", default=None,
                   help="Optional column naming the source (ISLRTC / ISH / DEF)")
    p.add_argument("--keep-source", dest="keep_source", default=None,
                   help="If set, keep only rows whose source column contains this substring")
    p.add_argument("--max_clips", type=int, default=10000)
    p.add_argument("--min_words", type=int, default=3,
                   help="Drop translations shorter than this (default 3)")
    p.add_argument("--max_words", type=int, default=30,
                   help="Drop translations longer than this (default 30)")
    p.add_argument("--max_clips_per_video", type=int, default=0,
                   help="Cap clips taken from any one source video (0 = no cap). "
                        "The first trial drew 226 clips from just 16 videos - 14 per "
                        "video - so the model saw almost no variation in signer, "
                        "background or topic. Capping this spends the same clip budget "
                        "on far more source videos, which is the cheapest way to buy "
                        "diversity.")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--dev_frac", type=float, default=0.10)
    p.add_argument("--test_frac", type=float, default=0.10)
    args = p.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.is_file():
        sys.exit(f"[err] Not found: {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[subset] read {len(df):,} rows from {csv_path.name}")

    if args.inspect:
        print(f"\ncolumns: {list(df.columns)}\n")
        print(df.head(5).to_string())
        print("\nnull counts:")
        print(df.isna().sum().to_string())
        return

    if not args.output_dir:
        sys.exit("[err] --output_dir is required unless --inspect")

    uid_col = pick_column(df, args.uid_col, UID_CANDIDATES, "uid")
    text_col = pick_column(df, args.text_col, TEXT_CANDIDATES, "text")
    print(f"[subset] uid column  = '{uid_col}'")
    print(f"[subset] text column = '{text_col}'")

    # ---------------- filtering ----------------
    n0 = len(df)
    df = df[[uid_col, text_col] + ([args.source_col] if args.source_col else [])].copy()
    df = df.dropna(subset=[uid_col, text_col])
    print(f"[subset] dropped {n0 - len(df):,} rows with a null uid/text")

    if args.keep_source:
        if not args.source_col:
            sys.exit("[err] --keep-source needs --source-col")
        before = len(df)
        df = df[df[args.source_col].astype(str).str.contains(args.keep_source, case=False, na=False)]
        print(f"[subset] source filter '{args.keep_source}': {before:,} -> {len(df):,}")

    df["translation"] = df[text_col].map(clean_text)
    df["name"] = df[uid_col].astype(str).str.strip()

    before = len(df)
    wc = df["translation"].str.split().str.len()
    df = df[(wc >= args.min_words) & (wc <= args.max_words)]
    print(f"[subset] word-count filter [{args.min_words},{args.max_words}]: {before:,} -> {len(df):,}")

    before = len(df)
    df = df.drop_duplicates(subset=["name"])
    print(f"[subset] dropped {before - len(df):,} duplicate uids")

    df["video_id"] = df["name"].map(video_id_of)
    print(f"[subset] {df['video_id'].nunique():,} distinct source videos remain")

    if args.max_clips_per_video:
        before = len(df)
        df = (df.sample(frac=1.0, random_state=args.seed)
                .groupby("video_id", group_keys=False)
                .head(args.max_clips_per_video)
                .sort_index())
        print(f"[subset] <={args.max_clips_per_video} clips per video: "
              f"{before:,} -> {len(df):,} clips across {df['video_id'].nunique():,} videos")

    # ---------------- sample N clips, whole videos at a time ----------------
    rng = np.random.default_rng(args.seed)
    vids = np.array(df["video_id"].unique().tolist(), dtype=object)
    rng.shuffle(vids)

    counts = df.groupby("video_id").size().to_dict()
    chosen, total = [], 0
    for v in vids:
        if total >= args.max_clips:
            break
        chosen.append(v)
        total += counts[v]
    sub = df[df["video_id"].isin(set(chosen))].copy()

    # Trim the overshoot from the LAST video only, so no video is half-used
    # across the boundary between "in the subset" and "not".
    if len(sub) > args.max_clips:
        last = chosen[-1]
        keep_from_last = args.max_clips - (len(sub) - counts[last])
        if keep_from_last <= 0:
            sub = sub[sub["video_id"] != last]
        else:
            head = sub[sub["video_id"] != last]
            tail = sub[sub["video_id"] == last].head(keep_from_last)
            sub = pd.concat([head, tail], ignore_index=True)
    print(f"[subset] sampled {len(sub):,} clips from {sub['video_id'].nunique():,} videos")

    # ---------------- 80/10/10 split by video_id ----------------
    vids = np.array(sub["video_id"].unique().tolist(), dtype=object)
    rng.shuffle(vids)
    n_dev = max(1, int(round(len(vids) * args.dev_frac)))
    n_test = max(1, int(round(len(vids) * args.test_frac)))
    dev_v = set(vids[:n_dev])
    test_v = set(vids[n_dev:n_dev + n_test])

    def split_of(v):
        return "dev" if v in dev_v else ("test" if v in test_v else "train")

    sub["split"] = sub["video_id"].map(split_of)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = []
    for split in ["train", "dev", "test"]:
        d = sub[sub["split"] == split]
        out = pd.DataFrame({
            "name": d["name"].values,
            "speaker": "Signer01",           # iSign has no signer labels; unused downstream
            # NOTE: `orth` must be a non-empty STRING. The dataloader always calls
            # item["orth"].split(" "), and pandas turns an empty field into NaN
            # (float) -> AttributeError on the first batch. Gloss-free training
            # never reads the value, so we mirror the translation.
            "orth": d["translation"].values,
            "translation": d["translation"].values,
            "start": -1,
            "end": -1,
        })
        path = out_dir / f"ISL.{split}.corpus.csv"
        out.to_csv(path, sep="|", index=False)

        # image_lmdb_creator.py (PHOENIX one) hard-codes this filename.
        alias = out_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        out.to_csv(alias, sep="|", index=False)

        manifest.extend(f"{split}\t{n}" for n in d["name"].values)
        print(f"[subset] {split:5s}: {len(out):5,} clips  ({d['video_id'].nunique():,} videos)  -> {path.name}")

    man_path = out_dir / "needed_videos.txt"
    man_path.write_text("\n".join(manifest) + "\n")
    print(f"[subset] wrote manifest {man_path} ({len(manifest):,} entries)")

    # Sanity: no video_id may appear in two splits.
    leak = sub.groupby("video_id")["split"].nunique()
    assert (leak == 1).all(), "video_id leaked across splits"
    print("[subset] OK - no video_id appears in more than one split")


if __name__ == "__main__":
    main()
