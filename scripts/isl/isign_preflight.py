"""
Preflight check: catch the five failure modes that waste a GPU day.

Run this after the corpus CSVs, the pseudo-gloss pkl and the LMDBs exist, and
BEFORE `python main.py`. Every check below corresponds to a real way this
pipeline fails *silently* (trains happily, learns nothing) rather than loudly.

    1. CSV shape       - 6 pipe-separated columns, no NaN in `orth`
                         (dataloader calls item["orth"].split(" ") unguarded)
    2. LMDB coverage   - how many CSV rows actually have an LMDB directory
    3. LMDB integrity  - "details" key readable, num_frames sane, frames decode
    4. pkl key match   - EVERY translation string must be a key in dict_sentence,
                         otherwise pseudo_gloss_ids comes back empty and stage 1
                         optimises a constant. This is the #1 silent killer.
    5. label density   - mean pseudo-glosses per clip after the 0.4 frequency
                         cut-off, and the resulting num_classes

Usage:
    python scripts/isl/isign_preflight.py \
        --csv_dir   /workspace/Sign2GPT/data/isl \
        --pkl       /workspace/Sign2GPT/data/isl/processed_words.isl_pkl \
        --lmdb_root /workspace/lmdb/isl/lmdb_videos
"""

import argparse
import io
import pickle
import sys
from pathlib import Path

import lmdb
import numpy as np
import pandas as pd
from PIL import Image

EXPECTED_COLS = ["name", "speaker", "orth", "translation", "start", "end"]


def fail(msg):
    print(f"  \033[1;31mFAIL\033[0m  {msg}")
    return 1


def ok(msg):
    print(f"  \033[1;32mok\033[0m    {msg}")
    return 0


def warn(msg):
    print(f"  \033[1;33mwarn\033[0m  {msg}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--pkl", required=True)
    ap.add_argument("--lmdb_root", required=True)
    ap.add_argument("--sample", type=int, default=25, help="LMDBs to open and decode")
    args = ap.parse_args()

    csv_dir, lmdb_root = Path(args.csv_dir), Path(args.lmdb_root)
    errors = 0
    dfs = {}

    print("\n[1] corpus CSVs")
    for split in ["train", "dev", "test"]:
        p = csv_dir / f"ISL.{split}.corpus.csv"
        if not p.is_file():
            errors += fail(f"missing {p}")
            continue
        df = pd.read_csv(p, sep="|")
        dfs[split] = df
        if list(df.columns) != EXPECTED_COLS:
            errors += fail(f"{p.name}: columns {list(df.columns)} != {EXPECTED_COLS}")
        elif df["orth"].isna().any():
            errors += fail(f"{p.name}: {int(df['orth'].isna().sum())} NaN in `orth` "
                           f"-> dataloader will crash on .split(' ')")
        elif df["translation"].isna().any():
            errors += fail(f"{p.name}: NaN in `translation`")
        elif df["name"].duplicated().any():
            errors += fail(f"{p.name}: duplicate `name` values")
        else:
            ok(f"{p.name}: {len(df):,} rows, columns fine")
        alias = csv_dir / f"PHOENIX-2014-T.{split}.corpus.csv"
        if not alias.is_file():
            warn(f"{alias.name} absent (only needed by the PHOENIX LMDB creator)")

    if not dfs:
        print("\nNo CSVs read - stopping.")
        return 1

    print("\n[2] LMDB coverage")
    total_missing = 0
    for split, df in dfs.items():
        have = df["name"].map(lambda n: (lmdb_root / str(n)).is_dir())
        n_missing = int((~have).sum())
        total_missing += n_missing
        pct = 100.0 * have.mean() if len(df) else 0.0
        line = f"{split:5s}: {int(have.sum()):,}/{len(df):,} clips have an LMDB ({pct:.1f}%)"
        if pct < 50:
            errors += fail(line)
        elif n_missing:
            warn(line + "  (missing rows are auto-filtered at load time)")
        else:
            ok(line)
    if total_missing:
        print(f"        {total_missing:,} rows will be dropped by the dataloader's LMDB filter.")

    print("\n[3] LMDB integrity")
    names = pd.concat(dfs.values())["name"].astype(str).tolist()
    present = [n for n in names if (lmdb_root / n).is_dir()]
    if not present:
        errors += fail("no LMDB directories at all")
    else:
        rng = np.random.default_rng(0)
        pick = rng.choice(present, size=min(args.sample, len(present)), replace=False)
        frame_counts, bad = [], 0
        for n in pick:
            try:
                env = lmdb.open(str(lmdb_root / n), readonly=True, lock=False,
                                readahead=False, meminit=False)
                with env.begin(write=False) as txn:
                    det = pickle.loads(txn.get(b"details"))
                    nf = det["num_frames"]
                    blob = txn.get(b"0")
                    img = Image.open(io.BytesIO(blob))
                    if img.size != (img.size[0], img.size[0]):
                        bad += 1
                    last = txn.get(f"{nf - 1}".encode("ascii"))
                    if last is None:
                        bad += 1
                frame_counts.append(nf)
                env.close()
            except Exception as e:                       # noqa: BLE001
                bad += 1
                print(f"        {n}: {type(e).__name__}: {e}")
        if bad:
            errors += fail(f"{bad}/{len(pick)} sampled LMDBs are damaged")
        else:
            fc = np.array(frame_counts)
            ok(f"{len(pick)} sampled LMDBs readable; frames min={fc.min()} "
               f"median={int(np.median(fc))} max={fc.max()} "
               f"(after stride-2 -> ~{int(np.median(fc)) // 2} tokens)")
            if np.median(fc) // 2 > 256:
                warn("median token count exceeds max_seq_len=256; frames will be randomly "
                     "subsampled - consider raising max_seq_len or lowering target_fps")

    print("\n[4] pseudo-gloss pkl <-> CSV key match")
    try:
        with open(args.pkl, "rb") as f:
            pg = pickle.load(f)
        dict_sentence = pg["dict_sentence"]
        dict_lem_to_id = pg["dict_lem_to_id"]
        dict_lem_counter = pg["dict_lem_counter"]
    except Exception as e:                               # noqa: BLE001
        print(f"  \033[1;31mFAIL\033[0m  cannot read {args.pkl}: {e}")
        return errors + 1

    for split, df in dfs.items():
        sents = df["translation"].astype(str)
        miss = int((~sents.isin(dict_sentence.keys())).sum())
        if miss:
            errors += fail(f"{split}: {miss:,}/{len(df):,} translations are NOT keys in "
                           f"dict_sentence -> those clips get EMPTY pseudo-gloss targets")
            for s in sents[~sents.isin(dict_sentence.keys())].head(3):
                print(f"        e.g. {s!r}")
        else:
            ok(f"{split}: all {len(df):,} translations found in dict_sentence")

    print("\n[5] label density / num_classes")
    n_classes = len(dict_lem_to_id)
    n_sent = len(dict_sentence)
    dense = []
    for s in pd.concat(dfs.values())["translation"].astype(str):
        lems = dict_sentence.get(s, [])
        kept = [l for l in lems if dict_lem_counter[l] / n_sent < 0.4]
        dense.append(len(kept))
    dense = np.array(dense)
    print(f"        num_classes (unique lemmas) = {n_classes:,}")
    print(f"        pseudo-glosses per clip: mean={dense.mean():.2f} "
          f"median={int(np.median(dense))} zero-label clips={int((dense == 0).sum()):,}")
    if dense.mean() < 1.0:
        errors += fail("mean pseudo-gloss count < 1 - stage 1 has almost no signal")
    elif (dense == 0).mean() > 0.05:
        warn(f"{100 * (dense == 0).mean():.1f}% of clips have zero pseudo-gloss labels")
    else:
        ok("label density looks healthy")
    if n_classes > 20000:
        warn(f"{n_classes:,} classes is large; the prototype head allocates a "
             f"{n_classes}x300 embedding - fine on 80 GB, watch it on smaller cards")

    print()
    if errors:
        print(f"\033[1;31m{errors} blocking problem(s). Fix before training.\033[0m\n")
        return 1
    print("\033[1;32mPreflight passed - safe to start stage 1.\033[0m\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
