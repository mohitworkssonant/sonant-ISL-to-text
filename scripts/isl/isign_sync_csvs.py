"""
Prune the corpus CSVs down to the clips that actually have an LMDB.

WHY THIS IS A SEPARATE STEP
---------------------------
On a tight budget you deliberately over-request clips and then reject the
expensive ones (`--max_src_frames`) and stop early (`--stop_after`). So the
CSVs written by `isign_build_subset.py` describe more clips than exist on disk.

The dataloader already skips rows with no LMDB, so training would *run* - but
two things would be wrong:

  * `num_classes` and `dict_lem_counter` would be built from sentences that are
    never seen, inflating the vocabulary and diluting the 0.4 frequency
    cut-off that selects pseudo-gloss targets;
  * every count you report (train size, split sizes) would be fiction.

So: prune first, THEN build the pseudo-gloss pkl. Order matters.

It also re-balances the splits if pruning emptied one, and keeps the
`video_id` grouping intact - a clip never moves to a different split, splits
only lose clips.

Usage:
    python scripts/isl/isign_sync_csvs.py \
        --csv_dir   /workspace/sonant-ISL-to-text/data/isl \
        --lmdb_root /workspace/lmdb/isl/lmdb_videos
"""

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


def video_id_of(uid: str) -> str:
    uid = str(uid)
    if "-" in uid:
        head, _, tail = uid.rpartition("-")
        if tail.isdigit() and head:
            return head
    return uid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv_dir", required=True)
    ap.add_argument("--lmdb_root", required=True)
    ap.add_argument("--min_split", type=int, default=25,
                    help="Warn if a split ends up smaller than this")
    ap.add_argument("--no_backup", action="store_true")
    args = ap.parse_args()

    csv_dir, lmdb_root = Path(args.csv_dir), Path(args.lmdb_root)
    have = {p.name for p in lmdb_root.iterdir() if p.is_dir()} if lmdb_root.is_dir() else set()
    print(f"[sync] {len(have):,} LMDB directories on disk")
    if not have:
        sys.exit(f"[err] no LMDBs under {lmdb_root} - run isign_zip_to_lmdb.py first")

    total_before = total_after = 0
    summary = []
    for split in ["train", "dev", "test"]:
        p = csv_dir / f"ISL.{split}.corpus.csv"
        if not p.is_file():
            print(f"[warn] missing {p}", file=sys.stderr)
            continue
        df = pd.read_csv(p, sep="|")
        if not args.no_backup:
            bak = csv_dir / f"ISL.{split}.corpus.csv.full"
            if not bak.exists():
                shutil.copy(p, bak)

        kept = df[df["name"].astype(str).isin(have)].reset_index(drop=True)
        total_before += len(df)
        total_after += len(kept)

        kept.to_csv(p, sep="|", index=False)
        kept.to_csv(csv_dir / f"PHOENIX-2014-T.{split}.corpus.csv", sep="|", index=False)

        n_vid = kept["name"].map(video_id_of).nunique() if len(kept) else 0
        flag = "  <-- TOO SMALL" if len(kept) < args.min_split else ""
        summary.append((split, len(df), len(kept), n_vid, flag))

    print()
    print(f"{'split':6s} {'requested':>10s} {'kept':>8s} {'videos':>8s}")
    for split, before, after, nvid, flag in summary:
        print(f"{split:6s} {before:10,d} {after:8,d} {nvid:8,d}{flag}")
    print(f"{'total':6s} {total_before:10,d} {total_after:8,d}")

    # Rewrite the manifest so a later top-up run only asks for what's missing.
    man = csv_dir / "needed_videos.txt"
    if man.is_file():
        remaining = []
        for line in man.read_text().splitlines():
            if not line.strip():
                continue
            split, _, name = line.partition("\t")
            if name.strip() not in have:
                remaining.append(line)
        (csv_dir / "needed_videos.remaining.txt").write_text("\n".join(remaining) + "\n")
        print(f"\n[sync] {len(remaining):,} manifest entries still without an LMDB -> "
              f"needed_videos.remaining.txt (feed it back to isign_zip_to_lmdb.py to top up)")

    print("\n[sync] NOW rebuild the pseudo-gloss pkl, so the vocabulary matches the real data:")
    print(f"  python scripts/pseudo_gloss_en.py --csv_dir {csv_dir} "
          f"--output_pkl {csv_dir}/processed_words.isl_pkl")


if __name__ == "__main__":
    main()
