"""
Pull a few named clips out of the iSign zip and write them to disk as MP4.

The LMDBs hold JPEG frames, not playable video, so a demo or an inference check
needs the original file. This grabs just those members - by HTTP range from
Hugging Face, or from downloaded parts - without touching the rest of the 58 GB.

Usage:
    # 3 clips from the test split, straight from HF
    python scripts/isl/isign_fetch_clip.py \
        --hf_repo Exploration-Lab/iSign \
        --from_csv /workspace/sonant-ISL-to-text/data/isl/ISL.test.corpus.csv \
        --n 3 --out_dir /workspace/trial_clips

    # named clips, from downloaded parts
    python scripts/isl/isign_fetch_clip.py \
        --zip_parts /data/iSign-videos_v1.1_part_a* \
        --clips abc123-4 def456-7 --out_dir /workspace/trial_clips
"""

import argparse
import io
import os
import shutil
import sys
import zipfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isign_http_zip import MultiPartHttpFile, hf_urls          # noqa: E402
from isign_zip_to_lmdb import MultiPartFile, VIDEO_EXTS        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_parts", nargs="+",
                    default=["iSign-videos_v1.1_part_aa", "iSign-videos_v1.1_part_ab"])
    ap.add_argument("--hf_revision", default="main")
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--zip_parts", nargs="+", default=None)
    ap.add_argument("--clips", nargs="+", default=None, help="Explicit clip ids")
    ap.add_argument("--from_csv", default=None, help="Take clip ids from this corpus CSV")
    ap.add_argument("--n", type=int, default=3, help="How many to take from --from_csv")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    if bool(args.hf_repo) == bool(args.zip_parts):
        sys.exit("[err] give exactly one of --hf_repo or --zip_parts")

    wanted = list(args.clips or [])
    if args.from_csv:
        df = pd.read_csv(args.from_csv, sep="|")
        wanted += df["name"].astype(str).head(args.n).tolist()
    if not wanted:
        sys.exit("[err] nothing to fetch: give --clips or --from_csv")

    if args.hf_repo:
        raw = MultiPartHttpFile(hf_urls(args.hf_repo, args.hf_parts, args.hf_revision),
                                args.hf_token)
    else:
        raw = MultiPartFile(args.zip_parts)
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))

    index = {}
    for name in zf.namelist():
        p = Path(name)
        if p.suffix.lower() in VIDEO_EXTS:
            index[p.stem] = name
    print(f"[fetch] archive holds {len(index):,} videos")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    got = 0
    for clip in wanted:
        member = index.get(clip)
        if member is None:
            print(f"[fetch] NOT FOUND: {clip}", file=sys.stderr)
            continue
        dst = out_dir / Path(member).name
        with zf.open(member) as src, open(dst, "wb") as fh:
            shutil.copyfileobj(src, fh, length=1 << 20)
        print(f"[fetch] {dst}  ({dst.stat().st_size / 1e6:.1f} MB)")
        got += 1
    print(f"[fetch] wrote {got}/{len(wanted)} clips to {out_dir}")


if __name__ == "__main__":
    main()
