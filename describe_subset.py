import sys, pandas as pd
from pathlib import Path
from collections import Counter
sys.path.insert(0, "scripts/isl")
from isign_build_subset import video_id_of

d = Path(sys.argv[1] if len(sys.argv) > 1 else "data/isl")
STOP = set("""a an the and or but if of to in on at by for with from as is are was were be been
being do does did have has had will would shall should can could may might must not no i you he
she it we they this that these those there here what which who whom whose when where why how""".split())

tot, vocab, vids = 0, Counter(), set()
print(f"\n  subset in {d}\n")
for s in ["train", "dev", "test"]:
    p = d / f"ISL.{s}.corpus.csv"
    if not p.is_file():
        print(f"  {s:6s}  MISSING"); continue
    df = pd.read_csv(p, sep="|")
    v = df["name"].astype(str).map(video_id_of)
    for t in df["translation"].astype(str):
        vocab.update(w for w in t.split() if w not in STOP and len(w) > 1)
    tot += len(df); vids |= set(v)
    print(f"  {s:6s}  {len(df):6,} clips   {v.nunique():5,} videos   "
          f"{len(df)/max(1,v.nunique()):5.1f} clips/video")

m = d / "needed_videos.txt"
n_man = sum(1 for _ in open(m)) if m.is_file() else 0
print(f"\n  total          {tot:6,} clips   {len(vids):5,} videos")
print(f"  content vocab  {len(vocab):6,} word types   ({len(vocab)/max(1,tot):.2f} new types per clip)")
print(f"  manifest       {n_man:6,} lines\n")

ok = True
if len(vids) < 300:
    print("  !! too few source videos - the model will see almost no variation."); ok = False
if len(vocab) > 5000:
    print(f"  !! {len(vocab):,} word types is far past the ~3,000 that PHOENIX learns from."); ok = False
if tot and len(vocab)/tot > 1.0:
    print("  !! vocabulary is growing about as fast as the data - the task never gets easier."); ok = False
print("  looks right - proceed\n" if ok else
      "  rebuild with --keeplist (from isign_corpus_stats.py) and --max_clips_per_video 5\n")
