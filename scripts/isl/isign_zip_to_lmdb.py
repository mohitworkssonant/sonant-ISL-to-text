"""
iSign video zip -> per-clip LMDB, in one pass, without unzipping 58 GB.

WHY THIS SCRIPT EXISTS
----------------------
The stock path is  mp4 -> PNG frames on disk -> image_lmdb_creator.py -> LMDB.
For PHOENIX (7.8k short clips) that is fine. For iSign it is not:

    10,000 clips x ~215 frames x ~120 KB per 256x256 PNG  ~=  250 GB

of intermediate files that get deleted straight after. Writing JPEG bytes
directly into the LMDB - which is exactly what the dataloader reads back -
costs ~40 GB and one pass instead of three. The PNG stage buys nothing: the
LMDB creator re-encodes to JPEG q90 anyway.

It reads the MP4s *out of the split zip* via a virtual concatenation of
`iSign-videos_v1.1_part_aa` + `part_ab`, so you never need the extra 58 GB for
`cat part_* > videos.zip`, nor the ~50 GB of extracted MP4s.

With `--hf_repo` it goes further and never downloads the archive at all: it
reads the zip's central directory and each wanted member by HTTP range request
straight from Hugging Face. A 3,000-clip subset touches ~2.5% of the 58 GB, so
this turns a 58 GB download + 80 GB volume into ~1.5 GB of traffic and a 40 GB
volume. See scripts/isl/isign_http_zip.py.

WHAT IT PRODUCES
----------------
    <lmdb_root>/<clip_name>/{data.mdb,lock.mdb}

with keys "0".."N-1" holding JPEG bytes and a "details" key holding
pickle({"num_frames": N, "id": clip_name}) - byte-identical in structure to
what scripts/phoenix2014t/image_lmdb_creator.py writes, so the dataloader,
augmentation and configs need no changes.

USAGE
-----
    python scripts/isl/isign_zip_to_lmdb.py \
        --zip_parts  /workspace/data/isign/iSign-videos_v1.1_part_aa \
                     /workspace/data/isign/iSign-videos_v1.1_part_ab \
        --manifest   /workspace/Sign2GPT/data/isl/needed_videos.txt \
        --lmdb_root  /workspace/lmdb/isl/lmdb_videos \
        --target_fps 25 --size 256 --workers 8

Already-unzipped instead? Use --videos_dir /path/to/videos (recursively
searched) and drop --zip_parts.
"""

import argparse
import io
import os
import pickle
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import lmdb
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isign_http_zip import MultiPartHttpFile, hf_urls  # noqa: E402

N_BYTES = 2 ** 34          # 16 GB map per clip-LMDB; LMDB is sparse, costs nothing
COMMIT_EVERY = 100
MIN_VALID_FRAMES = 16      # same floor as the PHOENIX creator
VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


# ---------------------------------------------------------------------------
# Virtual concatenation of the split-zip parts
# ---------------------------------------------------------------------------
class MultiPartFile(io.RawIOBase):
    """A read-only, seekable file object spanning several files in order.

    `zipfile` needs random access (the central directory lives at the end), so
    a pipe won't do - but it only ever calls read/seek/tell, which is cheap to
    implement across N handles. This is what lets us skip
    `cat part_aa part_ab > videos.zip`.
    """

    def __init__(self, paths):
        self.paths = [str(p) for p in paths]
        self.sizes = [os.path.getsize(p) for p in self.paths]
        self.offsets, acc = [], 0
        for s in self.sizes:
            self.offsets.append(acc)
            acc += s
        self.total = acc
        self._fh = [open(p, "rb") for p in self.paths]
        self._pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self._pos

    def seek(self, offset, whence=os.SEEK_SET):
        if whence == os.SEEK_SET:
            self._pos = offset
        elif whence == os.SEEK_CUR:
            self._pos += offset
        else:
            self._pos = self.total + offset
        self._pos = max(0, min(self._pos, self.total))
        return self._pos

    def readinto(self, b):
        want = len(b)
        if want == 0 or self._pos >= self.total:
            return 0
        got = 0
        while got < want and self._pos < self.total:
            i = max(j for j in range(len(self.paths)) if self.offsets[j] <= self._pos)
            local = self._pos - self.offsets[i]
            self._fh[i].seek(local)
            chunk = self._fh[i].read(min(want - got, self.sizes[i] - local))
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got

    def close(self):
        for f in self._fh:
            try:
                f.close()
            except Exception:
                pass
        super().close()


# ---------------------------------------------------------------------------
# Frame handling
# ---------------------------------------------------------------------------
def pad_to_square(bgr):
    """Letterbox to a square with edge-replicated borders.

    A plain cv2.resize to 256x256 squashes a 16:9 frame ~1.8x horizontally.
    DINOv2 was pretrained on undistorted images, and handshape is the entire
    signal here, so we pad instead of squash. Replicate (not black) borders
    keep the background statistics closer to the real frame.
    """
    h, w = bgr.shape[:2]
    if h == w:
        return bgr
    if w > h:
        pad = w - h
        top, bottom, left, right = pad // 2, pad - pad // 2, 0, 0
    else:
        pad = h - w
        top, bottom, left, right = 0, 0, pad // 2, pad - pad // 2
    return cv2.copyMakeBorder(bgr, top, bottom, left, right, cv2.BORDER_REPLICATE)


def video_to_jpegs(video_path, target_fps, size, max_frames, jpeg_quality=90,
                   max_src_frames=0):
    """Decode -> fps-resample -> pad -> resize -> JPEG bytes. Streaming.

    `max_src_frames` rejects long clips before decoding them. On a fixed GPU
    budget, training cost is proportional to total tokens, and long clips are
    both the most expensive and the hardest (more signs to align per sentence).
    Spending the budget on short utterances buys more clips per rupee and an
    easier learning problem.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None, "could_not_open"

    if max_src_frames:
        n_declared = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        # The header count is a hint, not a guarantee - we still enforce the
        # cap during decode below - but when it is present it saves the decode.
        if n_declared and n_declared > max_src_frames * 1.05:
            cap.release()
            return None, f"too_long_{int(n_declared)}f"

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1 or src_fps > 240:
        src_fps = 25.0
    step = max(1.0, src_fps / float(target_fps))

    out, i, next_wanted = [], 0, 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if max_src_frames and i > max_src_frames:
            cap.release()
            return None, f"too_long_{i}f"
        if i >= next_wanted:
            next_wanted += step
            bgr = pad_to_square(bgr)
            if bgr.shape[0] != size:
                interp = cv2.INTER_AREA if bgr.shape[0] > size else cv2.INTER_LINEAR
                bgr = cv2.resize(bgr, (size, size), interpolation=interp)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="jpeg", quality=jpeg_quality)
            out.append(buf.getvalue())
            if max_frames and len(out) >= max_frames:
                break
        i += 1
    cap.release()
    return out, None


def write_lmdb(lmdb_dir: Path, clip_id: str, jpegs):
    """Write one clip-LMDB atomically (tmp dir, then move)."""
    tmp = Path(tempfile.mkdtemp(prefix="lmdb_"))
    env = lmdb.open(path=str(tmp), map_size=N_BYTES)
    txn = env.begin(write=True)
    for idx, blob in enumerate(jpegs):
        txn.put(key=f"{idx}".encode("ascii"), value=blob, dupdata=False)
        if (idx + 1) % COMMIT_EVERY == 0:
            txn.commit()
            txn = env.begin(write=True)
    txn.put(key=b"details",
            value=pickle.dumps({"num_frames": len(jpegs), "id": clip_id}, protocol=4),
            dupdata=False)
    txn.commit()
    env.close()
    lmdb_dir.parent.mkdir(parents=True, exist_ok=True)
    if lmdb_dir.exists():
        shutil.rmtree(lmdb_dir, ignore_errors=True)
    shutil.move(str(tmp), str(lmdb_dir))


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
_ZIP = None      # per-process zipfile handle
_INDEX = None    # stem -> member name


def _open_archive(source):
    """source = ('local', [paths]) | ('http', [urls], token) | ('dir', None)"""
    kind = source[0]
    if kind == "local":
        return zipfile.ZipFile(io.BufferedReader(MultiPartFile(source[1]), buffer_size=1 << 20))
    if kind == "http":
        return zipfile.ZipFile(io.BufferedReader(MultiPartHttpFile(source[1], source[2]),
                                                 buffer_size=1 << 20))
    return None


def _init_worker(source, index):
    global _ZIP, _INDEX
    _INDEX = index
    _ZIP = _open_archive(source)


def _process(task):
    (clip_id, lmdb_root, videos_dir, target_fps, size, max_frames, overwrite,
     max_src_frames) = task
    out_dir = Path(lmdb_root) / clip_id
    if out_dir.is_dir() and not overwrite:
        return ("skip", clip_id, 0)

    tmp_video = None
    try:
        if _ZIP is not None:
            member = _INDEX.get(clip_id)
            if member is None:
                return ("missing", clip_id, 0)
            suffix = Path(member).suffix or ".mp4"
            fd, tmp_video = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(fd, "wb") as fh, _ZIP.open(member) as src:
                shutil.copyfileobj(src, fh, length=1 << 20)
            src_path = tmp_video
        else:
            member = _INDEX.get(clip_id)
            if member is None:
                return ("missing", clip_id, 0)
            src_path = str(Path(videos_dir) / member) if not os.path.isabs(member) else member

        jpegs, err = video_to_jpegs(src_path, target_fps, size, max_frames,
                                    max_src_frames=max_src_frames)
        if err:
            kind = "toolong" if err.startswith("too_long") else "error"
            return (kind, f"{clip_id}:{err}", 0)
        if len(jpegs) < MIN_VALID_FRAMES:
            return ("error", f"{clip_id}:only_{len(jpegs)}_frames", 0)
        write_lmdb(out_dir, clip_id, jpegs)
        return ("ok", clip_id, len(jpegs))
    except Exception as e:                      # noqa: BLE001 - one bad clip must not kill the run
        return ("error", f"{clip_id}:{type(e).__name__}:{e}", 0)
    finally:
        if tmp_video and os.path.exists(tmp_video):
            os.unlink(tmp_video)


# ---------------------------------------------------------------------------
def build_index_from_archive(source):
    zf = _open_archive(source)
    index = {}
    for name in zf.namelist():
        if name.endswith("/"):
            continue
        p = Path(name)
        if p.suffix.lower() in VIDEO_EXTS:
            index[p.stem] = name
    zf.close()
    return index


def build_index_from_dir(videos_dir):
    index = {}
    for root, _dirs, files in os.walk(videos_dir):
        for f in files:
            p = Path(f)
            if p.suffix.lower() in VIDEO_EXTS:
                index[p.stem] = str(Path(root) / f)
    return index


def read_manifest(path):
    """'<split>\\t<clip_id>' lines -> [(split, clip_id)]."""
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        if "\t" in line:
            split, _, name = line.partition("\t")
            rows.append((split.strip(), name.strip()))
        else:
            rows.append(("train", line.strip()))
    return rows


def interleave_by_split(rows):
    """Reorder so that ANY prefix keeps the original split proportions.

    Matters because `--stop_after` and `--max_src_frames` cut the run short.
    In manifest order (all train, then dev, then test) an early stop produces
    a full train split and an empty test split - which then silently becomes
    a training run with no test set. Emitting the split that is furthest
    behind its quota keeps 80/10/10 true at every stopping point.
    """
    from collections import defaultdict
    buckets = defaultdict(list)
    for split, name in rows:
        buckets[split].append(name)
    totals = {k: len(v) for k, v in buckets.items()}
    n = sum(totals.values())
    if n == 0:
        return []
    emitted = {k: 0 for k in buckets}
    idx = {k: 0 for k in buckets}
    out = []
    for _ in range(n):
        # pick the split with the largest shortfall against its share
        best, best_gap = None, None
        for k in buckets:
            if idx[k] >= totals[k]:
                continue
            share = totals[k] / n
            gap = share * (len(out) + 1) - emitted[k]
            if best_gap is None or gap > best_gap:
                best, best_gap = k, gap
        out.append(buckets[best][idx[best]])
        idx[best] += 1
        emitted[best] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_argument_group("video source (pick one)")
    src.add_argument("--zip_parts", nargs="+", default=None,
                     help="Downloaded iSign video zip parts IN ORDER (part_aa part_ab)")
    src.add_argument("--hf_repo", default=None,
                     help="Read the zip by HTTP range straight from this HF dataset repo "
                          "(e.g. Exploration-Lab/iSign) - no 58 GB download")
    src.add_argument("--hf_parts", nargs="+",
                     default=["iSign-videos_v1.1_part_aa", "iSign-videos_v1.1_part_ab"])
    src.add_argument("--hf_revision", default="main")
    src.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    src.add_argument("--videos_dir", default=None,
                     help="A directory of already-extracted videos (searched recursively)")

    ap.add_argument("--manifest", required=True,
                    help="needed_videos.txt from isign_build_subset.py ('<split>\\t<clip_id>' lines)")
    ap.add_argument("--lmdb_root", required=True)
    ap.add_argument("--target_fps", type=int, default=25)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--max_frames", type=int, default=600,
                    help="Cap on STORED frames per clip; 0 = no cap")
    ap.add_argument("--max_src_frames", type=int, default=0,
                    help="Reject clips longer than this many source frames (0 = keep all). "
                         "On a fixed GPU budget this is the strongest cost lever: training "
                         "time is proportional to total tokens.")
    ap.add_argument("--stop_after", type=int, default=0,
                    help="Stop once this many clips have been written successfully. Use with a "
                         "larger manifest so short-clip rejections don't shrink the dataset.")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) // 2))
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="Process only the first N clips (smoke test)")
    args = ap.parse_args()

    n_sources = sum(bool(x) for x in (args.zip_parts, args.hf_repo, args.videos_dir))
    if n_sources != 1:
        sys.exit("[err] give exactly one of --zip_parts / --hf_repo / --videos_dir")

    if args.zip_parts:
        source = ("local", args.zip_parts, None)
    elif args.hf_repo:
        source = ("http", hf_urls(args.hf_repo, args.hf_parts, args.hf_revision), args.hf_token)
    else:
        source = ("dir", None, None)

    clips = interleave_by_split(read_manifest(args.manifest))
    if args.limit:
        clips = clips[:args.limit]
    print(f"[lmdb] {len(clips):,} clips in manifest")

    print("[lmdb] indexing video archive ...")
    index = (build_index_from_dir(args.videos_dir) if source[0] == "dir"
             else build_index_from_archive(source))
    print(f"[lmdb] archive holds {len(index):,} video files")

    Path(args.lmdb_root).mkdir(parents=True, exist_ok=True)
    tasks = [(c, args.lmdb_root, args.videos_dir, args.target_fps, args.size,
              args.max_frames, args.overwrite, args.max_src_frames) for c in clips]

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "toolong": 0}
    frames_total = 0
    problems = []
    target = args.stop_after or 0

    def record(status, info, n):
        nonlocal frames_total
        counts[status] += 1
        frames_total += n
        if status in ("missing", "error", "toolong"):
            problems.append(f"{status}\t{info}")

    bar_total = min(len(tasks), target) if target else len(tasks)

    if args.workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        pool = ctx.Pool(args.workers, initializer=_init_worker, initargs=(source, index))
        try:
            with tqdm(total=bar_total, desc="mp4->lmdb") as bar:
                for status, info, n in pool.imap_unordered(_process, tasks, chunksize=2):
                    record(status, info, n)
                    if status in ("ok", "skip"):
                        bar.update(1)
                    if target and (counts["ok"] + counts["skip"]) >= target:
                        break
        finally:
            pool.terminate()
            pool.join()
    else:
        _init_worker(source, index)
        with tqdm(total=bar_total, desc="mp4->lmdb") as bar:
            for t in tasks:
                status, info, n = _process(t)
                record(status, info, n)
                if status in ("ok", "skip"):
                    bar.update(1)
                if target and (counts["ok"] + counts["skip"]) >= target:
                    break

    # Authoritative count: with --stop_after, workers that were already in
    # flight when the loop broke still finished and wrote their LMDBs, so the
    # number on disk can exceed `written` by up to (workers x chunksize).
    # Report what actually exists - isign_sync_csvs.py uses the disk, not this
    # counter, and a run that claims 5,000 clips while holding 5,006 is the
    # kind of small dishonesty that makes later numbers hard to trust.
    on_disk = sum(1 for d in Path(args.lmdb_root).iterdir() if d.is_dir())
    print(f"\n[lmdb] written={counts['ok']:,} already_there={counts['skip']:,} "
          f"too_long={counts['toolong']:,} missing={counts['missing']:,} errors={counts['error']:,}")
    print(f"[lmdb] LMDB directories on disk: {on_disk:,}  <- this is the dataset size")
    if counts["ok"]:
        print(f"[lmdb] mean stored frames/clip = {frames_total / counts['ok']:.1f} "
              f"(-> ~{frames_total / counts['ok'] / 2:.0f} tokens after stride-2)")
    if problems:
        pf = Path(args.lmdb_root).parent / "lmdb_problems.txt"
        pf.write_text("\n".join(problems) + "\n")
        print(f"[lmdb] {len(problems):,} rejected clips listed in {pf}")
    print("[lmdb] NEXT: run scripts/isl/isign_sync_csvs.py so the corpus CSVs and the "
          "pseudo-gloss vocabulary describe only the clips that exist.")


if __name__ == "__main__":
    main()
