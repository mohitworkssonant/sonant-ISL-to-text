"""
Random access into the iSign split zip **over HTTP**, without downloading it.

WHY
---
`iSign-videos_v1.1_part_aa` + `part_ab` are 58 GB. A 3,000-clip subset needs
roughly 2.5% of the bytes in there. Downloading all 58 GB to read 1.5 GB costs
pod-hours and forces an 80 GB volume - on a $7 budget that is ~15% of the money
spent on bytes you throw away.

A zip's central directory sits at the END of the archive and records the byte
offset of every member. HTTP range requests give random access. So:

    1. HEAD both parts        -> sizes, hence the offset map of the virtual
                                 concatenated archive
    2. read the last ~16 MB   -> EOCD + central directory
    3. per wanted clip        -> one ranged GET of just that member

`zipfile` drives all of it; this module only has to look like a seekable file.

FALLBACK
--------
If HEAD/Range is refused (rare, but CDNs change), fall back to
`huggingface-cli download` + the local `--zip_parts` path. Nothing else in the
pipeline changes.
"""

import io
import os
import threading
import time
import urllib.error
import urllib.request

HF_BASE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
CHUNK = 8 << 20          # 8 MB fetch granularity - big enough that the central
                         # directory costs ~2 requests, small enough that one
                         # member read doesn't pull megabytes it won't use
MAX_CACHED_CHUNKS = 48   # ~384 MB per process, LRU
RETRIES = 4


def _opener(token):
    headers = [("User-Agent", "sonant-isl/1.0")]
    if token:
        headers.append(("Authorization", f"Bearer {token}"))
    op = urllib.request.build_opener()
    op.addheaders = headers
    return op


class HttpRangeFile:
    """One remote file, read by ranges, with an LRU chunk cache."""

    def __init__(self, url, token=None, size=None):
        self.url = url
        self.token = token
        self._op = _opener(token)
        self._cache = {}
        self._order = []
        self._lock = threading.Lock()
        self.size = size if size is not None else self._probe_size()

    def _probe_size(self):
        """Content-Length via HEAD, falling back to a 1-byte ranged GET.

        HF redirects LFS files to a CDN; urllib follows the redirect and the
        final response carries the real length. Some CDNs dislike HEAD, hence
        the Content-Range fallback.
        """
        try:
            req = urllib.request.Request(self.url, method="HEAD")
            with self._op.open(req, timeout=60) as r:
                n = r.headers.get("Content-Length")
                if n:
                    return int(n)
        except Exception:
            pass
        req = urllib.request.Request(self.url)
        req.add_header("Range", "bytes=0-0")
        with self._op.open(req, timeout=60) as r:
            cr = r.headers.get("Content-Range", "")
            if "/" in cr:
                return int(cr.rsplit("/", 1)[1])
        raise RuntimeError(f"cannot determine size of {self.url} (no Content-Length, no Content-Range)")

    def _fetch(self, start, end):
        """GET [start, end] inclusive. Retries with backoff; re-opens on 403,
        which is how an expired CDN signature shows up."""
        last = None
        for attempt in range(RETRIES):
            try:
                req = urllib.request.Request(self.url)
                req.add_header("Range", f"bytes={start}-{end}")
                with self._op.open(req, timeout=120) as r:
                    if r.status not in (206, 200):
                        raise RuntimeError(f"expected 206, got {r.status}")
                    return r.read()
            except Exception as e:                      # noqa: BLE001
                last = e
                if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403):
                    self._op = _opener(self.token)      # fresh connection/signature
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"range fetch failed after {RETRIES} tries: {last}")

    def read_range(self, offset, length):
        out = bytearray()
        while length > 0:
            idx = offset // CHUNK
            with self._lock:
                blob = self._cache.get(idx)
            if blob is None:
                start = idx * CHUNK
                end = min(start + CHUNK, self.size) - 1
                blob = self._fetch(start, end)
                with self._lock:
                    self._cache[idx] = blob
                    self._order.append(idx)
                    while len(self._order) > MAX_CACHED_CHUNKS:
                        self._cache.pop(self._order.pop(0), None)
            local = offset - idx * CHUNK
            take = min(length, len(blob) - local)
            if take <= 0:
                break
            out += blob[local:local + take]
            offset += take
            length -= take
        return bytes(out)


class MultiPartHttpFile(io.RawIOBase):
    """Seekable view over several remote files concatenated in order.

    Same interface as the local `MultiPartFile`, so `zipfile.ZipFile` cannot
    tell the difference between this and a downloaded archive.
    """

    def __init__(self, urls, token=None):
        self.parts = [HttpRangeFile(u, token) for u in urls]
        self.sizes = [p.size for p in self.parts]
        self.offsets, acc = [], 0
        for s in self.sizes:
            self.offsets.append(acc)
            acc += s
        self.total = acc
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
            i = max(j for j in range(len(self.parts)) if self.offsets[j] <= self._pos)
            local = self._pos - self.offsets[i]
            take = min(want - got, self.sizes[i] - local)
            chunk = self.parts[i].read_range(local, take)
            if not chunk:
                break
            b[got:got + len(chunk)] = chunk
            got += len(chunk)
            self._pos += len(chunk)
        return got


def hf_urls(repo, paths, revision="main"):
    """Resolve URLs for the archive parts.

    ISIGN_HF_BASE overrides the template (used by the test harness to point at
    a local Range-capable server, and usable to point at an internal mirror).
    """
    tmpl = os.environ.get("ISIGN_HF_BASE", HF_BASE)
    return [tmpl.format(repo=repo, rev=revision, path=p) for p in paths]


def self_test(urls, token=None):
    """Prove range access works before committing an hour of pod time."""
    import zipfile
    f = io.BufferedReader(MultiPartHttpFile(urls, token), buffer_size=1 << 20)
    zf = zipfile.ZipFile(f)
    names = zf.namelist()
    print(f"[http-zip] archive opened over HTTP: {len(names):,} members, "
          f"{sum(p.size for p in f.raw.parts) / 1e9:.1f} GB total, 0 bytes downloaded in full")
    vids = [n for n in names if n.lower().endswith((".mp4", ".mov", ".avi", ".mkv", ".webm"))]
    if vids:
        with zf.open(vids[0]) as fh:
            head = fh.read(4096)
        print(f"[http-zip] read {len(head)} bytes of {vids[0]} by range - OK")
    return len(names)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Self-test HTTP range access to the iSign video zip")
    ap.add_argument("--repo", default="Exploration-Lab/iSign")
    ap.add_argument("--parts", nargs="+",
                    default=["iSign-videos_v1.1_part_aa", "iSign-videos_v1.1_part_ab"])
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()
    self_test(hf_urls(a.repo, a.parts), a.token)
