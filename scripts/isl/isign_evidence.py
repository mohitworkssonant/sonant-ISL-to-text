"""
Evidence pack: N held-out clips -> video file, prediction, ground truth, timing.

The trial proved the pipeline runs. This produces the artefact someone can
actually inspect: the video, what the model said, what the correct answer was,
and how long it took.

Timing is reported in the only way that is honest for a latency claim:

  * model load is measured once and reported separately - it is startup cost,
    not per-translation cost, and quoting it as latency would overstate by ~10x;
  * the first clip is a warm-up and excluded from the averages - CUDA kernel
    autotuning and cuDNN benchmarking make it several times slower than steady
    state;
  * decode/preprocess and GPU inference are timed separately, because they
    scale differently and only one of them is the model's fault;
  * the clip's own duration is recorded, so the number can be quoted as a
    real-time factor rather than an absolute that means nothing without it.

Usage:
    python scripts/isl/isign_evidence.py \
        --config configs.isl.isign_budget_stage2_config \
        --n 5 --out_dir /workspace/evidence

Writes into out_dir: the MP4s, results.csv, results.md, and evidence.html
(open it locally - the videos play inline next to their translations).
"""

import argparse
import csv
import html
import importlib
import io
import os
import shutil
import statistics
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from isign_http_zip import MultiPartHttpFile, hf_urls        # noqa: E402
from isign_zip_to_lmdb import MultiPartFile, VIDEO_EXTS      # noqa: E402
from isign_infer import read_video                           # noqa: E402


# ---------------------------------------------------------------------------
def fetch_clips(clip_ids, out_dir, hf_repo, hf_parts, hf_token, zip_parts=None):
    if zip_parts:
        raw = MultiPartFile(zip_parts)
    else:
        raw = MultiPartHttpFile(hf_urls(hf_repo, hf_parts), hf_token)
    zf = zipfile.ZipFile(io.BufferedReader(raw, buffer_size=1 << 20))
    index = {Path(n).stem: n for n in zf.namelist()
             if Path(n).suffix.lower() in VIDEO_EXTS}
    got = {}
    for cid in clip_ids:
        member = index.get(cid)
        if member is None:
            print(f"[evidence] not in archive: {cid}", file=sys.stderr)
            continue
        dst = Path(out_dir) / f"{cid}.mp4"
        with zf.open(member) as src, open(dst, "wb") as fh:
            shutil.copyfileobj(src, fh, length=1 << 20)
        got[cid] = dst
    return got


def write_outputs(rows, load_s, out_dir, cfg_name, gpu):
    out_dir = Path(out_dir)

    with open(out_dir / "results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["clip_id", "video_file", "clip_seconds", "frames_used",
                    "prediction", "ground_truth",
                    "preprocess_s", "inference_s", "total_s", "realtime_factor"])
        for r in rows:
            w.writerow([r["clip"], r["file"], f"{r['dur']:.2f}", r["tokens"],
                        r["pred"], r["tgt"],
                        f"{r['pre']:.3f}", f"{r['inf']:.3f}", f"{r['tot']:.3f}",
                        f"{r['rtf']:.2f}"])

    timed = rows[1:] if len(rows) > 1 else rows          # drop the warm-up
    med_tot = statistics.median(r["tot"] for r in timed)
    med_inf = statistics.median(r["inf"] for r in timed)
    med_pre = statistics.median(r["pre"] for r in timed)
    med_rtf = statistics.median(r["rtf"] for r in timed)

    md = [f"# iSign x Sign2GPT - held-out clip evidence\n",
          f"- Model: `{cfg_name}`",
          f"- GPU: {gpu}",
          f"- Model load (one-off startup): **{load_s:.1f} s**",
          f"- Clips: {len(rows)} (first excluded from timing as warm-up)\n",
          "## Timing per clip (median of the timed clips)\n",
          "| | seconds |", "|---|---|",
          f"| Video decode + preprocess | {med_pre:.2f} |",
          f"| Model inference (beam search) | {med_inf:.2f} |",
          f"| **Total per clip** | **{med_tot:.2f}** |",
          f"| Real-time factor (total / clip length) | {med_rtf:.2f}x |\n",
          "## Clip by clip\n",
          "| # | clip | length | model output | ground truth | total |",
          "|---|---|---|---|---|---|"]
    for i, r in enumerate(rows):
        warm = " *(warm-up)*" if i == 0 and len(rows) > 1 else ""
        md.append(f"| {i+1}{warm} | `{r['clip']}` | {r['dur']:.1f}s | {r['pred']} | "
                  f"{r['tgt']} | {r['tot']:.2f}s |")
    md += ["",
           "Predictions are expected to be wrong at this training scale - the trial used",
           "226 training clips. What these show is that a real ISL video goes in and a",
           "fluent English sentence comes out, at a measured speed.",
           "",
           "Clips are from the iSign dataset (Exploration-Lab), CC-BY-NC-SA-4.0,",
           "research use only."]
    (out_dir / "results.md").write_text("\n".join(md))

    cards = []
    for i, r in enumerate(rows):
        warm = ' <span class="warm">warm-up</span>' if i == 0 and len(rows) > 1 else ""
        cards.append(f"""  <div class="card">
    <video controls preload="metadata" src="{html.escape(Path(r['file']).name)}"></video>
    <div class="meta">
      <div class="cid">{html.escape(r['clip'])}{warm}</div>
      <div class="row"><span class="k">Model output</span><span class="v pred">{html.escape(r['pred'])}</span></div>
      <div class="row"><span class="k">Ground truth</span><span class="v">{html.escape(r['tgt'])}</span></div>
      <div class="row"><span class="k">Clip length</span><span class="v">{r['dur']:.1f}s</span></div>
      <div class="row"><span class="k">Time to translate</span><span class="v">{r['tot']:.2f}s
        <small>({r['pre']:.2f}s decode + {r['inf']:.2f}s model)</small></span></div>
    </div>
  </div>""")

    page = f"""<!doctype html>
<meta charset="utf-8"><title>iSign x Sign2GPT - clip evidence</title>
<style>
 body{{font:15px/1.5 system-ui,sans-serif;margin:0;padding:24px;background:#faf9f7;color:#1a1a1a}}
 h1{{font-size:22px;margin:0 0 4px}}
 .sub{{color:#666;margin-bottom:20px}}
 .summary{{background:#fff;border:1px solid #e5e2dd;border-radius:8px;padding:14px 18px;margin-bottom:22px;max-width:760px}}
 .summary table{{border-collapse:collapse}} .summary td{{padding:3px 18px 3px 0}}
 .summary td:last-child{{font-variant-numeric:tabular-nums;font-weight:600}}
 .card{{display:flex;gap:18px;background:#fff;border:1px solid #e5e2dd;border-radius:8px;
        padding:14px;margin-bottom:14px;max-width:960px;flex-wrap:wrap}}
 video{{width:280px;background:#000;border-radius:6px}}
 .meta{{flex:1;min-width:320px}}
 .cid{{font-family:ui-monospace,monospace;font-size:12px;color:#888;margin-bottom:8px}}
 .warm{{background:#f3efe6;color:#8a6d3b;border-radius:4px;padding:1px 6px;font-family:system-ui;font-size:11px}}
 .row{{display:flex;gap:12px;padding:4px 0;border-top:1px solid #f0ede8}}
 .k{{width:130px;color:#777;flex:none}} .v{{flex:1}}
 .pred{{font-weight:600}}
 small{{color:#999}}
 .note{{max-width:760px;color:#555;font-size:14px;margin-top:20px;
        border-left:3px solid #d9d4cb;padding-left:14px}}
</style>
<h1>iSign &times; Sign2GPT &mdash; held-out clip evidence</h1>
<div class="sub">{html.escape(gpu)} &middot; model <code>{html.escape(cfg_name)}</code></div>
<div class="summary"><table>
 <tr><td>Model load (one-off)</td><td>{load_s:.1f} s</td></tr>
 <tr><td>Median decode + preprocess</td><td>{med_pre:.2f} s</td></tr>
 <tr><td>Median model inference</td><td>{med_inf:.2f} s</td></tr>
 <tr><td>Median total per clip</td><td>{med_tot:.2f} s</td></tr>
 <tr><td>Real-time factor</td><td>{med_rtf:.2f}&times;</td></tr>
</table></div>
{chr(10).join(cards)}
<div class="note">
 Predictions are expected to be <strong>wrong</strong> at this training scale &mdash; the trial
 used 226 training clips, about 1/500th of the iSign corpus. What these clips demonstrate is
 that real ISL video goes in and fluent English comes out, at a measured speed. Accuracy is a
 question for the full-scale run.<br><br>
 Clips from the iSign dataset (Exploration-Lab), CC-BY-NC-SA-4.0, research use only.
</div>
"""
    (out_dir / "evidence.html").write_text(page)
    return med_pre, med_inf, med_tot, med_rtf


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs.isl.isign_budget_stage2_config")
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--out_dir", default="/workspace/evidence")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--hf_repo", default="Exploration-Lab/iSign")
    ap.add_argument("--hf_parts", nargs="+",
                    default=["iSign-videos_v1.1_part_aa", "iSign-videos_v1.1_part_ab"])
    ap.add_argument("--hf_token", default=os.environ.get("HF_TOKEN"))
    ap.add_argument("--zip_parts", nargs="+", default=None)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer
    from augmentation.get_aug import get_aug
    from models.get_models import get_model
    from train_utils.checkpoint_helpers import get_best_checkpoint_details

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu = torch.cuda.get_device_name(0) if device == "cuda" else "CPU"

    cfg = importlib.import_module(args.config).get_config()
    test_csv = Path(cfg.test_ds_params["csv_dir"])
    df = pd.read_csv(test_csv, sep="|").head(args.n)
    print(f"[evidence] {len(df)} clips from {test_csv.name}")

    print("[evidence] fetching videos ...")
    files = fetch_clips(df["name"].astype(str).tolist(), out_dir,
                        args.hf_repo, args.hf_parts, args.hf_token, args.zip_parts)

    # ---- model load, timed separately -------------------------------------
    t0 = time.perf_counter()
    ckpt = args.ckpt or get_best_checkpoint_details(
        cfg.save_dir, best_checkpoint_name="_result_checkpoint_")[0]
    if not ckpt:
        sys.exit(f"[evidence] no stage-2 checkpoint under {cfg.save_dir}")
    tok = AutoTokenizer.from_pretrained(cfg.lm_name)
    params = dict(cfg.model_params)
    params["pretext_length"] = 1
    params["stage1_ckpt"] = None
    model = get_model(cfg.model_name, params)
    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)
    model.eval().to(device)
    load_s = time.perf_counter() - t0
    print(f"[evidence] model loaded in {load_s:.1f}s from {ckpt}")

    transform = get_aug(cfg.aug_name, dict(cfg.aug_params))
    gen = {**dict(cfg.gen_params), "eos_token_id": tok.eos_token_id,
           "bos_token_id": tok.bos_token_id, "pad_token_id": tok.pad_token_id}
    bos = torch.tensor([[tok("")["input_ids"][0]]], device=device)

    rows = []
    for _, row in df.iterrows():
        cid = str(row["name"])
        if cid not in files:
            continue

        t = time.perf_counter()
        frames = read_video(files[cid])
        sel = np.arange(0, len(frames), transform.stride).astype(int)
        if len(sel) > transform.max_seq_len:
            sel = np.sort(np.random.choice(sel, transform.max_seq_len, replace=False))
        tensor = transform.aug_video([frames[i] for i in sel], isValid=True).to(device)
        if device == "cuda":
            torch.cuda.synchronize()
        pre = time.perf_counter() - t

        t = time.perf_counter()
        with torch.inference_mode(True):
            ctx = (torch.autocast(device_type="cuda", dtype=torch.float16)
                   if device == "cuda" else torch.autocast(device_type="cpu", enabled=False))
            with ctx:
                out = model(text_ids=bos, text_mask=torch.ones_like(bos, dtype=torch.bool),
                            frame_features=[tensor],
                            frame_mask=torch.ones(1, len(sel), dtype=torch.bool, device=device),
                            max_len=torch.tensor(512), generate=True, gen_params=gen)
        if device == "cuda":
            torch.cuda.synchronize()
        inf = time.perf_counter() - t

        ids = out["output_ids"][0]
        eos = (ids == tok.eos_token_id).nonzero()
        if len(eos):
            ids = ids[: eos[0, 0]]
        pred = tok.decode(ids, skip_special_tokens=True).strip()

        dur = len(frames) / 25.0                       # frames were resampled to 25 fps
        rows.append({"clip": cid, "file": str(files[cid]), "dur": dur,
                     "tokens": len(sel), "pred": pred,
                     "tgt": str(row["translation"]),
                     "pre": pre, "inf": inf, "tot": pre + inf,
                     "rtf": (pre + inf) / max(dur, 1e-6)})
        print(f"[evidence] {cid}  {pre+inf:.2f}s  ->  {pred!r}")

    if not rows:
        sys.exit("[evidence] no clips processed")

    med_pre, med_inf, med_tot, med_rtf = write_outputs(
        rows, load_s, out_dir, args.config.split(".")[-1], gpu)

    print(f"\n[evidence] {len(rows)} clips | model load {load_s:.1f}s (one-off)")
    print(f"[evidence] median per clip: {med_tot:.2f}s "
          f"({med_pre:.2f}s decode + {med_inf:.2f}s model), {med_rtf:.2f}x real time")
    print(f"[evidence] wrote {out_dir}/evidence.html, results.md, results.csv and the MP4s")
    print("[evidence] zip the folder and open evidence.html locally to see video + text together.")


if __name__ == "__main__":
    main()
