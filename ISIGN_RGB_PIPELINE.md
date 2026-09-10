# iSign → Sign2GPT: RGB video-to-text pipeline

Runs this repo's Sign2GPT on a subset of the public **iSign** ISL→English corpus
(`Exploration-Lab/iSign`), RGB video only — the `.pose` stream is not used, matching the
original Sign2GPT. Additive to the PHOENIX path: no existing file is modified, so the German
validation run still reproduces exactly as before.

**Licence:** iSign is CC-BY-NC-SA-4.0 — non-commercial, share-alike. Nothing trained on it can
ship. Keep that line attached to every number produced here.

---

## Two engineering changes vs. the stock path

### 1. No PNG stage — MP4 decodes straight into LMDB

The stock path is `mp4 → PNG frames on disk → image_lmdb_creator.py → LMDB`. For PHOENIX
(7.8k short clips) that's fine. For iSign it is not:

```
10,000 clips × ~215 frames × ~120 KB per 256×256 PNG  ≈  250 GB
```

of temporary files that are deleted immediately afterwards — because the LMDB creator
re-encodes them to JPEG q90 anyway. `isign_zip_to_lmdb.py` writes those JPEG bytes directly
into the LMDB: **~50 GB and one pass instead of ~300 GB and three.** The output is
structurally identical to what `scripts/phoenix2014t/image_lmdb_creator.py` produces (keys
`"0".."N-1"` + a `details` pickle), so the dataloader, augmentation and configs are unchanged.

Also here: frames are **padded to square** before resize. The stock `mp4_to_frames.py` does a
plain `cv2.resize` to 256×256, which squashes a 16:9 frame ~1.8× horizontally. DINOv2 was
pretrained on undistorted images and handshape is the entire signal, so squashing is not a
free simplification. Edge-replicated borders keep background statistics closer to the frame.

### 2. The archive is never unzipped — and optionally never downloaded

`iSign-videos_v1.1_part_aa` + `part_ab` are a split zip. `cat part_* > videos.zip` costs
another 58 GB for a file read once. `MultiPartFile` presents both parts to `zipfile` as one
seekable stream, so members are read in place.

`--hf_repo` goes further: a zip's central directory sits at the end and records every member's
byte offset, and HTTP supports ranges — so `isign_http_zip.py` reads the directory and then
only the members requested, straight from Hugging Face. A 5,000-clip subset touches ~2.5% of
the 58 GB. **58 GB download + 80 GB volume → ~2.5 GB of traffic and a 30 GB volume.**

Fallback if a CDN ever refuses ranges: `huggingface-cli download` the parts and pass
`--zip_parts` instead. Nothing else changes. `python scripts/isl/isign_http_zip.py` self-tests
range access in about a minute before you commit pod time.

---

## Files

| File | Role |
|---|---|
| `scripts/isl/isign_build_subset.py` | iSign master CSV → subset → `ISL.{train,dev,test}.corpus.csv` (+ PHOENIX-named copies) + `needed_videos.txt`. Splits **by `video_id`**, cleans text once. |
| `scripts/isl/isign_zip_to_lmdb.py` | MP4 → per-clip LMDB. `--zip_parts` / `--hf_repo` / `--videos_dir`; `--max_src_frames`, `--stop_after`. |
| `scripts/isl/isign_http_zip.py` | Seekable HTTP-range view over the split zip. Run as `__main__` to self-test. |
| `scripts/isl/isign_sync_csvs.py` | Prune corpus CSVs to clips that actually have an LMDB. **Run before building the pseudo-gloss pkl.** |
| `scripts/isl/isign_preflight.py` | Five checks for the failure modes that train happily and learn nothing. |
| `scripts/isl/isign_budget.py` | Measured sec/iteration + $ available → affordable epoch counts. |
| `scripts/isl/run_isign_budget.sh` | `check` / `prep` / `calibrate` / `train` — the ~$3.50 run on a 24 GB card. |
| `scripts/isl/run_isign_10k.sh` | `prep` / `train` / `all` — the ~$45 10k run on an A100 80GB. |
| `configs/isl/isign_budget_stage{1,2}_config.py` | 24 GB, `max_seq_len` 128, XGLM-564M, 12 epochs. |
| `configs/isl/isign10k_stage{1,2}_config.py` | A100 80GB, XGLM-1.7B, 20 epochs. |

---

## Quick start (budget run)

```bash
export SIGN2GPT_ROOT=$PWD SIGN2GPT_DATA=/workspace/data \
       SIGN2GPT_CKPT_PATH=/workspace/checkpoints SIGN2GPT_LMDB_PATH=/workspace/lmdb \
       SIGN2GPT_RESULTS=/workspace/results
export HF_TOKEN=hf_...                              # after accepting the iSign terms

bash scripts/isl/run_isign_budget.sh check          # prove range access, ~1 min
bash scripts/isl/run_isign_budget.sh prep           # subset + LMDB + vocab + preflight
bash scripts/isl/run_isign_budget.sh calibrate      # measure real sec/iteration
export ISIGN_S1_EPOCHS=12 ISIGN_S2_EPOCHS=12
bash scripts/isl/run_isign_budget.sh train
```

**Order matters:** prune the CSVs (`isign_sync_csvs.py`) *before* building
`processed_words.isl_pkl`. Otherwise `num_classes` and the 0.4 frequency cut-off that selects
pseudo-gloss targets are computed from sentences the model never sees. `run_isign_budget.sh`
does this in the right order.

**Calibrate before committing:** the cosine LR schedule is defined over `max_epochs`, so
"start small and extend later" changes the learning-rate curve. Decide the epoch count up front.

**Stage 2 must reach 10 epochs.** The trainer fires the beam-search tester on
`Events.EPOCH_COMPLETED(every=10)`. Below 10 there is no autoregressive BLEU at all. If the
budget doesn't fit, cut clips — not epochs.

---

## Reading the result

`valid/obleu` is **teacher-forced** and used only for checkpoint selection; on German it read
18.24 where the real beam-search score was 13.37. Report `valid_test/ableu`.

Calibration: iSign's own SignVideo2Text baselines are BLEU-4 **0.24–0.56** on the full 118k
corpus; human translators score 69.3 on the same data. Low single digits from a few thousand
clips is the dataset and the scale, not a broken pipeline. What a small run can honestly
conclude is that the pipeline runs on real ISL RGB video, whether stage-1 `class_f1` leaves
zero, and whether the frozen LM still emits fluent English — plus a measured cost-per-frame
number to price the full run.

---

## Five silent failure modes (all checked by `isign_preflight.py`)

1. **`orth` is NaN** — the dataloader calls `item["orth"].split(" ")` unguarded, and an empty
   CSV field becomes a float NaN. The subset builder mirrors `translation` into `orth`.
2. **Pseudo-gloss keys don't match** — `dict_sentence` is keyed by the exact translation
   string. Clean the text anywhere after the CSV is written and every lookup misses,
   `pseudo_gloss_ids` comes back empty, and stage 1 optimises a constant while the loss
   politely decreases. Clean once, in `isign_build_subset.py`.
3. **Pipes/tabs in the text** — CSVs are read with `sep="|"`; one literal `|` shifts every
   later column in that row. Stripped at build time.
4. **Rebuilding the pkl between stages** — `num_classes` is baked into the stage-1 head at
   config-load time; regenerate the pkl afterwards and stage 2 loads a checkpoint whose head
   shape no longer matches.
5. **`max_length` set in only one place** — `cfg.max_length` caps target tokenisation,
   `cfg.gen_params["max_length"]` caps generation. Change one and you truncate either the
   reference or the hypothesis.

Plus two operational ones: run from the repo root (the prototype head loads
`cc.en.300.bin` by relative path), and use `tmux`.

---

## Notes

- `--stop_after` stops mid-manifest. The manifest is written train→dev→test, so a naive early
  stop yields a full train split and an **empty test split**. The writer emits whichever split
  is furthest behind its quota, keeping 80/10/10 true at any prefix.
- Splitting is by `video_id`, never by clip: iSign uids are `<video_id>-<segment>` and
  consecutive segments share signer, background and topic. Clip-wise splits leak the test
  signer into training.
- XGLM-564M and XGLM-1.7B both have 24 decoder layers, so `adapt_layers`/`lora_layers =
  arange(0,24)` is correct for either, and `test_stage2_model` reads `lang_model.embed_dim`
  dynamically. Pointing `ISIGN_LM` at a model of different depth means editing those lists.
- Verified end to end on a synthetic iSign-shaped release (split zip, 16:9 videos, mixed fps,
  pipe/tab characters in the text, uids ending in `-`): subset builder → LMDB writer →
  sync → preflight → this repo's real `PhoenixVideoDataset`, which returned correct `frames`,
  `sentence` and `pseudo_gloss_ids`.

Longer write-ups live in the Claude project: `isign-sign2gpt-7-dollar-run.md` (budget run) and
`isign-10k-sign2gpt-complete-guide.md` (full guide).
