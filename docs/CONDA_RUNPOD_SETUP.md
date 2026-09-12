# Sign2GPT on RunPod — Conda Setup & Run (Sonant fork)

> Current setup guide for **this** repo (`mohitworkssonant/sonant-ISL-to-text`, branch
> `sonantisl`) using a **conda** environment. It reconciles the proven bootstrap in
> `docs/PROJECT_HANDOFF.md` §6/§17 (written for the previous engineer's pip/venv setup) into a
> clean conda flow, with the frozen stack (Python 3.10, torch 2.4.0 — no version changes).
>
> The `docs/PROJECT_HANDOFF.md` and `docs/ISL_TRAINING_GUIDE.md` are still the reference for the
> *why* and the ISL specifics — this file is the *how to stand it up in conda*.

---

## 0. Pick the pod (RunPod)

- **Template:** `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04` (gives CUDA 12.4 + driver).
- **GPU:** A100 80GB (best), or L4 24GB / RTX 4090 / RTX 4000-Ada (cheap, work).
  **AVOID Blackwell** (B200, RTX 5090, RTX PRO 4000 Blackwell) — torch 2.4.0 has no kernels for
  compute capability 12.x → `no kernel image is available`.
- **Network volume** mounted at `/workspace` (persists across pods), pod in the **same datacenter**
  as the volume.
- Container disk ≥ 30 GB.

---

## 1. System libraries (apt) — needed for Pillow 9.0.1 build + video

```bash
apt-get update && apt-get install -y \
    tmux ffmpeg git \
    zlib1g-dev libjpeg-dev libpng-dev libtiff-dev libfreetype6-dev build-essential
```

## 2. Get conda (skip if `conda` already works)

The pytorch template usually has no conda. Install Miniconda once:

```bash
cd /workspace
wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh -b -p /workspace/miniconda3
source /workspace/miniconda3/etc/profile.d/conda.sh
conda init bash && source ~/.bashrc
```

> Putting Miniconda on `/workspace` keeps the env across pod restarts. If your template already
> has conda, just `source` its `conda.sh` and skip the install.

## 3. Create the env (Python 3.10)

```bash
conda create -y -n sonant-isl python=3.10
conda activate sonant-isl
python -m pip install --upgrade pip wheel setuptools
```

## 4. Clone the repo (if not already on the volume)

```bash
cd /workspace
git clone -b sonantisl https://github.com/mohitworkssonant/sonant-ISL-to-text.git
cd sonant-ISL-to-text
```

## 5. Install the stack — ORDER MATTERS

```bash
# 5a. torch FIRST (pinned; cu124 matches the 12.4 template). Force-reinstall so nothing later downgrades it.
pip install --force-reinstall torch==2.4.0 torchvision==0.19.0 \
    --index-url https://download.pytorch.org/whl/cu124

# 5b. the pinned deps (requirements.txt is in the repo root)
pip install --no-cache-dir -r requirements.txt

# 5c. fasttext — MUST use --no-build-isolation (plain install fails to compile)
pip install --no-build-isolation fasttext==0.9.2

# 5d. xformers — MUST use --no-deps (else it pulls a newer torch → "driver too old")
pip install --no-deps xformers==0.0.27.post2

# 5e. spaCy English model (ISL→English pseudo-glosses) — direct wheel URL, NOT `spacy download`
pip install https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

# 5f. re-pin the ABI-sensitive pair last (some deps above nudge numpy)
pip install --force-reinstall scikit-image==0.22.0 numpy==1.24.4
```

## 6. Environment variables (the configs read these)

```bash
export SIGN2GPT_CKPT_PATH=/workspace/checkpoints
export SIGN2GPT_LMDB_PATH=/workspace/lmdb
mkdir -p "$SIGN2GPT_CKPT_PATH" "$SIGN2GPT_LMDB_PATH" /workspace/data /workspace/results
# make them stick for future shells:
echo 'export SIGN2GPT_CKPT_PATH=/workspace/checkpoints' >> ~/.bashrc
echo 'export SIGN2GPT_LMDB_PATH=/workspace/lmdb'         >> ~/.bashrc
```

## 7. Verify

```bash
python -c "
import torch
print('Torch:', torch.__version__, '| CUDA avail:', torch.cuda.is_available())
print('GPU:', torch.cuda.get_device_name(0))
print('Compute capability:', torch.cuda.get_device_capability(0))   # must be < (12, x)
"
python -c "import transformers, timm, lmdb, ignite, ml_collections, cv2, mediapipe, fasttext, einops, xformers; import spacy; spacy.load('en_core_web_lg'); print('All imports + spaCy en OK')"
```

If `import xformers` fails, you skipped `--no-deps`; if you see `numpy.dtype size changed`, re-run 5f;
if `no kernel image is available`, the GPU is Blackwell — swap it.

---

## 8. Run it

Use `tmux new -s work` first — a closed browser tab kills a non-tmux run.

**Path A — confirm the install actually trains (cheap smoke test).** Point the ISL configs at a tiny
handful of clips and set `cfg.max_epochs = 1`, then:

```bash
python main.py --config=configs/isl/isl_stage1_config.py
```

If one epoch runs and checkpoints, the environment is good.

**Path B — the real ISL run on iSign** (the current plan — you are NOT collecting your own clips, so
skip `ISL_TRAINING_GUIDE.md` Phases 1–4). Prepare the iSign subset exactly as in the project's
`isign-sign2gpt-run-plan` / `isl-modality-comparison-roadmap` (download iSign → convert to
`data/isl/ISL.{train,dev,test}.corpus.csv` → `scripts/isl/mp4_to_frames.py` →
`scripts/pseudo_gloss_en.py` → `scripts/phoenix2014t/image_lmdb_creator.py`), then train the two stages:

```bash
python main.py --config=configs/isl/isl_stage1_config.py 2>&1 | tee /workspace/results/isl_stage1.log
python main.py --config=configs/isl/isl_stage2_config.py 2>&1 | tee /workspace/results/isl_stage2.log
```

Watch progress: `grep "valid/ableu" /workspace/results/isl_stage2.log | tail`.

**Path C — optional PHOENIX (German) sanity run.** The repo's `setup_runpod.sh` orchestrates the full
German validation end-to-end (it will `pip install` too — harmless inside the conda env). Only needed if
you want to reproduce the BLEU-4 13.37 result; not required for ISL.

---

## Notes carried over from the fork's own docs

- These commands mirror `docs/PROJECT_HANDOFF.md` §17 (the proven bootstrap) — I only changed venv→conda,
  the repo/branch to yours, and the spaCy model German→English.
- The old docs assume the previous engineer's network volume named `sign2gpt-data` and paths like
  `D:\Sign2GPT` — substitute your own volume name and paths.
- Frozen stack is deliberate: torch 2.4.0 + xformers 0.0.27.post2 + transformers 4.31.0 are code-coupled
  (see the version analysis on file). Do not bump them without the code migration.
