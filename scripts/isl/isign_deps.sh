#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# isign_deps.sh - install the Python environment this repo actually needs.
#
# Shared by run_isign_trial.sh / run_isign_budget.sh / run_isign_10k.sh.
#
# The original pin list came from the 2024 PHOENIX run on a Python 3.10 pod.
# Four of those pins do not survive a modern Python 3.11 RunPod image, and each
# one fails in a way that wastes pod time:
#
#   Pillow==9.0.1        predates cp311 wheels -> pip compiles from source and
#                        dies on "RequiredDependencyException: zlib".
#   fasttext==0.9.2      last released 2020; its src/args.cc does not compile
#                        under modern g++ ("'size' was not declared in this
#                        scope"). fasttext-wheel ships prebuilt binaries and
#                        exposes the identical API (fasttext.load_model,
#                        fasttext.util.download_model), which is all the
#                        prototype head uses.
#   huggingface_hub[cli] the `cli` extra no longer exists in hub 1.x, and hub
#                        1.x renames the command. Pinned <1.0 so
#                        `huggingface-cli download` keeps working.
#   mediapipe==0.10.5    not imported anywhere in this repo. Dropped.
#
# THE IMPORTANT ONE - xformers.
# This repo imports xformers.components.feedforward and
# xformers.components.positional_embedding (metaformer attention + sine
# embeddings). Those submodules were REMOVED in xformers 0.0.29. And each
# xformers wheel pins one exact torch:
#
#     xformers 0.0.27.post2  -> torch 2.4.0   components: present
#     xformers 0.0.28.post3  -> torch 2.5.1   components: present
#     xformers 0.0.29+       -> torch 2.6+    components: GONE
#
# So the environment must be torch 2.4.0 or 2.5.1. If the pod's template ships
# anything newer, this script installs torch 2.4.0 rather than let the run fail
# at model construction, twenty minutes in.
# ---------------------------------------------------------------------------
set -euo pipefail

log() { echo -e "\n\033[1;36m[deps] $*\033[0m"; }
die() { echo -e "\033[1;31m[deps] $*\033[0m" >&2; exit 1; }

log "1/4 core packages"
pip install --no-cache-dir -q \
    albumentations==1.4.13 \
    numpy==1.24.4 \
    pandas==2.0.1 \
    transformers==4.31.0 \
    "lmdb>=1.4.1" \
    timm==0.9.16 \
    requests \
    ml-collections==0.1.1 \
    pytorch-ignite==0.4.13 \
    "Pillow>=9.5,<11" \
    matplotlib \
    nlpaug==1.1.11 \
    nltk==3.6.7 \
    fasttext-wheel \
    sentencepiece==0.1.99 \
    einops==0.8.0 \
    onnxscript \
    albucore==0.0.13 \
    spacy==3.7.4 \
    opencv-python==4.8.1.78 \
    "huggingface_hub<1.0"

log "2/4 torch <-> xformers compatibility"
TORCH_MM=$(python -c "import torch,sys;print('.'.join(torch.__version__.split('+')[0].split('.')[:2]))" 2>/dev/null || echo none)
echo "[deps] installed torch: ${TORCH_MM}"

case "${TORCH_MM}" in
  2.4) XF=0.0.27.post2 ;;
  2.5) XF=0.0.28.post3 ;;
  *)
    echo "[deps] torch ${TORCH_MM} is outside the window this repo can use."
    echo "[deps] xformers >=0.0.29 dropped xformers.components.{feedforward,positional_embedding},"
    echo "[deps] which models/metaformer imports. Installing torch 2.4.0 instead."
    pip install --no-cache-dir -q torch==2.4.0 torchvision==0.19.0 \
        --index-url https://download.pytorch.org/whl/cu121
    XF=0.0.27.post2
    ;;
esac

log "3/4 xformers ${XF}"
# --no-deps so pip cannot "helpfully" move torch underneath us.
pip install --no-cache-dir -q --no-deps "xformers==${XF}"

log "4/4 spaCy English model + verification"
pip install --no-cache-dir -q \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

python - <<'PY'
import sys
ok = True

import torch
print(f"[deps] torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}"
      + (f" | {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else ""))

# The two xformers submodules this repo cannot run without.
try:
    from xformers.components.positional_embedding import build_positional_embedding  # noqa: F401
    from xformers.components.feedforward import build_feedforward                    # noqa: F401
    import xformers
    print(f"[deps] xformers {xformers.__version__} - components API present")
except Exception as e:                                        # noqa: BLE001
    ok = False
    print(f"[deps] FAIL xformers: {type(e).__name__}: {e}")
    print("       This is the removed-components problem. Pin torch 2.4.0 + xformers 0.0.27.post2.")

for mod, why in [("fasttext", "prototype head word vectors"),
                 ("lmdb", "frame storage"),
                 ("cv2", "video decode"),
                 ("PIL", "JPEG encode"),
                 ("spacy", "pseudo-glosses"),
                 ("ignite", "training loop"),
                 ("transformers", "XGLM")]:
    try:
        __import__(mod)
    except Exception as e:                                    # noqa: BLE001
        ok = False
        print(f"[deps] FAIL {mod} ({why}): {type(e).__name__}: {e}")

try:
    import fasttext.util  # noqa: F401
    import fasttext
    assert hasattr(fasttext, "load_model") and hasattr(fasttext.util, "download_model")
except Exception as e:                                        # noqa: BLE001
    ok = False
    print(f"[deps] FAIL fasttext API: {e}")

try:
    import spacy
    spacy.load("en_core_web_lg")
    print("[deps] en_core_web_lg loads")
except Exception as e:                                        # noqa: BLE001
    ok = False
    print(f"[deps] FAIL en_core_web_lg: {type(e).__name__}: {e}")

print("\033[1;32m[deps] environment OK\033[0m" if ok else "\033[1;31m[deps] environment BROKEN - fix the FAILs above before continuing\033[0m")
sys.exit(0 if ok else 1)
PY
