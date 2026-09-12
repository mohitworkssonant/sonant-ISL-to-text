#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_isign_10k.sh - end-to-end Sign2GPT run on a 10,000-pair iSign subset.
#
#   Phase A (CPU pod, ~4-6 h) : download -> subset -> LMDB -> pseudo-gloss
#   Phase B (A100 80GB, ~35 h): stage 1 -> stage 2 -> evaluate
#
# Every phase is idempotent; re-running skips completed work.
#
#   export HF_TOKEN=hf_xxx           # after accepting the iSign terms
#   bash scripts/isl/run_isign_10k.sh prep     # on the CPU pod
#   bash scripts/isl/run_isign_10k.sh train    # on the GPU pod
# ---------------------------------------------------------------------------
set -euo pipefail

export SIGN2GPT_ROOT=${SIGN2GPT_ROOT:-/workspace/sonant-ISL-to-text}
export SIGN2GPT_DATA=${SIGN2GPT_DATA:-/workspace/data}
export SIGN2GPT_CKPT_PATH=${SIGN2GPT_CKPT_PATH:-/workspace/checkpoints}
export SIGN2GPT_LMDB_PATH=${SIGN2GPT_LMDB_PATH:-/workspace/lmdb}
export SIGN2GPT_RESULTS=${SIGN2GPT_RESULTS:-/workspace/results}

ISIGN_RAW="${SIGN2GPT_DATA}/isign"
ISL_DATA="${SIGN2GPT_ROOT}/data/isl"
PKL="${ISL_DATA}/processed_words.isl_pkl"
LMDB_VIDEOS="${SIGN2GPT_LMDB_PATH}/isl/lmdb_videos"
MAX_CLIPS=${MAX_CLIPS:-10000}

mkdir -p "${ISIGN_RAW}" "${ISL_DATA}" "${LMDB_VIDEOS}" "${SIGN2GPT_RESULTS}" "${SIGN2GPT_CKPT_PATH}"
log() { echo -e "\n\033[1;36m[isign10k] $*\033[0m"; }

prep() {
  log "A1/5  python deps"
  pip install --no-cache-dir -q \
      albumentations==1.4.13 numpy==1.24.4 pandas==2.0.1 transformers==4.31.0 \
      lmdb==1.2.1 timm==0.9.16 requests ml-collections==0.1.1 \
      pytorch-ignite==0.4.13 Pillow==9.0.1 matplotlib nlpaug==1.1.11 nltk==3.6.7 \
      fasttext==0.9.2 sentencepiece==0.1.99 einops==0.8.0 mediapipe==0.10.5 \
      onnxscript albucore==0.0.13 spacy==3.7.4 opencv-python==4.8.1.78 \
      "huggingface_hub[cli]"
  pip install --no-cache-dir -q --no-deps xformers==0.0.27.post2
  pip install --no-cache-dir -q \
      https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl

  log "A2/5  download iSign text + video parts (~58 GB, poses NOT downloaded)"
  if [ ! -f "${ISIGN_RAW}/iSign_v1.1.csv" ]; then
    huggingface-cli download Exploration-Lab/iSign --repo-type dataset \
      --include "iSign_v1.1.csv" "iSign-videos_v1.1_part_*" \
      --local-dir "${ISIGN_RAW}"
  else
    log "      already present, skipping"
  fi

  log "A3/5  build the ${MAX_CLIPS}-pair subset + corpus CSVs"
  cd "${SIGN2GPT_ROOT}"
  python scripts/isl/isign_build_subset.py \
      --csv "${ISIGN_RAW}/iSign_v1.1.csv" \
      --output_dir "${ISL_DATA}" \
      --max_clips "${MAX_CLIPS}" --seed 1

  log "A4/5  mp4 (streamed from the zip) -> per-clip LMDB"
  python scripts/isl/isign_zip_to_lmdb.py \
      --zip_parts "${ISIGN_RAW}"/iSign-videos_v1.1_part_a* \
      --manifest  "${ISL_DATA}/needed_videos.txt" \
      --lmdb_root "${LMDB_VIDEOS}" \
      --target_fps 25 --size 256 --workers "$(nproc)"

  log "A5/5  English pseudo-gloss vocabulary"
  if [ ! -f "${PKL}" ]; then
    python scripts/pseudo_gloss_en.py --csv_dir "${ISL_DATA}" --output_pkl "${PKL}"
  else
    log "      pkl already exists, skipping"
  fi

  log "preflight"
  python scripts/isl/isign_preflight.py \
      --csv_dir "${ISL_DATA}" --pkl "${PKL}" --lmdb_root "${LMDB_VIDEOS}"
}

train() {
  cd "${SIGN2GPT_ROOT}"
  log "B1/2  stage 1 - vision pretraining"
  python main.py --config=configs/isl/isign10k_stage1_config.py \
      2>&1 | tee -a "${SIGN2GPT_RESULTS}/isign10k_stage1.log"

  log "B2/2  stage 2 - translation"
  python main.py --config=configs/isl/isign10k_stage2_config.py \
      2>&1 | tee -a "${SIGN2GPT_RESULTS}/isign10k_stage2.log"

  log "results"
  grep -h "valid/class_f1_score" "${SIGN2GPT_RESULTS}/isign10k_stage1.log" | tail -3 || true
  grep -h "valid_test/ableu"     "${SIGN2GPT_RESULTS}/isign10k_stage2.log" | tail -3 || true
  echo "Quote valid_test/ableu (beam search). valid/obleu is teacher-forced and flatters."
}

case "${1:-}" in
  prep)  prep ;;
  train) train ;;
  all)   prep; train ;;
  *) echo "usage: $0 {prep|train|all}"; exit 1 ;;
esac
