#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_isign_budget.sh - Sign2GPT on iSign under a hard money ceiling (~$7).
#
# One 24 GB pod does everything. No 58 GB download: the MP4s are read out of
# the Hugging Face zip by HTTP range request.
#
#   export HF_TOKEN=hf_xxx            # after accepting the iSign terms
#   bash scripts/isl/run_isign_budget.sh check      # 1 min  - prove range access works
#   bash scripts/isl/run_isign_budget.sh prep       # ~1 h   - subset + LMDB + vocab
#   bash scripts/isl/run_isign_budget.sh calibrate  # ~5 min - measure real s/iter
#   bash scripts/isl/run_isign_budget.sh train      # the run you keep
# ---------------------------------------------------------------------------
set -euo pipefail

export SIGN2GPT_ROOT=${SIGN2GPT_ROOT:-/workspace/sonant-ISL-to-text}
export SIGN2GPT_DATA=${SIGN2GPT_DATA:-/workspace/data}
export SIGN2GPT_CKPT_PATH=${SIGN2GPT_CKPT_PATH:-/workspace/checkpoints}
export SIGN2GPT_LMDB_PATH=${SIGN2GPT_LMDB_PATH:-/workspace/lmdb}
export SIGN2GPT_RESULTS=${SIGN2GPT_RESULTS:-/workspace/results}

ISL_DATA="${SIGN2GPT_ROOT}/data/isl"
PKL="${ISL_DATA}/processed_words.isl_pkl"
LMDB_VIDEOS="${SIGN2GPT_LMDB_PATH}/isl/lmdb_videos"

# --- the three knobs that set the bill --------------------------------------
CLIPS=${CLIPS:-5000}               # clips actually trained on
CANDIDATES=${CANDIDATES:-9000}     # over-request: short-clip filter rejects many
MAX_SRC_FRAMES=${MAX_SRC_FRAMES:-130}   # ~5.2 s at 25 fps -> ~65 tokens
MAX_WORDS=${MAX_WORDS:-15}
export ISIGN_S1_EPOCHS=${ISIGN_S1_EPOCHS:-12}
export ISIGN_S2_EPOCHS=${ISIGN_S2_EPOCHS:-12}

mkdir -p "${ISL_DATA}" "${LMDB_VIDEOS}" "${SIGN2GPT_RESULTS}" "${SIGN2GPT_CKPT_PATH}"
log() { echo -e "\n\033[1;36m[budget] $*\033[0m"; }

deps() {
  pip install --no-cache-dir -q \
      albumentations==1.4.13 numpy==1.24.4 pandas==2.0.1 transformers==4.31.0 \
      lmdb==1.2.1 timm==0.9.16 requests ml-collections==0.1.1 \
      pytorch-ignite==0.4.13 Pillow==9.0.1 matplotlib nlpaug==1.1.11 nltk==3.6.7 \
      fasttext==0.9.2 sentencepiece==0.1.99 einops==0.8.0 mediapipe==0.10.5 \
      onnxscript albucore==0.0.13 spacy==3.7.4 opencv-python-headless==4.8.1.78 \
      "huggingface_hub[cli]"
  pip install --no-cache-dir -q --no-deps xformers==0.0.27.post2
  pip install --no-cache-dir -q \
      https://github.com/explosion/spacy-models/releases/download/en_core_web_lg-3.7.1/en_core_web_lg-3.7.1-py3-none-any.whl
}

check() {
  log "range-access self-test (spends about a minute and a few MB)"
  cd "${SIGN2GPT_ROOT}"
  python scripts/isl/isign_http_zip.py
  echo
  echo "If that printed the member count, --hf_repo works and you can skip the 58 GB download."
  echo "If it failed, fall back to:  huggingface-cli download Exploration-Lab/iSign \\"
  echo "  --repo-type dataset --include 'iSign-videos_v1.1_part_*' --local-dir \$SIGN2GPT_DATA/isign"
  echo "  ...then pass --zip_parts instead of --hf_repo below (needs an 80 GB volume)."
}

prep() {
  log "1/5 deps"; deps

  log "2/5 text CSV (10 MB) - the only file we download whole"
  cd "${SIGN2GPT_ROOT}"
  if [ ! -f "${SIGN2GPT_DATA}/isign/iSign_v1.1.csv" ]; then
    huggingface-cli download Exploration-Lab/iSign --repo-type dataset \
      --include "iSign_v1.1.csv" --local-dir "${SIGN2GPT_DATA}/isign"
  fi

  log "3/5 candidate subset (${CANDIDATES} clips, <=${MAX_WORDS} words)"
  python scripts/isl/isign_build_subset.py \
      --csv "${SIGN2GPT_DATA}/isign/iSign_v1.1.csv" \
      --output_dir "${ISL_DATA}" \
      --max_clips "${CANDIDATES}" --max_words "${MAX_WORDS}" --seed 1

  log "4/5 stream short clips out of the HF zip -> LMDB (stop at ${CLIPS})"
  python scripts/isl/isign_zip_to_lmdb.py \
      --hf_repo Exploration-Lab/iSign \
      --manifest "${ISL_DATA}/needed_videos.txt" \
      --lmdb_root "${LMDB_VIDEOS}" \
      --max_src_frames "${MAX_SRC_FRAMES}" \
      --stop_after "${CLIPS}" \
      --target_fps 25 --size 256 --workers "$(nproc)"

  log "5/5 prune CSVs to what exists, then build the vocabulary from THAT"
  python scripts/isl/isign_sync_csvs.py --csv_dir "${ISL_DATA}" --lmdb_root "${LMDB_VIDEOS}"
  rm -f "${PKL}"                       # must match the pruned CSVs
  python scripts/pseudo_gloss_en.py --csv_dir "${ISL_DATA}" --output_pkl "${PKL}"

  log "preflight"
  python scripts/isl/isign_preflight.py \
      --csv_dir "${ISL_DATA}" --pkl "${PKL}" --lmdb_root "${LMDB_VIDEOS}"
  du -sh "${LMDB_VIDEOS}" || true
}

calibrate() {
  log "running stage 1 briefly to measure real seconds/iteration"
  cd "${SIGN2GPT_ROOT}"
  SIGN2GPT_CKPT_PATH=/workspace/checkpoints_cal timeout 600 \
    python main.py --config=configs/isl/isign_budget_stage1_config.py \
    2>&1 | tee "${SIGN2GPT_RESULTS}/calibrate.log" || true
  rm -rf /workspace/checkpoints_cal
  echo
  echo "Read the iteration rate off the log above, then:"
  echo "  python scripts/isl/isign_budget.py --budget_usd <what you'll spend> \\"
  echo "     --gpu_usd_per_hour <pod rate> --train_clips <from preflight> \\"
  echo "     --dev_clips <from preflight> --bs <from the [config] line> \\"
  echo "     --s1_sec_per_iter <measured>"
  echo "Then set ISIGN_S1_EPOCHS / ISIGN_S2_EPOCHS and run 'train'."
}

train() {
  cd "${SIGN2GPT_ROOT}"
  log "stage 1 (${ISIGN_S1_EPOCHS} epochs)"
  python main.py --config=configs/isl/isign_budget_stage1_config.py \
      2>&1 | tee -a "${SIGN2GPT_RESULTS}/budget_stage1.log"

  log "stage 2 (${ISIGN_S2_EPOCHS} epochs, LM=${ISIGN_LM:-facebook/xglm-564M})"
  python main.py --config=configs/isl/isign_budget_stage2_config.py \
      2>&1 | tee -a "${SIGN2GPT_RESULTS}/budget_stage2.log"

  log "results"
  grep -h "valid/class_f1_score" "${SIGN2GPT_RESULTS}/budget_stage1.log" | tail -3 || true
  grep -h "valid_test/ableu"     "${SIGN2GPT_RESULTS}/budget_stage2.log" | tail -3 || true
  echo "Report valid_test/ableu. valid/obleu is teacher-forced and flatters."
  echo "STOP THE POD NOW - it bills while idle."
}

case "${1:-}" in
  check)     check ;;
  prep)      prep ;;
  calibrate) calibrate ;;
  train)     train ;;
  *) echo "usage: $0 {check|prep|calibrate|train}"; exit 1 ;;
esac
