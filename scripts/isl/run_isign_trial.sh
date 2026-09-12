#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# run_isign_trial.sh - COMPATIBILITY TRIAL: does iSign RGB video drive Sign2GPT?
#
# ~300 clips, both stages, one real translation, one PASS/FAIL report.
# About 45 minutes and well under $1 on an RTX 4090. This is NOT a training
# run: nothing it produces should be quoted as a quality result.
#
#   export HF_TOKEN=hf_xxx
#   bash scripts/isl/run_isign_trial.sh            # everything, in order
#   bash scripts/isl/run_isign_trial.sh prep       # or one phase at a time
#
# Checkpoints go to /workspace/checkpoints_trial, deliberately separate from
# the real run: the trial's pseudo-gloss vocabulary is smaller, so num_classes
# differs and a trial stage-1 checkpoint must never be picked up by a real
# stage 2.
# ---------------------------------------------------------------------------
set -euo pipefail

export SIGN2GPT_ROOT=${SIGN2GPT_ROOT:-/workspace/sonant-ISL-to-text}
export SIGN2GPT_DATA=${SIGN2GPT_DATA:-/workspace/data}
export SIGN2GPT_CKPT_PATH=${SIGN2GPT_CKPT_PATH:-/workspace/checkpoints_trial}
export SIGN2GPT_LMDB_PATH=${SIGN2GPT_LMDB_PATH:-/workspace/lmdb}
export SIGN2GPT_RESULTS=${SIGN2GPT_RESULTS:-/workspace/results}

ISL_DATA="${SIGN2GPT_ROOT}/data/isl"
PKL="${ISL_DATA}/processed_words.isl_pkl"
LMDB_VIDEOS="${SIGN2GPT_LMDB_PATH}/isl/lmdb_videos"
CLIPS_DIR="${SIGN2GPT_DATA}/trial_clips"

TRIAL_CLIPS=${TRIAL_CLIPS:-300}       # clips actually trained on
CANDIDATES=${CANDIDATES:-700}         # over-request; the length filter rejects many
MAX_SRC_FRAMES=${MAX_SRC_FRAMES:-130}
MAX_WORDS=${MAX_WORDS:-15}

# 6 and 10 are the minimum that make the trial meaningful:
#  - warmup is 2 epochs, so anything <=2 never leaves warmup (and a cosine
#    schedule with max_epochs == warmup_epochs divides by zero);
#  - the beam-search tester fires on EPOCH_COMPLETED(every=10), so stage 2
#    must reach 10 or you get no autoregressive BLEU at all.
export ISIGN_S1_EPOCHS=${ISIGN_S1_EPOCHS:-6}
export ISIGN_S2_EPOCHS=${ISIGN_S2_EPOCHS:-10}

mkdir -p "${ISL_DATA}" "${LMDB_VIDEOS}" "${SIGN2GPT_RESULTS}" "${SIGN2GPT_CKPT_PATH}" "${CLIPS_DIR}"
log() { echo -e "\n\033[1;36m[trial] $*\033[0m"; }

deps() {
  log "1/8 installing dependencies (~5 min, once per pod)"
  bash "${SIGN2GPT_ROOT}/scripts/isl/isign_deps.sh"
}

gpucheck() {
  log "1b/8 is this GPU usable with the pinned torch build?"
  python - <<'PY'
import sys
import torch
if not torch.cuda.is_available():
    sys.exit("[gpucheck] FAIL: no GPU visible. Terminate this pod and deploy one with a GPU.")
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
vram = torch.cuda.get_device_properties(0).total_memory / 1e9
print(f"[gpucheck] {name} | compute capability {cap[0]}.{cap[1]} | {vram:.0f} GB | torch {torch.__version__}")

# This repo pins torch 2.4.x / xformers 0.0.27.post2, whose kernels are built
# for sm_70..sm_90. Blackwell is sm_100/sm_120 and fails at the first matmul
# with "no kernel image is available", ~20 minutes into a run if unchecked.
if cap[0] >= 10:
    sys.exit(f"[gpucheck] FAIL: {name} is Blackwell-class (sm_{cap[0]}{cap[1]}). "
             "The pinned torch build has no kernels for it. Terminate and pick "
             "an Ada, Ampere or Hopper card (4090/A5000/A6000/A40/L4/L40S/A100/H100).")

try:
    x = torch.randn(64, 64, device="cuda", dtype=torch.float16)
    _ = (x @ x).sum().item()
except Exception as e:
    sys.exit(f"[gpucheck] FAIL: a trivial GPU matmul errored -> {type(e).__name__}: {e}\n"
             "This card is not compatible with the pinned torch build. Pick another.")

print(f"[gpucheck] bf16 supported: {torch.cuda.is_bf16_supported()}"
      + ("" if torch.cuda.is_bf16_supported() else
         "  (falls back to fp16 + GradScaler - works, but less exercised than the bf16 path)"))

if vram < 18:
    print(f"[gpucheck] {vram:.0f} GB is tight. Run with:  export ISIGN_BS=1 ISIGN_MAX_SEQ=96")
elif vram < 22:
    print(f"[gpucheck] {vram:.0f} GB. If stage 2 hits OOM:  export ISIGN_BS=1")
else:
    print("[gpucheck] VRAM is fine for the default settings.")
print("[gpucheck] PASS")
PY
}

check() {
  log "2/8 can we read the iSign zip by HTTP range? (a few MB)"
  cd "${SIGN2GPT_ROOT}" && python scripts/isl/isign_http_zip.py
}

prep() {
  cd "${SIGN2GPT_ROOT}"
  log "3/8 iSign text CSV (10 MB)"
  [ -f "${SIGN2GPT_DATA}/isign/iSign_v1.1.csv" ] || \
    huggingface-cli download Exploration-Lab/iSign --repo-type dataset \
      --include "iSign_v1.1.csv" --local-dir "${SIGN2GPT_DATA}/isign"

  log "4/8 pick ${CANDIDATES} candidate clips (<=${MAX_WORDS} words)"
  python scripts/isl/isign_build_subset.py \
      --csv "${SIGN2GPT_DATA}/isign/iSign_v1.1.csv" \
      --output_dir "${ISL_DATA}" \
      --max_clips "${CANDIDATES}" --max_words "${MAX_WORDS}" --seed 1

  log "5/8 stream ${TRIAL_CLIPS} short clips out of the zip into LMDB"
  python scripts/isl/isign_zip_to_lmdb.py \
      --hf_repo Exploration-Lab/iSign \
      --manifest "${ISL_DATA}/needed_videos.txt" \
      --lmdb_root "${LMDB_VIDEOS}" \
      --max_src_frames "${MAX_SRC_FRAMES}" --stop_after "${TRIAL_CLIPS}" \
      --target_fps 25 --size 256 --workers "$(nproc)"

  log "6/8 prune CSVs to what exists, build vocabulary from THAT, preflight"
  python scripts/isl/isign_sync_csvs.py --csv_dir "${ISL_DATA}" --lmdb_root "${LMDB_VIDEOS}"
  rm -f "${PKL}"
  python scripts/pseudo_gloss_en.py --csv_dir "${ISL_DATA}" --output_pkl "${PKL}"
  python scripts/isl/isign_preflight.py \
      --csv_dir "${ISL_DATA}" --pkl "${PKL}" --lmdb_root "${LMDB_VIDEOS}"
}

train() {
  cd "${SIGN2GPT_ROOT}"
  log "7/8 stage 1 - vision (${ISIGN_S1_EPOCHS} epochs)"
  python main.py --config=configs/isl/isign_budget_stage1_config.py \
      2>&1 | tee "${SIGN2GPT_RESULTS}/trial_stage1.log"

  log "7/8 stage 2 - translation (${ISIGN_S2_EPOCHS} epochs)"
  python main.py --config=configs/isl/isign_budget_stage2_config.py \
      2>&1 | tee "${SIGN2GPT_RESULTS}/trial_stage2.log"
}

verify() {
  cd "${SIGN2GPT_ROOT}"
  log "8/8 translate a held-out clip and print the verdict"
  python scripts/isl/isign_fetch_clip.py \
      --hf_repo Exploration-Lab/iSign \
      --from_csv "${ISL_DATA}/ISL.test.corpus.csv" --n 2 --out_dir "${CLIPS_DIR}"

  local clip
  clip=$(find "${CLIPS_DIR}" -name '*.mp4' | head -1)
  if [ -n "${clip}" ]; then
    python scripts/isl/isign_infer.py \
        --config configs.isl.isign_budget_stage2_config \
        --video "${clip}" 2>&1 | tee "${SIGN2GPT_RESULTS}/trial_infer.log"
  else
    echo "[trial] no clip fetched - skipping inference"
  fi

  local pred=""
  if [ -f "${SIGN2GPT_RESULTS}/trial_infer.log" ]; then
    pred=$(grep -oP "PREDICTION: \K.*" "${SIGN2GPT_RESULTS}/trial_infer.log" | tail -1 | sed "s/^'//;s/'$//")
  fi

  python scripts/isl/isign_trial_report.py \
      --csv_dir "${ISL_DATA}" --pkl "${PKL}" --lmdb_root "${LMDB_VIDEOS}" \
      --s1_log "${SIGN2GPT_RESULTS}/trial_stage1.log" \
      --s2_log "${SIGN2GPT_RESULTS}/trial_stage2.log" \
      ${pred:+--prediction "${pred}"} | tee "${SIGN2GPT_RESULTS}/trial_report.txt"

  echo
  echo "Report saved to ${SIGN2GPT_RESULTS}/trial_report.txt"
  echo "STOP THE POD NOW - RunPod bills a running pod even when it is idle."
}

case "${1:-all}" in
  deps)     deps ;;
  gpucheck) gpucheck ;;
  check)    check ;;
  prep)     prep ;;
  train)    train ;;
  verify)   verify ;;
  all)      deps; gpucheck; check; prep; train; verify ;;
  *) echo "usage: $0 {all|deps|gpucheck|check|prep|train|verify}"; exit 1 ;;
esac
