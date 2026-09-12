"""
Translate one ISL video with a trained stage-2 checkpoint.

Why not scripts/infer_video.py? That script hard-codes the PHOENIX repo root
and a German checkpoint, and it resizes frames with a plain cv2.resize. Our
iSign LMDBs are built with pad-to-square, so a plain resize at inference would
feed the model a differently-shaped world than it trained on. This script
mirrors the training preprocessing exactly:

    decode -> 25 fps -> pad to square -> 256 -> the config's own valid
    transform (Resize 224 + ImageNet normalise) -> stride-2 subsample

and builds the model from the same config object the trainer used, so there is
no second copy of the architecture to drift out of sync.

Usage:
    python scripts/isl/isign_infer.py \
        --config configs.isl.isign_budget_stage2_config \
        --video  /workspace/trial_clips/<some_clip>.mp4

    # a specific checkpoint instead of the best one
    python scripts/isl/isign_infer.py --config ... --video ... --ckpt /path/to.pt
"""

import argparse
import importlib
import sys
from pathlib import Path

import cv2
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from augmentation.get_aug import get_aug                      # noqa: E402
from models.get_models import get_model                       # noqa: E402
from train_utils.checkpoint_helpers import get_best_checkpoint_details  # noqa: E402
from transformers import AutoTokenizer                        # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from isign_zip_to_lmdb import pad_to_square                   # noqa: E402


def read_video(path, target_fps=25, size=256):
    """Same decode path as isign_zip_to_lmdb.video_to_jpegs, minus the JPEG hop."""
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not src_fps or src_fps <= 1 or src_fps > 240:
        src_fps = 25.0
    step = max(1.0, src_fps / float(target_fps))

    frames, i, want = [], 0, 0.0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if i >= want:
            want += step
            bgr = pad_to_square(bgr)
            if bgr.shape[0] != size:
                interp = cv2.INTER_AREA if bgr.shape[0] > size else cv2.INTER_LINEAR
                bgr = cv2.resize(bgr, (size, size), interpolation=interp)
            frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        i += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path}")
    print(f"[infer] {Path(path).name}: {i} source frames at {src_fps:.1f} fps "
          f"-> {len(frames)} kept at {target_fps} fps")
    return frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True,
                    help="Config MODULE path, e.g. configs.isl.isign_budget_stage2_config")
    ap.add_argument("--video", required=True)
    ap.add_argument("--ckpt", default=None, help="Override the best-checkpoint lookup")
    ap.add_argument("--num_beams", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    cfg = importlib.import_module(args.config).get_config()

    ckpt = args.ckpt or get_best_checkpoint_details(
        cfg.save_dir, best_checkpoint_name="_result_checkpoint_")[0]
    if not ckpt:
        sys.exit(f"[err] no stage-2 checkpoint under {cfg.save_dir}")
    print(f"[infer] checkpoint: {ckpt}")
    print(f"[infer] LM: {cfg.lm_name}")

    tok = AutoTokenizer.from_pretrained(cfg.lm_name)
    pretext_tokens = tok(cfg.pretext)["input_ids"]
    if cfg.pretext and cfg.pretext[-1] == " ":
        pretext_tokens = pretext_tokens[:-1]
    pretext_length = len(pretext_tokens) if cfg.pretext else 1

    params = dict(cfg.model_params)
    params["pretext_length"] = pretext_length
    # The stage-2 checkpoint already contains the stage-1 weights, so skip the
    # separate stage-1 load and its disk read.
    params["stage1_ckpt"] = None
    model = get_model(cfg.model_name, params)

    state = torch.load(ckpt, map_location="cpu")
    sd = state.get("model", state)
    sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in sd.items()}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing:
        print(f"[infer] {len(missing)} missing keys (first: {missing[:3]})")
    if unexpected:
        print(f"[infer] {len(unexpected)} unexpected keys (first: {unexpected[:3]})")
    model.eval().to(args.device)

    frames = read_video(args.video)
    transform = get_aug(cfg.aug_name, dict(cfg.aug_params))
    sel = np.arange(0, len(frames), transform.stride).astype(int)
    if len(sel) > transform.max_seq_len:
        sel = np.sort(np.random.choice(sel, transform.max_seq_len, replace=False))
    tensor = transform.aug_video([frames[i] for i in sel], isValid=True)
    print(f"[infer] {len(sel)} tokens -> {tuple(tensor.shape)}")

    frame_features = [tensor.to(args.device)]
    frame_mask = torch.ones(1, len(sel), dtype=torch.bool, device=args.device)
    text_ids = torch.tensor([pretext_tokens if cfg.pretext else [tok("")["input_ids"][0]]],
                            device=args.device)
    text_mask = torch.ones_like(text_ids, dtype=torch.bool)

    gen = {**dict(cfg.gen_params),
           "eos_token_id": tok.eos_token_id,
           "bos_token_id": tok.bos_token_id,
           "pad_token_id": tok.pad_token_id}
    if args.num_beams:
        gen["num_beams"] = args.num_beams

    with torch.inference_mode(True):
        autocast = torch.autocast(device_type="cuda", dtype=torch.float16) \
            if args.device == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with autocast:
            out = model(text_ids=text_ids, text_mask=text_mask,
                        frame_features=frame_features, frame_mask=frame_mask,
                        max_len=torch.tensor(512), generate=True, gen_params=gen)

    ids = out["output_ids"][0]
    eos = (ids == tok.eos_token_id).nonzero()
    if len(eos):
        ids = ids[: eos[0, 0]]
    text = tok.decode(ids, skip_special_tokens=True).strip()

    print("\n" + "=" * 60)
    print(f"PREDICTION: {text!r}")
    print("=" * 60)
    print("\nFor a compatibility trial the question is not whether this is CORRECT")
    print("- with a few hundred clips it will not be - but whether it is fluent")
    print("English at all. Fluent-but-wrong means the frozen LM survived and the")
    print("wiring is sound. Gibberish or empty means something upstream is broken.")


if __name__ == "__main__":
    main()
