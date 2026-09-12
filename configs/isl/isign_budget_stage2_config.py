"""
Stage 2 (translation) — $7-budget iSign run on a 24 GB GPU.

The one substantive change from the paper setting, and the reasoning:

  LM = facebook/xglm-564M  (was xglm-1.7B)
      Stage 2 is the expensive half: the LM's activations dominate both memory
      and time. XGLM-564M has the SAME 24 decoder layers, so `adapt_layers` /
      `lora_layers` = arange(0,24) is unchanged and no other code moves;
      `test_stage2_model` reads `lang_model.embed_dim` dynamically, so the
      projection resizes itself (2048 -> 1024).
      What you lose: a weaker language prior, so expect slightly less fluent
      output. What you keep: the thing this run is actually testing - whether
      sign tokens can steer a frozen multilingual LM into English at all.
      What you gain: it fits comfortably in 24 GB and roughly halves stage-2
      time, which at this budget is the difference between finishing and not.
      Set ISIGN_LM=facebook/xglm-1.7B to go back to the paper setting.

  max_epochs 12   The trainer runs the beam-search tester every 10 epochs, so
                  12 buys exactly one honest `valid_test/ableu` plus the
                  cheaper per-epoch teacher-forced `obleu` curve. Fewer than 10
                  would give you NO autoregressive number at all - that is the
                  floor, not 1.

  max_length 48   Pair this config with `--max_words 15` in the subset builder.
                  15 English words is ~22 XGLM tokens, so 48 is generous, and
                  a shorter generation budget makes beam search cheaper.
                  If you widen the word filter, raise BOTH `cfg.max_length` and
                  `cfg.gen_params["max_length"]`.

  num_beams 4     Left alone. Dropping to greedy would save a little eval time
                  but makes the number incomparable with the German run and
                  with iSign's published baselines.

Metric reminder: `valid/obleu` is TEACHER-FORCED (checkpoint selection only).
Report `valid_test/ableu`.
"""

from ml_collections import config_dict
from pathlib import Path
import os
import numpy as np
import torch
from configs.base.base_utils import *
import importlib
from train_utils.checkpoint_helpers import get_best_checkpoint_details


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[2])

    ckpt_path = get_checkpoint_path(base_name, cfg.name)
    lmdb_path = get_lmdb_path()
    cfg.save_dir = f"{ckpt_path}/{base_name}/{cfg.name}"

    cfg.main_runner = "trainer.complete_translation_trainer"
    cfg.project_name = "isign_budget_translation"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": int(os.environ.get("ISIGN_MAX_SEQ", 128)),
    }

    _ngpu = max(1, torch.cuda.device_count())
    _vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0.0)
    _bs_env = os.environ.get("ISIGN_BS")
    if _bs_env:
        cfg.bs = int(_bs_env) * _ngpu
        cfg.accum = max(1, 8 // int(_bs_env))
    elif _vram_gb >= 70:
        cfg.bs = 6 * _ngpu
        cfg.accum = 2
    elif _vram_gb >= 40:
        cfg.bs = 3 * _ngpu
        cfg.accum = 3
    else:                       # 24 GB with xglm-564M
        cfg.bs = 2 * _ngpu
        cfg.accum = 4
    cfg.num_workers = min(4 * _ngpu, 8)

    cfg.gate_grad_multiplier = 1.0

    cfg.lr = 3e-4
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict({
        "lr_scale_factor": 0.01,
        "num_cycles": 1,
        "start_value_mult": 0.7,
        "end_value_mult": 0.7,
        "warmup_epochs": 2,
    })

    cfg.optimizer_name = "adamw"
    cfg.optimizer_params = config_dict.ConfigDict({
        "lr": cfg.lr,
        "weight_decay": 0.001,
    })

    cfg.criterion_name = "losses.base_loss"
    cfg.criterion_params = config_dict.ConfigDict({
        "dict_of_loss_params": {
            "ce": {
                "cls_name": "losses.loss_functions.ce_loss",
                "loss_params": {
                    "src_name": "logits",
                    "tgt_name": "gt_ids",
                    "mask_name": "gt_text_mask",
                    "label_smoothing": 0.1,
                },
                "weight": 1.0,
            },
        }
    })

    cfg.max_epochs = int(os.environ.get("ISIGN_S2_EPOCHS", 12))
    cfg.model_checkpoint_dir = ""
    # A cosine schedule sized as (max_epochs - warmup_epochs) collapses to a
    # zero-length cycle when the two are equal, which surfaces as an opaque
    # ZeroDivisionError deep in ignite rather than a config error. Someone
    # trying "1 epoch just to test" hits this immediately, so fail loudly here.
    _warm = cfg.lr_scheduler_params["warmup_epochs"]
    assert cfg.max_epochs > _warm, (
        f"max_epochs ({cfg.max_epochs}) must exceed warmup_epochs ({_warm}). "
        f"For a quick smoke test use {_warm + 1} or more, not 1."
    )


    data_dir = f"{code_path}/data/isl"
    pkl_dir = f"{data_dir}/processed_words.isl_pkl"
    lmdb_videos = f"{lmdb_path}/isl/lmdb_videos"

    cfg.train_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.valid_ds_name = "dataloaders.phoenix_video_dataset"
    cfg.test_ds_name = "dataloaders.phoenix_video_dataset"

    def _ds(split, is_valid, shuffle, drop_last):
        return config_dict.ConfigDict({
            "csv_dir": f"{data_dir}/ISL.{split}.corpus.csv",
            "pseudo_gloss_dir": pkl_dir,
            "sep": "|",
            "name": "translation",
            "ds_params": {"lmdb_video_dir": lmdb_videos, "isValid": is_valid},
            "shuffle": shuffle,
            "num_workers": cfg.num_workers,
            "bs": cfg.bs,
            "drop_last": drop_last,
        })

    cfg.train_ds_params = _ds("train", False, True, True)
    cfg.valid_ds_params = _ds("dev", True, False, False)
    cfg.test_ds_params = _ds("test", True, False, False)

    cfg.lm_name = os.environ.get("ISIGN_LM", "facebook/xglm-564M")
    cfg.additional_tokens = {}
    cfg.pretext = ""
    cfg.replacement_pickle = None

    cfg.stage1_name = "configs.isl.isign_budget_stage1_config"
    mod = importlib.import_module(cfg.stage1_name, package=None)
    stage1_config = mod.get_config()

    stage1_name = stage1_config["model_name"]
    stage1_params = stage1_config["model_params"].to_dict()
    stage1_ckpt_dir = stage1_config["save_dir"]
    stage1_ckpt = get_best_checkpoint_details(
        stage1_ckpt_dir, best_checkpoint_name="_result_checkpoint_"
    )[0]
    assert stage1_ckpt is not None and stage1_ckpt != "", (
        f"No stage-1 checkpoint under {stage1_ckpt_dir}. Run stage 1 to completion first."
    )
    print(f"[config] stage 2: LM = {cfg.lm_name}, bs = {cfg.bs} x accum {cfg.accum}, "
          f"epochs = {cfg.max_epochs}")
    print(f"[config] stage-1 checkpoint: {stage1_ckpt}")

    cfg.model_name = "models.trial_models.test_stage2_model"
    cfg.model_params = {
        "stage1_name": stage1_name,
        "stage1_params": stage1_params,
        "stage1_ckpt": stage1_ckpt,
        "post_name": "models.post_models.linear_pos_head",
        "post_params": {"pos_type": "sine", "pre_pos": True},
        "llm_name": cfg.lm_name,
        "lang_backbone_name": "models.huggingface.modeling_xglm",
        # XGLM-564M and XGLM-1.7B both have 24 decoder layers, so this range is
        # correct for either. If you ever point ISIGN_LM at a model with a
        # different depth, change these two lists to match.
        "adaptor_params": {
            "adapt_layers": list(np.arange(0, 24, 1)),
            "lora_layers": list(np.arange(0, 24, 1)),
            "w_lora_ff": False,
            "lora_rank": 4,
            "lora_drop": 0.1,
            "gate_type": "clamp",
            "lora_a": 4.0,
            "adapt_tokens": False,
        },
        "freeze": False,
    }

    cfg.max_length = int(os.environ.get("ISIGN_TXT_LEN", 48))
    cfg.gen_params = {"max_length": cfg.max_length, "temperature": 1.0, "num_beams": 4}

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.mixup = False
    cfg.logger_name = ["text"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 50
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/obleu"
    cfg.bfloat16_only = False
    cfg.apply_metric_splitter = False
    cfg.append_string = ""
    cfg.watch_grad = True
    return cfg
