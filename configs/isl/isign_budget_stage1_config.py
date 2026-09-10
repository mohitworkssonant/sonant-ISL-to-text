"""
Stage 1 (vision pretraining) — $7-budget iSign run on a 24 GB GPU.

This is the isign10k config re-cut for a hard money ceiling. What changed and
why, in order of how much money each choice saves:

  SHORT CLIPS      Training cost is proportional to (clips x tokens-per-clip).
  (in prep, not    Restricting the subset to clips under ~130 source frames
   this file)      halves tokens-per-clip (~65 vs ~107) for the same number of
                   examples. It also makes the task easier - fewer signs to
                   align per sentence - which is what you want when the budget
                   only allows a small model-fitting run.

  max_seq_len 128  ~65 tokens is the norm after the short-clip filter, so 128
                   only clips outliers. Lowering it from 256 bounds the worst-
                   case activation spike, which is what actually decides
                   whether bs=2 fits in 24 GB.

  bs 2 x accum 4   24 GB, not 80. Effective batch stays 8, so the optimisation
                   dynamics still match the German validation run. Override
                   with ISIGN_BS once you've watched `nvidia-smi` for a few
                   hundred steps.

  max_epochs 12    Enough for `class_f1` to clearly leave zero, which is the
                   only question stage 1 has to answer at this budget. Cheaper
                   than 20 by ~40%.

  warmup 2         Scaled to the 12-epoch schedule.

Unchanged on purpose: DINOv2 ViT-S/14 + Metaformer, the FastText prototype
head, stride 2, the BCE loss. Changing the model would make the result
incomparable with everything else in the project - and the point of a cheap
run is a comparable datapoint, not a different experiment.
"""

from ml_collections import config_dict
from pathlib import Path
import os
import numpy as np
import torch
from configs.base.base_utils import *


def get_config():
    cfg = config_dict.ConfigDict()
    cfg.name = Path(os.path.realpath(__file__)).stem
    base_name = Path(os.path.realpath(__file__)).parent.name
    code_path = str(Path(os.path.realpath(__file__)).resolve().parents[2])

    ckpt_path = get_checkpoint_path(base_name, cfg.name)
    lmdb_path = get_lmdb_path()
    cfg.save_dir = f"{ckpt_path}/{base_name}/{cfg.name}"

    cfg.main_runner = "trainer.psuedo_gloss_trainer"
    cfg.project_name = "isign_budget_pretrain"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,            # keep 12.5 token/s - do NOT raise this to save
                                # money; it trades away fingerspelling
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
    elif _vram_gb >= 40:        # A40 / A6000 / L40S
        cfg.bs = 4 * _ngpu
        cfg.accum = 2
    else:                       # 4090 / A5000 / 3090 - the budget targets
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
            "bce": {
                "cls_name": "losses.loss_functions.bce_loss",
                "loss_params": {
                    "src_name": ("dict_post_output", "logits"),
                    "tgt_name": "pseudo_gloss_ids",
                },
                "weight": 10.0,
            },
        }
    })

    cfg.max_epochs = int(os.environ.get("ISIGN_S1_EPOCHS", 12))
    cfg.model_checkpoint_dir = ""

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

    from configs.standards.standard_meta_model_zero_config import get_sign_encoder

    model_name, sign_model_params, dim_model = get_sign_encoder()
    cfg.model_name = model_name

    import pickle as _pickle
    with open(pkl_dir, "rb") as _f:
        _pg = _pickle.load(_f)
    _num_classes = len(_pg["dict_lem_to_id"])
    print(f"[config] budget run: num_classes = {_num_classes}, bs = {cfg.bs} x accum {cfg.accum}, "
          f"epochs = {cfg.max_epochs}, max_seq_len = {cfg.aug_params['max_seq_len']}")

    cfg.model_params = {
        "sign_model_name": "models.model_sign_encoder.basic_sign_encoder",
        "sign_model_params": sign_model_params,
        "post_name": "models.metaformer.post.zero_fasttext_prototype_head",
        "post_params": {
            "in_dim": dim_model,
            "hidden_dim": 300,
            "num_classes": _num_classes,
            "dropout": 0.2,
            "class_temperature": 0.1,
            "time_temperature": 0.1,
            "dynamic_time_temperatures": True,
            "dynamic_class_temperatures": True,
            "emb_lang": "en",
            "emb_pkl_dir": pkl_dir,
            "trainable_emb": False,
        },
    }

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.logger_name = ["text"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 50          # more frequent, so you can measure sec/iter early
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/class_f1_score"
    cfg.bfloat16_only = False
    cfg.mixup = False

    return cfg
