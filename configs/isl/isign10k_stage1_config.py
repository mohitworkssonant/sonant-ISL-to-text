"""
Stage 1 (vision pretraining) for the iSign 10k subset.

Differences from configs/isl/isl_stage1_config.py and WHY:

  bs / accum      iSign clips average ~215 frames vs PHOENIX's ~120, so after
                  stride-2 each sample carries ~107 tokens instead of ~60.
                  Activation memory scales with that, so bs=8 (the PHOENIX
                  80 GB setting) can OOM. We use bs=4 x accum=2 -> the SAME
                  effective batch of 8, identical optimisation dynamics, lower
                  peak VRAM. If `nvidia-smi` shows headroom after a few
                  hundred steps, raise ISIGN_BS to 6 or 8.

  max_epochs 20   The point of this run is a decisive first ISL number, not a
                  SOTA one. PHOENIX reached BLEU-4 13.37 in 12 epochs (the paper
                  used 100), so the curve is already mostly flat by 20. On iSign
                  each epoch costs ~0.7 GPU-h, and epochs 20-60 would buy maybe
                  a point of BLEU for ~28 extra GPU-hours. Raise with
                  ISIGN_S1_EPOCHS if the first run justifies it - but decide
                  BEFORE launching: the cosine schedule is defined over
                  max_epochs, so changing it mid-run changes the LR curve.

  warmup 3        Scaled with the shorter schedule (was 5 of 60).

Everything else - the DINOv2 ViT-S/14 + Metaformer encoder, the FastText
prototype head, the BCE pseudo-gloss loss - is untouched, so a result here is
comparable with the German validation run.
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
    cfg.project_name = "isign10k_pretrain"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,          # 25 fps -> 12.5 fps of tokens; below this,
                              # fingerspelling and fast transitions get lost
        "max_seq_len": 256,   # iSign p90 = 371 frames -> 186 tokens, still under
    }

    # ---- batch sizing -----------------------------------------------------
    # Probe VRAM defensively so the config can also be imported on a CPU-only
    # box (e.g. to eyeball paths before renting a GPU).
    _ngpu = max(1, torch.cuda.device_count())
    _vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0.0)
    _bs_env = os.environ.get("ISIGN_BS")
    if _bs_env:
        cfg.bs = int(_bs_env) * _ngpu
        cfg.accum = max(1, 8 // int(_bs_env))
    elif _vram_gb >= 70:          # A100 80GB / H100
        cfg.bs = 4 * _ngpu
        cfg.accum = 2
    elif _vram_gb >= 40:          # A100 40GB / A6000 / L40S
        cfg.bs = 3 * _ngpu
        cfg.accum = 3
    elif _vram_gb >= 22:          # 4090 / L4 / A5000
        cfg.bs = 2 * _ngpu
        cfg.accum = 4
    else:
        cfg.bs = 1 * _ngpu
        cfg.accum = 8
    cfg.num_workers = min(min(cfg.bs * 2, int(10 * _ngpu)), 10)

    cfg.gate_grad_multiplier = 1.0

    cfg.lr = 3e-4
    cfg.lr_scheduler = "warmupwithcosine"
    cfg.lr_scheduler_params = config_dict.ConfigDict({
        "lr_scale_factor": 0.01,
        "num_cycles": 1,
        "start_value_mult": 0.7,
        "end_value_mult": 0.7,
        "warmup_epochs": 3,
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

    cfg.max_epochs = int(os.environ.get("ISIGN_S1_EPOCHS", 20))
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
    print(f"[config] iSign10k num_classes = {_num_classes} (from {pkl_dir})")

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
            "emb_lang": "en",                 # FastText cc.en.300.bin
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
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/class_f1_score"
    cfg.bfloat16_only = False
    cfg.mixup = False

    return cfg
