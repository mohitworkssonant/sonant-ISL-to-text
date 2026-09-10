"""
Stage 2 (translation) for the iSign 10k subset.

Loads the best stage-1 checkpoint automatically, freezes XGLM-1.7B and trains
only the LoRA adapters + the projection into the LM.

Differences from configs/isl/isl_stage2_config.py and WHY:

  stage1_name     -> configs.isl.isign10k_stage1_config (not the old ISL one).

  bs / accum      Same reasoning as stage 1, but stage 2 also holds the 1.7B LM
                  activations, so we keep bs=4 x accum=2 on an 80 GB card.

  max_epochs 20   PHOENIX used 10 epochs for BLEU-4 13.37. Two further reasons
                  for exactly 20 rather than 15 or 25: the trainer fires the
                  beam-search tester on Events.EPOCH_COMPLETED(every=10), so 20
                  gives you an honest `valid_test/ableu` at epoch 10 AND at the
                  end (15 would give only one, at epoch 10); and at ~1.3 GPU-h
                  per epoch, 20 is ~26 GPU-h, which keeps the whole run inside
                  a two-day, sub-$100 budget. Raise with ISIGN_S2_EPOCHS, but
                  decide before launching (cosine schedule, as in stage 1).

  max_length 64   `isign_build_subset.py` filters translations to <=30 words;
                  the XGLM tokenizer turns 30 English words into roughly 40
                  tokens, so 64 covers essentially every kept sentence. Both the
                  target-tokenisation length (`cfg.max_length`) and the
                  generation cap (`cfg.gen_params["max_length"]`) are set - if
                  you raise the word filter, raise BOTH or targets get truncated
                  while predictions do not, which quietly depresses BLEU.

Metric note: `valid/obleu` (used for checkpoint selection) is TEACHER-FORCED
and will read several points higher than the real number. The honest,
autoregressive beam-search score is `valid_test/ableu`, produced every 10
epochs by the tester engine. Quote `ableu`.
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
    cfg.project_name = "isign10k_translation"
    cfg.aug_name = "augmentation.video.base_video_aug"

    cfg.aug_params = {
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "strength": 0.2,
        "random_shift": 4,
        "stride": 2,
        "max_seq_len": 256,
    }

    # Probe VRAM defensively so the config can also be imported on a CPU-only
    # box (e.g. to eyeball paths before renting a GPU).
    _ngpu = max(1, torch.cuda.device_count())
    _vram_gb = (torch.cuda.get_device_properties(0).total_memory / 1e9
                if torch.cuda.is_available() else 0.0)
    _bs_env = os.environ.get("ISIGN_BS")
    if _bs_env:
        cfg.bs = int(_bs_env) * _ngpu
        cfg.accum = max(1, 8 // int(_bs_env))
    elif _vram_gb >= 70:
        cfg.bs = 4 * _ngpu
        cfg.accum = 2
    elif _vram_gb >= 40:
        cfg.bs = 2 * _ngpu
        cfg.accum = 4
    elif _vram_gb >= 22:
        cfg.bs = 1 * _ngpu
        cfg.accum = 8
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

    cfg.max_epochs = int(os.environ.get("ISIGN_S2_EPOCHS", 20))
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

    cfg.lm_name = "facebook/xglm-1.7B"     # multilingual, English native - no LM swap for ISL->EN
    cfg.additional_tokens = {}
    cfg.pretext = ""
    cfg.replacement_pickle = None

    cfg.stage1_name = "configs.isl.isign10k_stage1_config"
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
    print(f"[config] stage 2 will load stage-1 checkpoint: {stage1_ckpt}")

    cfg.model_name = "models.trial_models.test_stage2_model"
    cfg.model_params = {
        "stage1_name": stage1_name,
        "stage1_params": stage1_params,
        "stage1_ckpt": stage1_ckpt,
        "post_name": "models.post_models.linear_pos_head",
        "post_params": {"pos_type": "sine", "pre_pos": True},
        "llm_name": cfg.lm_name,
        "lang_backbone_name": "models.huggingface.modeling_xglm",
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

    # Target tokenisation length. Keep in step with gen_params["max_length"].
    cfg.max_length = 64
    cfg.gen_params = {"max_length": 64, "temperature": 1.0, "num_beams": 4}

    cfg.seed = 1
    cfg.grad_clip_norm = 1.0
    cfg.grad_clip_value = 1.0
    cfg.mixup = False
    cfg.logger_name = ["text"]
    cfg.resume = True
    cfg.train_length = None
    cfg.val_length = None
    cfg.log_every = 100
    cfg.save_ckpt = True
    cfg.score_factor = 1
    cfg.score_name = "valid/obleu"
    cfg.bfloat16_only = False
    cfg.apply_metric_splitter = False
    cfg.append_string = ""
    cfg.watch_grad = True
    return cfg
