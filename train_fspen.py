#!/usr/bin/env python3
"""
Unified FSPEN training script (replaces notebooks/train_fspen_*.ipynb).

CLI only:
  python train_fspen.py --config TrainConfig_48kHz_overlap --epochs 50

YAML + CLI overrides (common ML workflow):
  python train_fspen.py --train-config configs/train/fspen_48khz_overlap.yaml
  python train_fspen.py -c configs/train/fspen_48khz_overlap.yaml --epochs 10 --device cpu
"""

from __future__ import annotations

import os


import random
import sys

import numpy as np
import torch
from torch.optim import Adam

from src.reproducibility import enable_determinism
from src.train_cli import dump_train_config, parse_train_args
from src.training import (
    DEFAULT_MODEL_FOR_CONFIG,
    EVAL_FUNCTIONS,
    MetricsContext,
    build_dataloaders,
    build_model,
    default_eval_fn_for_config,
    learning_loop,
    load_config_class,
)


def _print_registry() -> None:
    from src.training import (
        CONFIG_NAMES,
        DEFAULT_MODEL_FOR_CONFIG,
        EVAL_FUNCTIONS,
        MODEL_CLASSES,
    )

    print("Available configs and default model/eval pairs:\n")
    for name in CONFIG_NAMES:
        model = DEFAULT_MODEL_FOR_CONFIG[name]
        eval_fn = default_eval_fn_for_config(name)
        print(f"  {name}")
        print(f"    default model: {model}")
        print(f"    default eval:  {eval_fn}")
    print("\nModel classes:", ", ".join(MODEL_CLASSES))
    print("Eval functions:", ", ".join(EVAL_FUNCTIONS))
    print("\nExample YAML configs in configs/train/")


def main() -> None:
    cfg = parse_train_args()

    if cfg["list_configs"]:
        _print_registry()
        return

    if cfg["dump_config"]:
        print(dump_train_config(cfg), end="")
        return

    if not cfg["config"]:
        print(
            "Error: model config is required. Use --config or --train-config YAML.",
            file=sys.stderr,
        )
        sys.exit(2)

    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.", file=sys.stderr)
        device = torch.device("cpu")

    if cfg["deterministic"]:
        enable_determinism(cfg["seed"])
        # if cfg["num_workers"] != 0:
        #     print(
        #         f"deterministic=True: forcing num_workers 0 (was {cfg['num_workers']})",
        #         file=sys.stderr,
        #     )
        #     cfg["num_workers"] = 0
    else:
        np.random.seed(cfg["seed"])
        torch.manual_seed(cfg["seed"])
        random.seed(cfg["seed"])

    model_class = cfg["model"] or DEFAULT_MODEL_FOR_CONFIG[cfg["config"]]
    eval_fn_name = cfg["eval_fn"] or default_eval_fn_for_config(cfg["config"])
    eval_fn = EVAL_FUNCTIONS[eval_fn_name]

    config_cls = load_config_class(cfg["config"])
    configs = config_cls()
    n_fft = configs.n_fft
    hop_length = configs.hop_length
    sample_rate = configs.sample_rate
    max_seq_len = int(sample_rate * cfg["max_seq_len_sec"])

    model = build_model(model_class, configs).to(device)
    optimizer = Adam(model.parameters(), lr=cfg["lr"])
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg["scheduler_step"],
        gamma=cfg["scheduler_gamma"],
    )

    starting_epoch = 0
    plots = None
    ckpt_model_name = cfg["model_name"]

    if cfg["resume"]:
        state = torch.load(cfg["resume"], map_location=device, weights_only=False)
        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        if state.get("scheduler_state_dict") and scheduler is not None:
            scheduler.load_state_dict(state["scheduler_state_dict"])
        plots = state.get("plots")
        starting_epoch = int(state.get("epoch", 0))
        print(f"Resumed from {cfg['resume']} at epoch {starting_epoch}")

    # NISQA/STOI before datasets — same order as train_fspen_*.ipynb notebooks.
    metrics = MetricsContext(
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
        nisqa_config=cfg["nisqa_config"],
        enable_nisqa=not cfg["no_nisqa"],
    )

    train_dataset, val_dataset, train_loader, val_loader = build_dataloaders(
        data_dir=cfg["data_dir"],
        val_data_dir=cfg["val_data_dir"],
        noise_dir_train=cfg["noise_dir_train"],
        noise_dir_val=cfg["noise_dir_val"],
        rir_dirs_train=cfg["rir_dirs_train"],
        rir_dirs_val=cfg["rir_dirs_val"],
        sample_rate=sample_rate,
        snr=cfg["snr"],
        noise_proba=cfg["noise_proba"],
        rir_proba=cfg["rir_proba"],
        max_seq_len=max_seq_len,
        val_partition=cfg["val_partition"],
        batch_size=cfg["batch_size"],
        num_workers=cfg["num_workers"],
        n_fft=n_fft,
        hop_length=hop_length,
        device=device,
        seed=cfg["seed"],
    )

    if cfg["warmup_dataset_epoch"]:
        train_dataset.set_epoch(1)
        val_dataset.set_epoch(1)

    if ckpt_model_name is None:
        ckpt_model_name = f"{cfg['config']}_{cfg['seed']}"

    if cfg["train_config"]:
        print(f"Loaded training config: {cfg['train_config']}")

    print(
        f"Training {model_class} with {cfg['config']} on {device}\n"
        f"  eval_fn={eval_fn_name}\n"
        f"  deterministic={cfg['deterministic']} seed={cfg['seed']} num_workers={cfg['num_workers']}\n"
        f"  train samples={len(train_dataset)} val samples={len(val_dataset)}\n"
        f"  SR={sample_rate} n_fft={n_fft} hop={hop_length} max_len={max_seq_len}\n"
        f"  checkpoint name={ckpt_model_name}"
    )

    learning_loop(
        model,
        configs,
        optimizer,
        train_loader,
        val_loader,
        train_dataset,
        val_dataset,
        scheduler,
        metrics,
        eval_fn,
        epochs=cfg["epochs"],
        val_every=cfg["val_every"],
        model_name=ckpt_model_name,
        chkp_folder=cfg["chkp_dir"],
        plots=plots,
        starting_epoch=starting_epoch,
        device=device,
        n_fft=n_fft,
        hop_length=hop_length,
        plot_every=cfg["plot_every"],
    )


if __name__ == "__main__":
    main()
