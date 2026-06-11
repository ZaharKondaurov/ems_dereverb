"""CLI + YAML configuration for train_fspen.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.training import CONFIG_NAMES, MODEL_CLASSES, EVAL_FUNCTIONS

DEFAULT_DATA_DIR = os.path.join("data", "DS_10283_2791/clean_trainset_56spk_wav")
DEFAULT_VAL_DATA_DIR = os.path.join("data", "DS_10283_2791/clean_trainset_28spk_wav")
DEFAULT_NOISE_TRAIN = os.path.join("data", "demand_train")
DEFAULT_NOISE_VAL = os.path.join("data", "demand_val")
DEFAULT_RIR_TRAIN = [
    os.path.join("data", "rirs48_small_3_train"),
    os.path.join("data", "rirs48_medium_3_train"),
    os.path.join("data", "rirs48_large_3_train"),
    os.path.join("data", "rirs48_super_large_3_train"),
]
DEFAULT_RIR_VAL = [
    os.path.join("data", "rirs48_small_3_val"),
    os.path.join("data", "rirs48_medium_3_val"),
    os.path.join("data", "rirs48_large_3_val"),
    os.path.join("data", "rirs48_super_large_3_val"),
]

# Built-in defaults when neither YAML nor CLI override a field.
BUILTIN_DEFAULTS: dict[str, Any] = {
    "config": None,
    "model": None,
    "eval_fn": None,
    "model_name": None,
    "resume": None,
    "epochs": 50,
    "batch_size": 32,
    "lr": 5e-4,
    "scheduler_step": 4,
    "scheduler_gamma": 0.98,
    "val_every": 1,
    "plot_every": 1,
    "num_workers": 4,
    "seed": 42,
    "device": "cuda",
    "chkp_dir": "checkpoints/fspen_chkp",
    "data_dir": DEFAULT_DATA_DIR,
    "val_data_dir": DEFAULT_VAL_DATA_DIR,
    "noise_dir_train": DEFAULT_NOISE_TRAIN,
    "noise_dir_val": DEFAULT_NOISE_VAL,
    "rir_dirs_train": DEFAULT_RIR_TRAIN,
    "rir_dirs_val": DEFAULT_RIR_VAL,
    "snr": [0, 5, 10, 15],
    "noise_proba": 0.85,
    "rir_proba": 0.85,
    "max_seq_len_sec": 4.0,
    "val_partition": 5000,
    "nisqa_config": "NISQA_s/config/nisqa_s.yaml",
    "no_nisqa": False,
    # Match notebooks that call set_epoch(1) on datasets before learning_loop.
    "warmup_dataset_epoch": False,
    # Full reproducibility: cudnn deterministic, num_workers forced to 0 in train_fspen.py.
    "deterministic": True,
}


def _normalize_list_field(value: Any, *, as_int: bool = False) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        items = list(value)
    else:
        raise ValueError(f"Expected list or comma-separated string, got {type(value)}")

    if as_int:
        return [int(x) for x in items]
    return [str(x) for x in items]


def normalize_train_config(raw: dict[str, Any]) -> dict[str, Any]:
    """Validate and normalize keys loaded from YAML."""
    unknown = set(raw) - set(BUILTIN_DEFAULTS)
    if unknown:
        known = ", ".join(sorted(BUILTIN_DEFAULTS))
        raise ValueError(
            f"Unknown keys in training config: {', '.join(sorted(unknown))}. "
            f"Known keys: {known}"
        )

    cfg = dict(BUILTIN_DEFAULTS)
    cfg.update(raw)

    if cfg["config"] is not None and cfg["config"] not in CONFIG_NAMES:
        raise ValueError(
            f"Unknown config {cfg['config']!r}. Choose one of: {CONFIG_NAMES}"
        )

    if cfg["model"] is not None and cfg["model"] not in MODEL_CLASSES:
        raise ValueError(
            f"Unknown model {cfg['model']!r}. Choose one of: {sorted(MODEL_CLASSES)}"
        )

    if cfg["eval_fn"] is not None and cfg["eval_fn"] not in EVAL_FUNCTIONS:
        raise ValueError(
            f"Unknown eval_fn {cfg['eval_fn']!r}. Choose one of: {sorted(EVAL_FUNCTIONS)}"
        )

    cfg["rir_dirs_train"] = _normalize_list_field(cfg["rir_dirs_train"])
    cfg["rir_dirs_val"] = _normalize_list_field(cfg["rir_dirs_val"])
    cfg["snr"] = _normalize_list_field(cfg["snr"], as_int=True)
    cfg["no_nisqa"] = bool(cfg["no_nisqa"])
    cfg["warmup_dataset_epoch"] = bool(cfg["warmup_dataset_epoch"])
    cfg["deterministic"] = bool(cfg["deterministic"])
    cfg["val_partition"] = int(cfg["val_partition"]) if cfg["val_partition"] else None
    return cfg


def load_train_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Training config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Training config must be a YAML mapping, got {type(raw)}")
    return normalize_train_config(raw)


_INTERNAL_KEYS = frozenset({"train_config", "dump_config", "list_configs"})


def dump_train_config(cfg: dict[str, Any]) -> str:
    export = {k: v for k, v in cfg.items() if k not in _INTERNAL_KEYS}
    return yaml.safe_dump(export, sort_keys=False, allow_unicode=True)


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train FSPEN speech enhancement models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "YAML workflow (common in ML projects):\n"
            "  python train_fspen.py --train-config configs/train/fspen_48khz_overlap.yaml\n"
            "  python train_fspen.py -c configs/train/fspen_48khz_overlap.yaml --epochs 10\n"
            "CLI flags always override values from the YAML file."
        ),
    )

    p.add_argument(
        "-c",
        "--train-config",
        dest="train_config",
        default=None,
        metavar="YAML",
        help="Path to YAML file with training hyperparameters",
    )
    p.add_argument(
        "--dump-config",
        action="store_true",
        help="Print merged effective config as YAML and exit",
    )
    p.add_argument(
        "--config",
        choices=CONFIG_NAMES,
        default=None,
        help="Model config class from src/fspen_configs.py",
    )
    p.add_argument(
        "--model",
        choices=sorted(MODEL_CLASSES),
        default=None,
        help="Model class from models/fspen.py",
    )
    p.add_argument(
        "--eval-fn",
        choices=sorted(EVAL_FUNCTIONS),
        default=None,
        help="Forward pass for training",
    )
    p.add_argument("--model-name", default=None, help="Checkpoint base name")
    p.add_argument("--resume", default=None, help="Resume from .pt checkpoint")
    p.add_argument(
        "--list-configs", action="store_true", help="List model configs and exit"
    )

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--scheduler-step", type=int, default=None)
    p.add_argument("--scheduler-gamma", type=float, default=None)
    p.add_argument("--val-every", type=int, default=None)
    p.add_argument("--plot-every", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--chkp-dir", default=None)

    p.add_argument("--data-dir", default=None)
    p.add_argument("--val-data-dir", default=None)
    p.add_argument("--noise-dir-train", default=None)
    p.add_argument("--noise-dir-val", default=None)
    p.add_argument(
        "--rir-dirs-train", default=None, help="Comma-separated RIR train dirs"
    )
    p.add_argument("--rir-dirs-val", default=None, help="Comma-separated RIR val dirs")
    p.add_argument("--snr", default=None, help="Comma-separated SNR values in dB")
    p.add_argument("--noise-proba", type=float, default=None)
    p.add_argument("--rir-proba", type=float, default=None)
    p.add_argument("--max-seq-len-sec", type=float, default=None)
    p.add_argument(
        "--val-partition", type=int, default=None, help="0 = use all val files"
    )

    p.add_argument("--nisqa-config", default=None)
    p.add_argument(
        "--no-nisqa", action="store_true", default=None, help="Disable NISQA metrics"
    )
    p.add_argument(
        "--warmup-dataset-epoch",
        action="store_true",
        default=None,
        help="Call set_epoch(1) on datasets before learning_loop (notebook parity)",
    )
    p.add_argument(
        "--deterministic",
        action="store_true",
        default=None,
        help="Enable deterministic training (disable via deterministic: false in YAML)",
    )

    return p


def parse_train_args(argv: Optional[list[str]] = None) -> dict[str, Any]:
    """
    Merge YAML defaults with CLI overrides.

    Pattern used in many ML codebases (lighter than Hydra):
    1. Load YAML via --train-config / -c
    2. Apply argparse defaults from YAML
    3. CLI flags override YAML
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--train-config", dest="train_config")
    pre.add_argument("--dump-config", action="store_true")
    pre.add_argument("--list-configs", action="store_true")
    pre_args, remaining = pre.parse_known_args(argv)

    yaml_defaults = (
        load_train_yaml(pre_args.train_config)
        if pre_args.train_config
        else dict(BUILTIN_DEFAULTS)
    )

    parser = build_argument_parser()
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(remaining)

    cfg = dict(yaml_defaults)
    for key, value in vars(args).items():
        if key in ("train_config", "dump_config", "list_configs"):
            continue
        if value is not None:
            cfg[key] = value

    if args.no_nisqa is True:
        cfg["no_nisqa"] = True
    if args.warmup_dataset_epoch is True:
        cfg["warmup_dataset_epoch"] = True
    if args.deterministic is True:
        cfg["deterministic"] = True

    if isinstance(cfg.get("rir_dirs_train"), str):
        cfg["rir_dirs_train"] = _normalize_list_field(cfg["rir_dirs_train"])
    if isinstance(cfg.get("rir_dirs_val"), str):
        cfg["rir_dirs_val"] = _normalize_list_field(cfg["rir_dirs_val"])
    if isinstance(cfg.get("snr"), str):
        cfg["snr"] = _normalize_list_field(cfg["snr"], as_int=True)

    cfg["train_config"] = pre_args.train_config
    cfg["dump_config"] = pre_args.dump_config or args.dump_config
    cfg["list_configs"] = pre_args.list_configs or args.list_configs
    cfg["val_partition"] = (
        int(cfg["val_partition"]) if cfg.get("val_partition") else None
    )

    return cfg
