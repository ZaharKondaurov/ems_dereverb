"""CLI + YAML configuration for test_fspen.py."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Optional

import yaml

from src.fspen_eval import DEFAULT_NOISE_TEST, DEFAULT_RIR_TEST, DEFAULT_TEST_DIR
from src.training import CONFIG_NAMES, MODEL_CLASSES

BUILTIN_DEFAULTS: dict[str, Any] = {
    "checkpoint": None,
    "config": None,
    "model": None,
    "test_dir": DEFAULT_TEST_DIR,
    "noise_dir": DEFAULT_NOISE_TEST,
    "rir_dirs": DEFAULT_RIR_TEST,
    "snr": [0, 5, 10, 15],
    "noise_proba": 0.85,
    "rir_proba": 0.85,
    "dataset_epoch": 99,
    "seed": 1984,
    "num_workers": 0,
    "max_samples": None,
    "benchmark": True,
    "benchmark_device": None,
    "benchmark_chunk_size": None,
    "benchmark_max_samples": None,
    "device": "cuda",
    "normalize_output": True,
    "nisqa_config": "NISQA_s/config/nisqa_s.yaml",
    "no_nisqa": False,
    "output_csv": None,
}


def _normalize_list_field(value: Any, *, as_int: bool = False) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        items = [x.strip() for x in value.split(",") if x.strip()]
    elif isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], str) and "," in value[0]:
            items = [x.strip() for x in value[0].split(",") if x.strip()]
        else:
            items = list(value)
    else:
        raise ValueError(f"Expected list or comma-separated string, got {type(value)}")
    if as_int:
        return [int(x) for x in items]
    return [str(x) for x in items]


def normalize_test_config(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = set(raw) - set(BUILTIN_DEFAULTS)
    if unknown:
        known = ", ".join(sorted(BUILTIN_DEFAULTS))
        raise ValueError(
            f"Unknown keys in test config: {', '.join(sorted(unknown))}. Known keys: {known}"
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
    if not cfg["checkpoint"]:
        raise ValueError("checkpoint path is required in test config or --checkpoint")

    cfg["rir_dirs"] = _normalize_list_field(cfg["rir_dirs"])
    cfg["snr"] = _normalize_list_field(cfg["snr"], as_int=True)
    cfg["no_nisqa"] = bool(cfg["no_nisqa"])
    cfg["normalize_output"] = bool(cfg["normalize_output"])
    cfg["benchmark"] = bool(cfg["benchmark"])
    cfg["max_samples"] = int(cfg["max_samples"]) if cfg.get("max_samples") else None
    cfg["benchmark_max_samples"] = (
        int(cfg["benchmark_max_samples"]) if cfg.get("benchmark_max_samples") else None
    )
    if cfg.get("benchmark_chunk_size") is not None:
        cfg["benchmark_chunk_size"] = int(cfg["benchmark_chunk_size"])
    return cfg


def load_test_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Test config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Test config must be a YAML mapping, got {type(raw)}")
    return normalize_test_config(raw)


def dump_test_config(cfg: dict[str, Any]) -> str:
    export = {k: v for k, v in cfg.items() if k not in ("test_config", "dump_config")}
    return yaml.safe_dump(export, sort_keys=False, allow_unicode=True)


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Evaluate FSPEN checkpoint on simulated test set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python test_fspen.py --test-config configs/test/fspen_48khz_overlap.yaml\n"
            "  python test_fspen.py --checkpoint checkpoints/fspen_chkp/model.pt "
            "--config TrainConfig_48kHz_overlap\n"
        ),
    )
    p.add_argument(
        "-c", "--test-config", dest="test_config", default=None, metavar="YAML"
    )
    p.add_argument("--dump-config", action="store_true")

    p.add_argument("--checkpoint", default=None)
    p.add_argument("--config", choices=CONFIG_NAMES, default=None)
    p.add_argument("--model", choices=sorted(MODEL_CLASSES), default=None)

    p.add_argument("--test-dir", default=None)
    p.add_argument("--noise-dir", default=None)
    p.add_argument(
        "--rir-dirs",
        nargs="+",
        default=None,
        metavar="DIR",
        help="RIR test dirs (space- or comma-separated)",
    )
    p.add_argument(
        "--snr",
        nargs="+",
        default=None,
        metavar="DB",
        help="SNR values in dB (space- or comma-separated)",
    )
    p.add_argument("--noise-proba", type=float, default=None)
    p.add_argument("--rir-proba", type=float, default=None)
    p.add_argument("--dataset-epoch", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--benchmark", action="store_true", default=None)
    p.add_argument("--no-benchmark", action="store_true", default=None)
    p.add_argument(
        "--benchmark-device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for RTF benchmark (default: same as --device)",
    )
    p.add_argument("--benchmark-chunk-size", type=int, default=None)
    p.add_argument("--benchmark-max-samples", type=int, default=None)
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--normalize-output", action="store_true", default=None)
    p.add_argument("--no-normalize-output", action="store_true", default=None)
    p.add_argument("--nisqa-config", default=None)
    p.add_argument("--no-nisqa", action="store_true", default=None)
    p.add_argument("--output-csv", default=None)
    return p


def parse_test_args(argv: Optional[list[str]] = None) -> dict[str, Any]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--test-config", dest="test_config")
    pre.add_argument("--dump-config", action="store_true")
    pre_args, remaining = pre.parse_known_args(argv)

    yaml_defaults = (
        load_test_yaml(pre_args.test_config)
        if pre_args.test_config
        else dict(BUILTIN_DEFAULTS)
    )

    parser = build_argument_parser()
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(remaining)

    cfg = dict(yaml_defaults)
    for key, value in vars(args).items():
        if key in ("test_config", "dump_config"):
            continue
        if value is not None:
            cfg[key] = value

    if args.no_nisqa is True:
        cfg["no_nisqa"] = True
    if args.no_benchmark is True:
        cfg["benchmark"] = False
    elif args.benchmark is True:
        cfg["benchmark"] = True
    if args.no_normalize_output is True:
        cfg["normalize_output"] = False
    elif args.normalize_output is True:
        cfg["normalize_output"] = True

    if cfg.get("rir_dirs") is not None:
        cfg["rir_dirs"] = _normalize_list_field(cfg["rir_dirs"])
    if cfg.get("snr") is not None:
        cfg["snr"] = _normalize_list_field(cfg["snr"], as_int=True)

    cfg["test_config"] = pre_args.test_config
    cfg["dump_config"] = pre_args.dump_config or args.dump_config
    cfg["max_samples"] = int(cfg["max_samples"]) if cfg.get("max_samples") else None
    cfg["benchmark_max_samples"] = (
        int(cfg["benchmark_max_samples"]) if cfg.get("benchmark_max_samples") else None
    )
    if cfg.get("benchmark_chunk_size") is not None:
        cfg["benchmark_chunk_size"] = int(cfg["benchmark_chunk_size"])
    return cfg
