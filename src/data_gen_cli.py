"""CLI + YAML configuration for generate_data.py."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Optional

import yaml

from src.data_gen import RIR_PRESETS

BUILTIN_DEFAULTS: dict[str, Any] = {
    "tasks": ["split_demand", "generate_rirs"],
    "seed": 1984,
    "sample_rate": 48_000,
    "stimulus": "data/DS_10283_2791/clean_trainset_28spk_wav/p226_001.wav",
    "rir_presets": list(RIR_PRESETS),
    "train_fraction": 0.7,
    "val_fraction": 0.5,
    "demand_src_dir": "data/demand",
    "demand_train_dir": "data/demand_train",
    "demand_val_test_dir": "data/demand_val_test",
    "demand_val_dir": "data/demand_val",
    "demand_test_dir": "data/demand_test",
}


def _normalize_list_field(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [x.strip() for x in value.split(",") if x.strip()]
    if isinstance(value, (list, tuple)):
        if len(value) == 1 and isinstance(value[0], str) and "," in value[0]:
            return [x.strip() for x in value[0].split(",") if x.strip()]
        return [str(x) for x in value]
    raise ValueError(f"Expected list or comma-separated string, got {type(value)}")


def normalize_data_config(raw: dict[str, Any]) -> dict[str, Any]:
    unknown = set(raw) - set(BUILTIN_DEFAULTS)
    if unknown:
        known = ", ".join(sorted(BUILTIN_DEFAULTS))
        raise ValueError(
            f"Unknown keys in data config: {', '.join(sorted(unknown))}. Known keys: {known}"
        )

    cfg = dict(BUILTIN_DEFAULTS)
    cfg.update(raw)
    cfg["tasks"] = _normalize_list_field(cfg["tasks"])
    cfg["rir_presets"] = _normalize_list_field(cfg["rir_presets"])

    for preset in cfg["rir_presets"]:
        if preset not in RIR_PRESETS:
            raise ValueError(
                f"Unknown RIR preset {preset!r}. Choose from: {sorted(RIR_PRESETS)}"
            )

    for task in cfg["tasks"]:
        if task not in ("split_demand", "generate_rirs"):
            raise ValueError(
                f"Unknown task {task!r}. Choose from: split_demand, generate_rirs"
            )

    return cfg


def load_data_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Data config not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Data config must be a YAML mapping, got {type(raw)}")
    return normalize_data_config(raw)


def dump_data_config(cfg: dict[str, Any]) -> str:
    export = {k: v for k, v in cfg.items() if k not in ("data_config", "dump_config")}
    return yaml.safe_dump(export, sort_keys=False, allow_unicode=True)


def build_argument_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Generate training data (notebooks/sandbox.ipynb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python generate_data.py -c configs/data/default.yaml\n"
            "  python generate_data.py --tasks split_demand\n"
            "  python generate_data.py --tasks generate_rirs --rir-presets small medium\n"
        ),
    )
    p.add_argument(
        "-c", "--data-config", dest="data_config", default=None, metavar="YAML"
    )
    p.add_argument("--dump-config", action="store_true")

    p.add_argument(
        "--tasks",
        nargs="+",
        choices=["split_demand", "generate_rirs"],
        default=None,
        metavar="TASK",
        help="Tasks to run (space-separated)",
    )
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sample-rate", type=int, default=None)
    p.add_argument("--stimulus", default=None, help="Stimulus wav for RIR simulation")
    p.add_argument(
        "--rir-presets",
        nargs="+",
        choices=sorted(RIR_PRESETS),
        default=None,
        metavar="PRESET",
        help="RIR room presets to generate",
    )
    p.add_argument("--train-fraction", type=float, default=None)
    p.add_argument("--val-fraction", type=float, default=None)

    p.add_argument("--demand-src-dir", default=None)
    p.add_argument("--demand-train-dir", default=None)
    p.add_argument("--demand-val-test-dir", default=None)
    p.add_argument("--demand-val-dir", default=None)
    p.add_argument("--demand-test-dir", default=None)
    return p


def parse_data_args(argv: Optional[list[str]] = None) -> dict[str, Any]:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("-c", "--data-config", dest="data_config")
    pre.add_argument("--dump-config", action="store_true")
    pre_args, remaining = pre.parse_known_args(argv)

    yaml_defaults = (
        load_data_yaml(pre_args.data_config)
        if pre_args.data_config
        else dict(BUILTIN_DEFAULTS)
    )

    parser = build_argument_parser()
    parser.set_defaults(**yaml_defaults)
    args = parser.parse_args(remaining)

    cfg = dict(yaml_defaults)
    for key, value in vars(args).items():
        if key in ("data_config", "dump_config"):
            continue
        if value is not None:
            cfg[key] = value

    if cfg.get("tasks") is not None:
        cfg["tasks"] = _normalize_list_field(cfg["tasks"])
    if cfg.get("rir_presets") is not None:
        cfg["rir_presets"] = _normalize_list_field(cfg["rir_presets"])

    data_config_path = pre_args.data_config
    dump_config = pre_args.dump_config or args.dump_config
    cfg = normalize_data_config(cfg)
    cfg["data_config"] = data_config_path
    cfg["dump_config"] = dump_config
    return cfg
