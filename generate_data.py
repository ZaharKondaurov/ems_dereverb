#!/usr/bin/env python3
"""
Generate training data (notebooks/sandbox.ipynb).

Tasks:
  split_demand   - split DEMAND noise into train / val / test
  generate_rirs  - simulate RIRs with pyroomacoustics and split into train / val / test

Examples:
  python generate_data.py -c configs/data/default.yaml
  python generate_data.py --tasks split_demand
  python generate_data.py --tasks generate_rirs --rir-presets small --seed 1984
"""

from __future__ import annotations

import sys
from pathlib import Path

from src.data_gen import (
    RIR_PRESETS,
    generate_and_split_rir_preset,
    split_demand_noise,
)
from src.data_gen_cli import dump_data_config, parse_data_args


def main() -> None:
    cfg = parse_data_args()

    if cfg["dump_config"]:
        print(dump_data_config(cfg), end="")
        return

    if cfg["data_config"]:
        print(f"Loaded data config: {cfg['data_config']}")

    stimulus = Path(cfg["stimulus"])
    if "generate_rirs" in cfg["tasks"] and not stimulus.is_file():
        print(f"Error: stimulus not found: {stimulus}", file=sys.stderr)
        sys.exit(2)

    if "split_demand" in cfg["tasks"]:
        demand_src = Path(cfg["demand_src_dir"])
        if not demand_src.is_dir():
            print(f"Error: DEMAND source dir not found: {demand_src}", file=sys.stderr)
            sys.exit(2)
        split_demand_noise(
            src_dir=cfg["demand_src_dir"],
            train_dir=cfg["demand_train_dir"],
            val_test_dir=cfg["demand_val_test_dir"],
            val_dir=cfg["demand_val_dir"],
            test_dir=cfg["demand_test_dir"],
            train_fraction=cfg["train_fraction"],
            val_fraction=cfg["val_fraction"],
            seed=cfg["seed"],
        )

    if "generate_rirs" in cfg["tasks"]:
        print(
            f"Generating RIR presets: {', '.join(cfg['rir_presets'])}\n"
            f"  stimulus: {cfg['stimulus']}\n"
            f"  sample_rate: {cfg['sample_rate']} seed: {cfg['seed']}"
        )
        for preset_name in cfg["rir_presets"]:
            preset = RIR_PRESETS[preset_name]
            print(
                f"\n[{preset_name}] {preset['output_dir']} "
                f"count={preset['count']} rt60={preset['rt60']}"
            )
            generate_and_split_rir_preset(
                preset_name,
                stimulus_path=cfg["stimulus"],
                sample_rate=cfg["sample_rate"],
                seed=cfg["seed"],
                train_fraction=cfg["train_fraction"],
                val_fraction=cfg["val_fraction"],
            )


if __name__ == "__main__":
    main()
