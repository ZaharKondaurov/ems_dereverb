#!/usr/bin/env python3
"""
Evaluate FSPEN checkpoint on simulated test set (notebooks/test_fspen_*.ipynb).

Examples:
  python test_fspen.py -c configs/test/fspen_48khz_overlap.yaml
  python test_fspen.py --checkpoint checkpoints/fspen_chkp/model.pt \\
      --config TrainConfig_48kHz_overlap --max-samples 100
"""

from __future__ import annotations

import sys

import torch

from src.fspen_eval import (
    TestMetrics,
    benchmark_inference,
    build_test_dataloader,
    evaluate_model,
    load_model_from_checkpoint,
    print_metrics,
    save_metrics_csv,
    summarize_metrics,
)
from src.test_cli import dump_test_config, parse_test_args
from src.training import EVAL_FUNCTIONS, default_eval_fn_for_config


def main() -> None:
    cfg = parse_test_args()

    if cfg["dump_config"]:
        print(dump_test_config(cfg), end="")
        return

    if not cfg["checkpoint"]:
        print("Error: --checkpoint is required.", file=sys.stderr)
        sys.exit(2)

    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.", file=sys.stderr)
        device = torch.device("cpu")

    model, configs, config_name, model_name = load_model_from_checkpoint(
        cfg["checkpoint"],
        device=device,
        config_name=cfg["config"],
        model_name=cfg["model"],
    )
    eval_fn_name = default_eval_fn_for_config(config_name)
    eval_fn = EVAL_FUNCTIONS[eval_fn_name]

    dataset, loader = build_test_dataloader(
        test_dir=cfg["test_dir"],
        noise_dir=cfg["noise_dir"],
        rir_dirs=cfg["rir_dirs"],
        sample_rate=configs.sample_rate,
        snr=cfg["snr"],
        noise_proba=cfg["noise_proba"],
        rir_proba=cfg["rir_proba"],
        dataset_epoch=cfg["dataset_epoch"],
        seed=cfg["seed"],
        num_workers=cfg["num_workers"],
    )

    metrics_ctx = TestMetrics(
        sample_rate=configs.sample_rate,
        n_fft=configs.n_fft,
        hop_length=configs.hop_length,
        device=device,
        nisqa_config=cfg["nisqa_config"],
        enable_nisqa=not cfg["no_nisqa"],
    )

    if cfg["test_config"]:
        print(f"Loaded test config: {cfg['test_config']}")

    print(
        f"Evaluating {model_name} ({config_name})\n"
        f"  checkpoint: {cfg['checkpoint']}\n"
        f"  eval_fn: {eval_fn_name}\n"
        f"  device: {device}\n"
        f"  test_dir: {cfg['test_dir']}\n"
        f"  dataset_epoch: {cfg['dataset_epoch']} seed: {cfg['seed']}\n"
        f"  normalize_output: {cfg['normalize_output']}"
    )

    raw = evaluate_model(
        model,
        configs,
        loader,
        metrics_ctx,
        eval_fn,
        device=device,
        normalize_output=cfg["normalize_output"],
        max_samples=cfg["max_samples"],
    )
    summary = summarize_metrics(raw)

    if cfg["benchmark"]:
        benchmark_device_name = cfg["benchmark_device"] or cfg["device"]
        benchmark_device = torch.device(benchmark_device_name)
        if benchmark_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA not available for benchmark, using CPU.", file=sys.stderr)
            benchmark_device = torch.device("cpu")
        benchmark_model = (
            model if benchmark_device == device else model.to(benchmark_device)
        )
        print(
            f"\nBenchmarking RTF on {benchmark_device} "
            f"(chunk_size={cfg['benchmark_chunk_size'] or configs.n_fft * 5})"
        )
        summary.update(
            benchmark_inference(
                benchmark_model,
                configs,
                dataset,
                eval_fn,
                device=benchmark_device,
                chunk_window_size=cfg["benchmark_chunk_size"],
                max_samples=cfg["benchmark_max_samples"],
            )
        )
        if benchmark_device != device:
            model.to(device)

    print_metrics(summary)

    if cfg["output_csv"]:
        save_metrics_csv(summary, cfg["output_csv"])


if __name__ == "__main__":
    main()
