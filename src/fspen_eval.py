"""Batch evaluation of FSPEN checkpoints (notebooks/test_fspen_*.ipynb)."""

from __future__ import annotations

import os
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader
from torchaudio.transforms import Resample
from torchmetrics.audio import (
    DeepNoiseSuppressionMeanOpinionScore,
    ScaleInvariantSignalDistortionRatio,
    SpeechReverberationModulationEnergyRatio,
)
from torchmetrics.audio.pesq import PerceptualEvaluationSpeechQuality
from torch_stoi import NegSTOILoss
from tqdm import tqdm

from src.dataset import SignalDataset
from src.training import (
    DEFAULT_MODEL_FOR_CONFIG,
    EVAL_FUNCTIONS,
    build_model,
    default_eval_fn_for_config,
    load_config_class,
    make_worker_init_fn,
    rir_dict_from_paths,
)
from src.utils import vorbis_window

DEFAULT_TEST_DIR = os.path.join("data", "DS_10283_2791", "clean_testset_wav")
DEFAULT_NOISE_TEST = os.path.join("data", "demand_test")
DEFAULT_RIR_TEST = [
    os.path.join("data", "rirs48_small_3_test"),
    os.path.join("data", "rirs48_medium_3_test"),
    os.path.join("data", "rirs48_large_3_test"),
    os.path.join("data", "rirs48_super_large_3_test"),
]
METRIC_SR = 16_000


def make_waveform_collate_fn():
    def collate_fn(batch):
        if not batch:
            empty = torch.zeros(0)
            return empty, empty
        input_signal, target_signal, _, _ = zip(*batch)
        max_len = max(s.shape[-1] for s in input_signal)
        padded_input = torch.zeros(len(input_signal), max_len)
        padded_target = torch.zeros(len(target_signal), max_len)
        for i, s in enumerate(input_signal):
            padded_input[i, : s.shape[-1]] = s
            padded_target[i, : target_signal[i].shape[-1]] = target_signal[i]
        return padded_input.reshape(-1, max_len), padded_target.reshape(-1, max_len)

    return collate_fn


def build_test_dataloader(
    *,
    test_dir: str,
    noise_dir: str,
    rir_dirs: list[str],
    sample_rate: int,
    snr: list[int],
    noise_proba: float,
    rir_proba: float,
    dataset_epoch: int,
    seed: int,
    num_workers: int,
) -> tuple[SignalDataset, DataLoader]:
    dataset = SignalDataset(
        test_dir,
        sr=sample_rate,
        noise_dir=noise_dir,
        rir_dir=rir_dict_from_paths(rir_dirs),
        snr=snr,
        rir_proba=rir_proba,
        noise_proba=noise_proba,
        rir_target=False,
        return_noise=False,
        return_rir=False,
        verbose=False,
        base_seed=seed,
    )
    dataset.set_epoch(dataset_epoch)

    worker_init_fn = make_worker_init_fn(seed) if num_workers > 0 else None
    gen = torch.Generator()
    gen.manual_seed(seed)
    loader_kwargs: dict[str, Any] = dict(
        batch_size=1,
        shuffle=False,
        drop_last=False,
        collate_fn=make_waveform_collate_fn(),
        num_workers=num_workers,
        generator=gen,
    )
    if worker_init_fn is not None:
        loader_kwargs["worker_init_fn"] = worker_init_fn

    return dataset, DataLoader(dataset, **loader_kwargs)


def load_model_from_checkpoint(
    checkpoint: str,
    *,
    device: torch.device,
    config_name: Optional[str] = None,
    model_name: Optional[str] = None,
) -> tuple[torch.nn.Module, object, str, str]:
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    resolved_config = config_name or state.get("config_name")
    if not resolved_config:
        raise ValueError(
            "Config name is required (--config or config in YAML). "
            "Checkpoint has no config_name field."
        )

    config_cls = load_config_class(resolved_config)
    configs = config_cls()
    resolved_model = model_name or DEFAULT_MODEL_FOR_CONFIG.get(
        resolved_config, "FullSubPathExtension"
    )
    model = build_model(resolved_model, configs)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, configs, resolved_config, resolved_model


class TestMetrics:
    """Objective metrics from test_fspen_48khz_*.ipynb notebooks."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        device: torch.device,
        nisqa_config: str,
        enable_nisqa: bool,
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.device = device
        self.enable_nisqa = enable_nisqa
        self.nisqa = None
        self.h0_nisqa = None
        self.c0_nisqa = None
        self.nisqa_args: Optional[dict] = None
        self.process_fn = None

        self.stoi = NegSTOILoss(METRIC_SR, use_vad=False, do_resample=False).to(device)
        self.pesq = PerceptualEvaluationSpeechQuality(fs=METRIC_SR, mode="wb").to(
            device
        )
        self.srmr = SpeechReverberationModulationEnergyRatio(fs=METRIC_SR, norm=False)
        self.sisdr = ScaleInvariantSignalDistortionRatio().to(device)
        self.dnsmos = DeepNoiseSuppressionMeanOpinionScore(
            METRIC_SR, False, device=device
        )
        self.resampler = Resample(sample_rate, METRIC_SR)

        if enable_nisqa:
            if not nisqa_config or not os.path.isfile(nisqa_config):
                warnings.warn("NISQA config not found; disabling NISQA metrics.")
                self.enable_nisqa = False
            else:
                try:
                    from NISQA_s.src.core.model_torch import model_init
                    from NISQA_s.src.utils.process_utils import process as nisqa_process
                except ImportError as exc:
                    warnings.warn(
                        f"NISQA not available ({exc}); disabling NISQA metrics."
                    )
                    self.enable_nisqa = False
                else:
                    with open(nisqa_config, "r", encoding="utf-8") as stream:
                        nisqa_args = yaml.safe_load(stream)
                    nisqa_args["ms_n_fft"] = n_fft
                    nisqa_args["hop_length"] = hop_length
                    nisqa_args["ms_win_length"] = n_fft
                    if (
                        isinstance(nisqa_args.get("ckp"), str)
                        and len(nisqa_args["ckp"]) > 3
                    ):
                        nisqa_args["ckp"] = nisqa_args["ckp"][3:]
                    nisqa_args["inf_device"] = device
                    self.nisqa, self.h0_nisqa, self.c0_nisqa = model_init(nisqa_args)
                    self.nisqa_args = nisqa_args
                    self.process_fn = nisqa_process

    def nisqa_score(self, waveform: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_nisqa or self.process_fn is None:
            return None
        score, _, _ = self.process_fn(
            waveform.detach().cpu(),
            self.sample_rate,
            self.nisqa,
            self.h0_nisqa,
            self.c0_nisqa,
            self.nisqa_args,
        )
        return score[0].detach().cpu()


@torch.no_grad()
def evaluate_model(
    model: torch.nn.Module,
    configs,
    loader: DataLoader,
    metrics: TestMetrics,
    eval_fn: Callable,
    *,
    device: torch.device,
    normalize_output: bool = True,
    max_samples: Optional[int] = None,
) -> dict[str, list[torch.Tensor]]:
    model.eval()
    n_fft = configs.n_fft
    hop_length = configs.hop_length
    window = vorbis_window(n_fft).to(device)

    results: dict[str, list[torch.Tensor]] = {
        "nisqa": [],
        "stoi": [],
        "sisdr": [],
        "srmr": [],
        "pesq": [],
        "dnsmos": [],
    }
    n_evaluated = 0

    for ind, (signal, target) in enumerate(tqdm(loader, desc="Evaluate")):
        if max_samples is not None and n_evaluated >= max_samples:
            break

        signal = signal.to(device)
        target = target.to(device)

        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            normalized=True,
            center=True,
        )
        output, _ = eval_fn(model, spec, configs)
        output = torch.istft(
            output,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=False,
            normalized=True,
            center=True,
        )

        if normalize_output:
            out_peak = output.abs().max()
            in_peak = signal.abs().max()
            if out_peak > 0 and in_peak > 0:
                output = output / (out_peak / in_peak)

        min_l = min(output.shape[-1], target.shape[-1])
        output = output[:, :min_l]
        target = target[:, :min_l]

        nisqa = metrics.nisqa_score(output)
        if nisqa is not None:
            results["nisqa"].append(nisqa)

        out_16k = metrics.resampler(output.cpu()).to(device)
        tgt_16k = metrics.resampler(target.cpu()).to(device)
        min_l_16k = min(out_16k.shape[-1], tgt_16k.shape[-1])
        out_16k = out_16k[..., :min_l_16k]
        tgt_16k = tgt_16k[..., :min_l_16k]

        try:
            pesq_score = metrics.pesq(out_16k, tgt_16k)
        except Exception:
            continue

        results["stoi"].append(-metrics.stoi(out_16k, tgt_16k).detach().cpu())
        results["srmr"].append(metrics.srmr(out_16k.detach().cpu()))
        results["sisdr"].append(-metrics.sisdr(out_16k, tgt_16k).detach().cpu())
        results["pesq"].append(pesq_score.detach().cpu())
        results["dnsmos"].append(metrics.dnsmos(out_16k.detach()).cpu())
        n_evaluated += 1

    if n_evaluated == 0:
        raise RuntimeError(
            "No samples were evaluated (check dataset paths and PESQ failures)."
        )

    return results


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


@torch.no_grad()
def benchmark_inference(
    model: torch.nn.Module,
    configs,
    dataset: SignalDataset,
    eval_fn: Callable,
    *,
    device: torch.device,
    chunk_window_size: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> dict[str, float]:
    """RTF benchmark from notebooks/test_fspen_*.ipynb (full utterance + streaming chunks)."""
    model.eval()
    n_fft = configs.n_fft
    hop_length = configs.hop_length
    sample_rate = configs.sample_rate
    chunk_window_size = chunk_window_size or (n_fft * 5)
    window = vorbis_window(n_fft).to(device)

    rtf_full: list[float] = []
    rtf_chunk: list[float] = []
    n_processed = 0

    for idx in tqdm(range(len(dataset)), desc="Benchmark chunk"):
        if max_samples is not None and n_processed >= max_samples:
            break

        signal, _, _, _ = dataset[idx]
        signal = signal.reshape(1, -1).to(device)

        _sync_device(device)
        start = time.perf_counter()
        spec = torch.stft(
            signal,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            normalized=True,
            center=True,
        )
        output, _ = eval_fn(model, spec, configs)
        output = torch.istft(
            output,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=False,
            normalized=True,
            center=True,
        )
        _sync_device(device)
        elapsed = time.perf_counter() - start
        rtf_full.append((signal.shape[-1] / sample_rate) / elapsed)

        h0 = None
        for offset in range(0, signal.shape[-1], chunk_window_size):
            chunk = signal[..., offset : offset + chunk_window_size]
            if chunk.shape[-1] < chunk_window_size:
                continue

            _sync_device(device)
            start = time.perf_counter()
            spec = torch.stft(
                chunk,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=True,
                normalized=True,
                center=True,
            )
            output, h0 = eval_fn(model, spec, configs, h0)
            output = torch.istft(
                output,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=n_fft,
                window=window,
                return_complex=False,
                normalized=True,
                center=True,
            )
            _sync_device(device)
            elapsed = time.perf_counter() - start
            rtf_chunk.append((chunk.shape[-1] / sample_rate) / elapsed)

        n_processed += 1

    if not rtf_full:
        raise RuntimeError("No samples were benchmarked (empty dataset).")

    mean_full = sum(rtf_full) / len(rtf_full)
    mean_chunk = sum(rtf_chunk) / len(rtf_chunk) if rtf_chunk else 0.0
    return {
        "rtf_full_speedup": mean_full,
        "rtf_chunk_speedup": mean_chunk,
        "rtf_full": 1.0 / mean_full,
        "rtf_chunk": (1.0 / mean_chunk) if mean_chunk > 0 else float("inf"),
        "n_benchmark_samples": n_processed,
        "benchmark_chunk_size": chunk_window_size,
    }


def summarize_metrics(raw: dict[str, list[torch.Tensor]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"n_samples": len(raw.get("pesq", []))}
    if raw.get("nisqa"):
        summary["nisqa"] = torch.vstack(raw["nisqa"]).mean(dim=0).tolist()
    for key in ("stoi", "sisdr", "srmr", "pesq", "dnsmos"):
        if raw.get(key):
            summary[key] = torch.vstack(raw[key]).mean(dim=0).tolist()
    return summary


def print_metrics(summary: dict[str, Any]) -> None:
    print(f"Samples evaluated: {summary['n_samples']}")
    if "nisqa" in summary:
        print(f"NISQA:  {summary['nisqa']}")
    if "pesq" in summary:
        print(f"PESQ:   {summary['pesq']}")
    if "srmr" in summary:
        print(f"SRMR:   {summary['srmr']}")
    if "stoi" in summary:
        print(f"STOI:   {summary['stoi']}")
    if "sisdr" in summary:
        print(f"SI-SDR: {summary['sisdr']}")
    if "dnsmos" in summary:
        print(f"DNSMOS: {summary['dnsmos']}")
    if "rtf_full" in summary:
        print(
            f"RTF full:  {summary['rtf_full']:.4f} "
            f"(speedup {summary.get('rtf_full_speedup', 0):.2f}x)"
        )
    if "rtf_chunk" in summary:
        print(
            f"RTF chunk: {summary['rtf_chunk']:.4f} "
            f"(speedup {summary.get('rtf_chunk_speedup', 0):.2f}x, "
            f"chunk={summary.get('benchmark_chunk_size')})"
        )


def save_metrics_csv(summary: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row: dict[str, Any] = {"n_samples": summary["n_samples"]}
    for key in (
        "rtf_full",
        "rtf_chunk",
        "rtf_full_speedup",
        "rtf_chunk_speedup",
        "benchmark_chunk_size",
        "n_benchmark_samples",
    ):
        if key in summary:
            row[key] = summary[key]
    for key in ("nisqa", "stoi", "sisdr", "srmr", "pesq", "dnsmos"):
        if key not in summary:
            continue
        values = summary[key]
        if isinstance(values, list) and len(values) == 1:
            row[key] = values[0]
        else:
            for i, value in enumerate(values):
                row[f"{key}_{i}"] = value
    pd.DataFrame([row]).to_csv(path, index=False)
    print(f"Saved metrics: {path}")
