"""FSPEN training loop extracted from notebooks/train_fspen_*.ipynb."""

from __future__ import annotations

import os
import random
import warnings
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402
from torch_stoi import NegSTOILoss  # noqa: E402
from tqdm import tqdm  # noqa: E402

from models.fspen import FullSubPathExtension, FullSubPathExtension_ext
from src.dataset import SignalDataset
from src.loss import loss_MR
from src.streaming import config_uses_ext_eval
from src.utils import model_eval, model_eval_old, vorbis_window

CONFIG_NAMES = (
    "TrainConfig",
    "TrainConfig_baseline",
    "TrainConfig_48khz",
    "TrainConfig_48kHz_overlap",
    "TrainConfig_48kHz_enc_ext",
    "TrainConfig_48kHz_enc_ext_lay_1_overlap",
)

MODEL_CLASSES: dict[str, type[torch.nn.Module]] = {
    "FullSubPathExtension": FullSubPathExtension,
    "FullSubPathExtension_ext": FullSubPathExtension_ext,
}

DEFAULT_MODEL_FOR_CONFIG: dict[str, str] = {
    "TrainConfig": "FullSubPathExtension",
    "TrainConfig_baseline": "FullSubPathExtension",
    "TrainConfig_48khz": "FullSubPathExtension",
    "TrainConfig_48kHz_overlap": "FullSubPathExtension",
    "TrainConfig_48kHz_enc_ext": "FullSubPathExtension_ext",
    "TrainConfig_48kHz_enc_ext_lay_1_overlap": "FullSubPathExtension_ext",
}

EVAL_FUNCTIONS: dict[str, Callable] = {
    "model_eval_old": model_eval_old,
    "model_eval": model_eval,
}


def default_eval_fn_for_config(config_name: str) -> str:
    return "model_eval" if config_uses_ext_eval(config_name) else "model_eval_old"


def load_config_class(config_name: str):
    import src.fspen_configs as fspen_configs

    cls = getattr(fspen_configs, config_name, None)
    if cls is None:
        known = ", ".join(CONFIG_NAMES)
        raise ValueError(f"Unknown config {config_name!r}. Choose one of: {known}")
    return cls


def build_model(model_name: str, configs) -> torch.nn.Module:
    if model_name not in MODEL_CLASSES:
        known = ", ".join(MODEL_CLASSES)
        raise ValueError(f"Unknown model {model_name!r}. Choose one of: {known}")
    return MODEL_CLASSES[model_name](configs=configs)


def rir_dict_from_paths(paths: list[str]) -> dict[int, str]:
    """Map RIR directories to curriculum steps (all active from epoch 1)."""
    return {i + 1: p for i, p in enumerate(paths)}


def make_collate_fn(n_fft: int, hop_length: int):
    window = vorbis_window(n_fft)

    def collate_fn(batch):
        input_signal, target_signal, _, _ = zip(*batch)
        padded_input = torch.stack(input_signal)
        padded_target = torch.stack(target_signal)

        if torch.isnan(padded_input).any():
            raise ValueError("Waveform batch contains NaNs")

        padded_input = padded_input.reshape(-1, padded_input.shape[-1])
        input_spec = torch.stft(
            padded_input,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            normalized=True,
            center=True,
        )
        padded_target = padded_target.reshape(-1, padded_target.shape[-1])
        return input_spec, padded_target, padded_input

    return collate_fn


def make_worker_init_fn(seed: int) -> Callable[[int], None]:
    def worker_init_fn(worker_id: int) -> None:
        np.random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)
        random.seed(seed + worker_id)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed + worker_id)
            torch.cuda.manual_seed_all(seed + worker_id)
            torch.backends.cudnn.deterministic = True

    return worker_init_fn


def _has_nan_weights(model: torch.nn.Module) -> bool:
    for name, param in model.named_parameters():
        if param.requires_grad and torch.isnan(param).any():
            print(f"NaN in weights: {name}")
            return True
    return False


def _has_nan_grads(model: torch.nn.Module) -> bool:
    for name, param in model.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN in gradients: {name}")
            return True
    return False


def _filter_nan_specs(input_spec, gt_signal, input_signal):
    mask = torch.isnan(input_spec).any(dim=(1, 2))
    return input_spec[~mask], gt_signal[~mask], input_signal[~mask]


def _waveform_from_output(output, n_fft: int, hop_length: int, device: torch.device):
    window = vorbis_window(n_fft).to(device)
    return torch.istft(
        output,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=n_fft,
        window=window,
        return_complex=False,
        normalized=True,
        center=True,
    )


class MetricsContext:
    """Optional NISQA + STOI helpers for train/val loops."""

    def __init__(
        self,
        *,
        sample_rate: int,
        n_fft: int,
        hop_length: int,
        device: torch.device,
        nisqa_config: Optional[str],
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

        self.stoi = NegSTOILoss(sample_rate, use_vad=False, do_resample=False).to(
            device
        )

        if enable_nisqa:
            if not nisqa_config or not os.path.isfile(nisqa_config):
                warnings.warn("NISQA config not found; disabling NISQA metrics.")
                self.enable_nisqa = False
                return
            try:
                from NISQA_s.src.core.model_torch import model_init
                from NISQA_s.src.utils.process_utils import process as nisqa_process
            except ImportError as exc:
                warnings.warn(f"NISQA not available ({exc}); disabling NISQA metrics.")
                self.enable_nisqa = False
                return

            with open(nisqa_config, "r", encoding="utf-8") as stream:
                nisqa_args = yaml.safe_load(stream)
            nisqa_args["ms_n_fft"] = n_fft
            nisqa_args["hop_length"] = hop_length
            nisqa_args["ms_win_length"] = n_fft
            if isinstance(nisqa_args.get("ckp"), str) and len(nisqa_args["ckp"]) > 3:
                nisqa_args["ckp"] = nisqa_args["ckp"][3:]
            nisqa_args["inf_device"] = device
            self.nisqa, self.h0_nisqa, self.c0_nisqa = model_init(nisqa_args)
            self.nisqa_args = nisqa_args
            self.process_fn = nisqa_process

    def stoi_score(
        self, out_wave: torch.Tensor, gt_signal: torch.Tensor
    ) -> torch.Tensor:
        min_l = min(out_wave.shape[-1], gt_signal.shape[-1])
        return -self.stoi(out_wave[..., :min_l], gt_signal[..., :min_l])

    def nisqa_score(self, out_wave: torch.Tensor) -> Optional[torch.Tensor]:
        if not self.enable_nisqa or self.process_fn is None:
            return None
        nisqa_score, _, _ = self.process_fn(
            out_wave.detach(),
            self.sample_rate,
            self.nisqa,
            self.h0_nisqa,
            self.c0_nisqa,
            self.nisqa_args,
        )
        return nisqa_score[0].detach().cpu()


def train_epoch(
    model: torch.nn.Module,
    configs,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    *,
    eval_fn: Callable,
    metrics: MetricsContext,
    device: torch.device,
    epoch: int = 0,
    n_fft: int,
    hop_length: int,
    compute_nisqa_every: int = 5,
) -> tuple[
    float, float, float, Optional[torch.Tensor], Optional[float], float, Optional[float]
]:
    model.train()
    total_loss = 0.0
    total_loss_mr = 0.0
    total_nisqa = torch.zeros(5)
    nisqa_batches = 0
    stoi_scores: list[torch.Tensor] = []
    n_batches = 0
    n_loader_batches = len(train_loader)

    for input_spec, gt_signal, _input_signal in tqdm(train_loader, desc="Train"):
        input_spec, gt_signal, _ = _filter_nan_specs(
            input_spec, gt_signal, _input_signal
        )
        if input_spec.numel() == 0:
            continue

        input_spec = input_spec.to(device)
        gt_signal = gt_signal.to(device)

        output, _ = eval_fn(model, input_spec, configs)
        out_wave = _waveform_from_output(output, n_fft, hop_length, device)

        min_l = min(out_wave.shape[-1], gt_signal.shape[-1])
        stoi_scores.append(metrics.stoi_score(out_wave, gt_signal).detach().cpu())
        loss_mr = loss_MR(
            out_wave[..., :min_l],
            gt_signal[..., :min_l],
            nffts=[128, 256, 512, 1024],
            gamma=0.3,
        )
        loss = loss_mr
        loss.backward()

        if _has_nan_grads(model):
            raise RuntimeError("NaN gradients detected")
        optimizer.step()
        optimizer.zero_grad()
        if _has_nan_weights(model):
            raise RuntimeError("NaN weights detected")

        if epoch % compute_nisqa_every == 0:
            nisqa = metrics.nisqa_score(out_wave)
            if nisqa is not None:
                total_nisqa += nisqa
                nisqa_batches += 1

        total_loss += loss.detach().cpu().item()
        total_loss_mr += loss_mr.detach().cpu().item()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Train loader produced no batches")

    # Notebooks average over len(loader), not only non-empty batches after NaN filter.
    out_nisqa = total_nisqa / n_loader_batches if nisqa_batches else None
    out_stoi = torch.hstack(stoi_scores).mean().item()
    return (
        total_loss / n_loader_batches,
        total_loss_mr / n_loader_batches,
        0.0,
        out_nisqa,
        None,
        out_stoi,
        None,
    )


@torch.no_grad()
def evaluate_epoch(
    model: torch.nn.Module,
    configs,
    val_loader: DataLoader,
    *,
    eval_fn: Callable,
    metrics: MetricsContext,
    device: torch.device,
    epoch: int = 0,
    n_fft: int,
    hop_length: int,
    compute_nisqa_every: int = 5,
) -> tuple[
    float, float, float, Optional[torch.Tensor], Optional[float], float, Optional[float]
]:
    model.eval()
    total_loss = 0.0
    total_loss_mr = 0.0
    total_nisqa = torch.zeros(5)
    nisqa_batches = 0
    stoi_scores: list[torch.Tensor] = []
    n_batches = 0
    n_loader_batches = len(val_loader)

    for input_spec, gt_signal, _input_signal in tqdm(val_loader, desc="Validate"):
        input_spec, gt_signal, _ = _filter_nan_specs(
            input_spec, gt_signal, _input_signal
        )
        if input_spec.numel() == 0:
            continue

        input_spec = input_spec.to(device)
        gt_signal = gt_signal.to(device)

        output, _ = eval_fn(model, input_spec, configs)
        out_wave = _waveform_from_output(output, n_fft, hop_length, device)

        min_l = min(out_wave.shape[-1], gt_signal.shape[-1])
        stoi_scores.append(metrics.stoi_score(out_wave, gt_signal).detach().cpu())
        loss_mr = loss_MR(
            out_wave[..., :min_l],
            gt_signal[..., :min_l],
            nffts=[128, 256, 512, 1024],
            gamma=0.3,
        )
        loss = loss_mr

        if epoch % compute_nisqa_every == 0:
            nisqa = metrics.nisqa_score(out_wave)
            if nisqa is not None:
                total_nisqa += nisqa
                nisqa_batches += 1

        total_loss += loss.detach().cpu().item()
        total_loss_mr += loss_mr.detach().cpu().item()
        n_batches += 1

    if n_batches == 0:
        raise RuntimeError("Validation loader produced no batches")

    out_nisqa = total_nisqa / n_loader_batches if nisqa_batches else None
    out_stoi = torch.hstack(stoi_scores).mean().item()
    return (
        total_loss / n_loader_batches,
        total_loss_mr / n_loader_batches,
        0.0,
        out_nisqa,
        None,
        out_stoi,
        None,
    )


def get_model_name(chkp_folder: str, model_name: Optional[str]) -> str:
    if model_name is None:
        if os.path.exists(chkp_folder):
            num_starts = len(os.listdir(chkp_folder)) + 1
        else:
            num_starts = 1
        model_name = f"model#{num_starts}"
    elif "#" not in model_name:
        model_name += "#0"

    changed = False
    while os.path.exists(os.path.join(chkp_folder, model_name + ".pt")):
        base, ind = model_name.split("#")
        model_name = f"{base}#{int(ind) + 1}"
        changed = True
    if changed:
        warnings.warn(
            f"Checkpoint name already exists; using {model_name} to avoid overwrite."
        )
    return model_name


def _get_lr(optimizer: torch.optim.Optimizer) -> float:
    return optimizer.param_groups[0]["lr"]


def _init_plots() -> dict[str, list]:
    return {
        "train loss": [],
        "train loss MR": [],
        "train loss MR low": [],
        "train loss SI-SNR": [],
        "train NISQA": [],
        "train SRMR": [],
        "train STOI": [],
        "train PESQ": [],
        "val loss": [],
        "val loss MR": [],
        "val loss MR low": [],
        "val loss SI-SNR": [],
        "val NISQA": [],
        "val SRMR": [],
        "val PESQ": [],
        "val STOI": [],
        "learning rate": [],
    }


def _append_metric(plots: dict, key: str, value, *, copy_last: bool):
    if value is None and plots[key]:
        plots[key].append(plots[key][-1])
    elif value is None:
        plots[key].append(None)
    else:
        plots[key].append(value)
    if copy_last and value is None and len(plots[key]) >= 2:
        plots[key][-1] = plots[key][-2]


def save_training_plots(
    plots: dict[str, list], epoch: int, epochs: int, out_path: Path
) -> None:
    fig, axes = plt.subplots(4, 2, figsize=(20, 16))
    fig.suptitle(f"Training progress #{epoch}/{epochs}")
    x = np.arange(1, epoch + 1)

    def _plot_pair(ax, title: str, train_key: str, val_key: str):
        ax.set_title(title)
        if plots[train_key]:
            ax.plot(x, plots[train_key][:epoch], "r.-", label="train", alpha=0.7)
        if plots.get(val_key):
            ax.plot(x, plots[val_key][:epoch], "g.-", label="val", alpha=0.7)
        ax.grid(True)
        ax.legend()

    axes[0, 0].set_title("Learning rate")
    axes[0, 0].plot(plots["learning rate"][:epoch], "b.-", alpha=0.7)
    axes[0, 0].grid(True)

    _plot_pair(axes[0, 1], "Loss", "train loss", "val loss")
    _plot_pair(axes[1, 0], "Loss MR", "train loss MR", "val loss MR")
    _plot_pair(axes[1, 1], "STOI", "train STOI", "val STOI")
    _plot_pair(axes[2, 0], "Loss MR low", "train loss MR low", "val loss MR low")
    _plot_pair(axes[2, 1], "Loss SI-SNR", "train loss SI-SNR", "val loss SI-SNR")

    if plots["train NISQA"] and any(
        p is not None for p in plots["train NISQA"][:epoch]
    ):
        nisqa_plot = torch.stack(
            [p for p in plots["train NISQA"][:epoch] if p is not None]
        )
        axes[3, 0].set_title("Train NISQA")
        for i, label in enumerate(["MOS", "NOI", "DISC", "COL", "LOUD"]):
            axes[3, 0].plot(
                np.arange(1, len(nisqa_plot) + 1), nisqa_plot[..., i], ".-", label=label
            )
        axes[3, 0].grid(True)
        axes[3, 0].legend()

    if plots["val NISQA"] and any(p is not None for p in plots["val NISQA"][:epoch]):
        nisqa_plot = torch.stack(
            [p for p in plots["val NISQA"][:epoch] if p is not None]
        )
        axes[3, 1].set_title("Val NISQA")
        for i, label in enumerate(["MOS", "NOI", "DISC", "COL", "LOUD"]):
            axes[3, 1].plot(
                np.arange(1, len(nisqa_plot) + 1), nisqa_plot[..., i], ".-", label=label
            )
        axes[3, 1].grid(True)
        axes[3, 1].legend()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def learning_loop(
    model: torch.nn.Module,
    configs,
    optimizer: torch.optim.Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_dataset: SignalDataset,
    val_dataset: SignalDataset,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
    metrics: MetricsContext,
    eval_fn: Callable,
    *,
    epochs: int = 50,
    val_every: int = 1,
    model_name: Optional[str] = None,
    chkp_folder: str = "checkpoints/fspen_chkp",
    plots: Optional[dict] = None,
    starting_epoch: int = 0,
    device: torch.device,
    n_fft: int,
    hop_length: int,
    plot_every: int = 1,
) -> tuple[torch.nn.Module, torch.optim.Optimizer, dict]:
    model_name = get_model_name(chkp_folder, model_name)
    plots = plots or _init_plots()
    os.makedirs(chkp_folder, exist_ok=True)

    max_mos = -1.0
    total_epochs = starting_epoch + epochs

    for epoch in range(starting_epoch + 1, total_epochs + 1):
        print(f"#{epoch}/{total_epochs}")
        train_dataset.set_epoch(epoch)
        val_dataset.set_epoch(epoch)
        plots["learning rate"].append(_get_lr(optimizer))

        (
            train_loss,
            train_loss_mr,
            train_loss_mr_low,
            train_nisqa,
            train_srmr,
            train_stoi,
            train_pesq,
        ) = train_epoch(
            model,
            configs,
            train_loader,
            optimizer,
            eval_fn=eval_fn,
            metrics=metrics,
            device=device,
            epoch=epoch - 1,
            n_fft=n_fft,
            hop_length=hop_length,
        )

        plots["train loss"].append(train_loss)
        plots["train loss MR"].append(train_loss_mr)
        plots["train loss MR low"].append(train_loss_mr_low)
        plots["train loss SI-SNR"].append(0.0)
        plots["train STOI"].append(train_stoi)

        copy_nisqa = (epoch - 1) % 5 != 0
        if train_nisqa is not None:
            plots["train NISQA"].append(train_nisqa[None, :])
            plots["train SRMR"].append(train_srmr)
            plots["train PESQ"].append(train_pesq)
        elif plots["train NISQA"]:
            _append_metric(plots, "train NISQA", None, copy_last=copy_nisqa)
            _append_metric(plots, "train SRMR", None, copy_last=copy_nisqa)
            _append_metric(plots, "train PESQ", None, copy_last=copy_nisqa)
        else:
            plots["train NISQA"].append(None)
            plots["train SRMR"].append(None)
            plots["train PESQ"].append(None)

        val_loss = None
        val_nisqa = None
        if epoch % val_every == 0:
            (
                val_loss,
                val_loss_mr,
                val_loss_mr_low,
                val_nisqa,
                val_srmr,
                val_stoi,
                val_pesq,
            ) = evaluate_epoch(
                model,
                configs,
                val_loader,
                eval_fn=eval_fn,
                metrics=metrics,
                device=device,
                epoch=epoch - 1,
                n_fft=n_fft,
                hop_length=hop_length,
            )
            plots["val loss"].append(val_loss)
            plots["val loss MR"].append(val_loss_mr)
            plots["val loss MR low"].append(val_loss_mr_low)
            plots["val loss SI-SNR"].append(0.0)
            plots["val STOI"].append(val_stoi)

            if val_nisqa is not None:
                plots["val NISQA"].append(val_nisqa[None, :])
                plots["val SRMR"].append(val_srmr)
                plots["val PESQ"].append(val_pesq)
            elif plots["val NISQA"]:
                _append_metric(plots, "val NISQA", None, copy_last=copy_nisqa)
                _append_metric(plots, "val SRMR", None, copy_last=copy_nisqa)
                _append_metric(plots, "val PESQ", None, copy_last=copy_nisqa)
            else:
                plots["val NISQA"].append(None)
                plots["val SRMR"].append(None)
                plots["val PESQ"].append(None)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "plots": plots,
            "config_name": configs.__class__.__name__,
        }
        torch.save(checkpoint, os.path.join(chkp_folder, model_name + ".pt"))
        print(f"Saved checkpoint: {os.path.join(chkp_folder, model_name + '.pt')}")

        if (
            val_nisqa is not None
            and len(plots["val NISQA"]) > 0
            and plots["val NISQA"][-1] is not None
            and plots["val NISQA"][-1][0][0] > max_mos
        ):
            max_mos = float(plots["val NISQA"][-1][0][0])
            best_path = os.path.join(chkp_folder, model_name + "_best_mos.pt")
            torch.save(checkpoint, best_path)
            print(f"New best MOS checkpoint: {best_path} ({max_mos:.3f})")

        if scheduler is not None:
            if (
                val_loss is not None
                and hasattr(scheduler, "step")
                and "ReduceLROnPlateau" in scheduler.__class__.__name__
            ):
                scheduler.step(val_loss)
            else:
                scheduler.step()

        if epoch % plot_every == 0:
            plot_path = Path("plots") / f"{model_name}_epoch{epoch:03d}.png"
            save_training_plots(plots, epoch, total_epochs, plot_path)
            print(f"Saved plot: {plot_path}")

        if val_loss is not None:
            print(
                f"  train loss={train_loss:.4f} | val loss={val_loss:.4f} | "
                f"train STOI={train_stoi:.3f} | val STOI={val_stoi:.3f}"
            )
        else:
            print(f"  train loss={train_loss:.4f} | train STOI={train_stoi:.3f}")

    return model, optimizer, plots


def build_dataloaders(
    *,
    data_dir: str,
    val_data_dir: str,
    noise_dir_train: str,
    noise_dir_val: str,
    rir_dirs_train: list[str],
    rir_dirs_val: list[str],
    sample_rate: int,
    snr: list[int],
    noise_proba: float,
    rir_proba: float,
    max_seq_len: int,
    val_partition: Optional[int],
    batch_size: int,
    num_workers: int,
    n_fft: int,
    hop_length: int,
    device: torch.device,
    seed: int,
) -> tuple[SignalDataset, SignalDataset, DataLoader, DataLoader]:
    train_dataset = SignalDataset(
        data_dir,
        sr=sample_rate,
        noise_dir=noise_dir_train,
        rir_dir=rir_dict_from_paths(rir_dirs_train),
        snr=snr,
        rir_proba=rir_proba,
        noise_proba=noise_proba,
        max_seq_len=max_seq_len,
        base_seed=seed,
    )
    val_dataset = SignalDataset(
        val_data_dir,
        sr=sample_rate,
        noise_dir=noise_dir_val,
        rir_dir=rir_dict_from_paths(rir_dirs_val),
        snr=snr,
        rir_proba=rir_proba,
        noise_proba=noise_proba,
        max_seq_len=max_seq_len,
        partition=val_partition,
        base_seed=seed,
    )

    collate_fn = make_collate_fn(n_fft, hop_length)
    worker_init_fn = make_worker_init_fn(seed)
    gen = torch.Generator()
    gen.manual_seed(seed)
    pin_memory = device.type == "cuda"
    n_devices = max(torch.cuda.device_count(), 1)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * n_devices,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * n_devices,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        worker_init_fn=worker_init_fn,
        generator=gen,
    )
    return train_dataset, val_dataset, train_loader, val_loader
