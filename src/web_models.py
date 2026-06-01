"""FSPEN model presets and file enhancement for the web app."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from src.streaming import StreamingEnhancer

CHECKPOINT_SUBDIR = Path("checkpoints") / "fspen_chkp"


@dataclass(frozen=True)
class ModelPreset:
    id: str
    label: str
    config: str
    checkpoint: str
    model_class: str
    eval_fn: str

    def checkpoint_path(self, project_root: Path) -> Path:
        return (project_root / CHECKPOINT_SUBDIR / self.checkpoint).resolve()


MODEL_PRESETS: tuple[ModelPreset, ...] = (
    ModelPreset(
        id="fspen_48khz",
        label="FSPEN+48kHz",
        config="TrainConfig_48khz",
        checkpoint="TrainConfig_48khz_baseline.pt",
        model_class="FullSubPathExtension",
        eval_fn="model_eval_old",
    ),
    ModelPreset(
        id="fspen_48khz_overlap",
        label="FSPEN+48kHz+overlap",
        config="TrainConfig_48kHz_overlap",
        checkpoint="TrainConfig_48kHz_overlap.pt",
        model_class="FullSubPathExtension",
        eval_fn="model_eval_old",
    ),
    ModelPreset(
        id="fspen_48khz_sble",
        label="FSPEN+48kHz+SBLE",
        config="TrainConfig_48kHz_enc_ext",
        checkpoint="TrainConfig_48kHz_enc_ext.pt",
        model_class="FullSubPathExtension_ext",
        eval_fn="model_eval",
    ),
    ModelPreset(
        id="fspen_48khz_sbdc_overlap",
        label="FSPEN+48kHz+SBDC+overlap",
        config="TrainConfig_48kHz_enc_ext_lay_1_overlap",
        checkpoint="TrainConfig_48kHz_enc_ext_lay_1_overlap.pt",
        model_class="FullSubPathExtension_ext",
        eval_fn="model_eval",
    ),
)

DEFAULT_PRESET_ID = "fspen_48khz_overlap"


def get_preset(preset_id: str) -> ModelPreset:
    for p in MODEL_PRESETS:
        if p.id == preset_id:
            return p
    ids = ", ".join(p.id for p in MODEL_PRESETS)
    raise ValueError(f"Unknown preset {preset_id!r}. Choose one of: {ids}")


def list_presets(project_root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for p in MODEL_PRESETS:
        path = p.checkpoint_path(project_root)
        out.append(
            {
                "id": p.id,
                "label": p.label,
                "config": p.config,
                "checkpoint": p.checkpoint,
                "checkpoint_path": str(path),
                "model_class": p.model_class,
                "eval_fn": p.eval_fn,
                "available": path.is_file(),
            }
        )
    return out


def enhance_wav_bytes(
    enhancer: StreamingEnhancer,
    *,
    file_bytes: bytes,
    filename: str,
    sample_rate: int,
    chunked: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Load audio from upload, run ``process_file``, return WAV bytes + metadata."""
    import torchaudio

    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(file_bytes)
        tmp.flush()
        wav, file_sr = torchaudio.load(tmp.name)

    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != sample_rate:
        wav = torchaudio.functional.resample(wav, file_sr, sample_rate)
    mono = wav.reshape(-1).numpy().astype(np.float32)

    enhancer.reset()
    out = enhancer.process_file(mono, chunked=chunked)
    duration = len(out) / sample_rate

    out_t = torch.from_numpy(out).unsqueeze(0)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as out_tmp:
        torchaudio.save(out_tmp.name, out_t, sample_rate)
        wav_bytes = Path(out_tmp.name).read_bytes()
    meta = {
        "duration_sec": round(duration, 3),
        "samples": int(out.size),
        "sample_rate": sample_rate,
        "rtf": round(float(enhancer.mean_rtf), 3),
    }
    return wav_bytes, meta
