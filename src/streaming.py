"""Stateful chunk inference for real-time FSPEN enhancement."""

from __future__ import annotations

from typing import Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch

from models.fspen import FullSubPathExtension, FullSubPathExtension_ext
from src.utils import model_eval, model_eval_old, vorbis_window

# Live AGC: do not track peaks below this (avoids boosting noise in silence).
_LIVE_PEAK_FLOOR = 0.12
_LIVE_ATTACK = 0.35
_LIVE_RELEASE = 0.002

# Configs that use FullSubPathExtension_ext + model_eval (others: model_eval_old).
_EXT_CONFIG_NAMES = frozenset(
    {
        "TrainConfig_48kHz_enc_ext",
        "TrainConfig_48kHz_enc_ext_lay_1_overlap",
    }
)


def config_uses_ext_eval(config_name: str) -> bool:
    return config_name in _EXT_CONFIG_NAMES


def load_enhancer(
    checkpoint_path: str,
    device: str = "cpu",
    config_name: str = "TrainConfig_48kHz_enc_ext",
) -> Tuple[torch.nn.Module, object]:
    """Load FSPEN model and config from a training checkpoint."""
    from src import fspen_configs

    config_cls = getattr(fspen_configs, config_name, None)
    if config_cls is None:
        raise ValueError(f"Unknown config: {config_name}")

    configs = config_cls()
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)

    state["model_state_dict"] = OrderedDict([(k, v) for k, v in state["model_state_dict"].items() if "matcher" not in k])

    model_cls = (
        FullSubPathExtension_ext
        if config_uses_ext_eval(config_name)
        else FullSubPathExtension
    )

    model = model_cls(configs=configs)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    return model, configs


def _normalize_waveform(waveform: torch.Tensor) -> torch.Tensor:
    peak = waveform.abs().max()
    if peak > 0:
        waveform = waveform / peak
    return waveform


class StreamingEnhancer:
    """
    Chunk-based enhancer with hidden-state carry-over between chunks.

    ``push()`` / ``flush()`` — overlap-hop path for low-latency live I/O.
    ``process_file()`` — offline path: one global normalize, no chunk gain jumps.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        configs,
        device: str = "cpu",
        chunk_samples: Optional[int] = None,
        hop_samples: Optional[int] = None,
        crossfade_samples: Optional[int] = None,
    ):
        self.model = model
        self.configs = configs
        self.device = torch.device(device)

        self.sample_rate = configs.sample_rate
        self.n_fft = configs.n_fft
        self.hop_length = configs.hop_length

        self.chunk_samples = chunk_samples or (self.sample_rate // 4)
        self.hop_samples = hop_samples or (self.chunk_samples // 2)
        self.crossfade_samples = crossfade_samples or self.hop_length
        self._window: Optional[torch.Tensor] = None

        self._pending = torch.zeros(0, dtype=torch.float32)
        self._h0 = None
        self._prev_tail: Optional[np.ndarray] = None
        self._live_norm_scale = 1.0
        self._emit_carry = np.zeros(0, dtype=np.float32)

        self.chunks_processed = 0
        self.total_process_time = 0.0

    @property
    def algorithmic_latency_sec(self) -> float:
        """Approximate end-to-end latency from chunk buffering (seconds)."""
        return (self.chunk_samples + self.hop_samples) / self.sample_rate

    def reset(self) -> None:
        self._pending = torch.zeros(0, dtype=torch.float32)
        self._h0 = None
        self._prev_tail = None
        self._live_norm_scale = 1.0
        self._emit_carry = np.zeros(0, dtype=np.float32)
        self.chunks_processed = 0
        self.total_process_time = 0.0

    def _get_window(self) -> torch.Tensor:
        if self._window is None:
            self._window = vorbis_window(self.n_fft).to(self.device)
        return self._window

    def _infer(
        self,
        waveform: torch.Tensor,
        *,
        update_state: bool = True,
    ) -> np.ndarray:
        """STFT → model → iSTFT for a contiguous waveform (already scaled)."""
        import time

        t0 = time.perf_counter()
        signal = waveform.unsqueeze(0).to(self.device)
        window = self._get_window()

        with torch.inference_mode():
            spec = torch.stft(
                signal,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=True,
                normalized=True,
                center=True,
            )
            h0 = self._h0

            if isinstance(self.model, FullSubPathExtension_ext):
                enhanced_spec, new_h0 = model_eval(
                    self.model, spec, self.configs, h0=h0
                )
            else:
                enhanced_spec, new_h0 = model_eval_old(
                    self.model, spec, self.configs, h0=h0
                )
            if update_state:
                self._h0 = new_h0

            out_wave = torch.istft(
                enhanced_spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                win_length=self.n_fft,
                window=window,
                return_complex=False,
                normalized=True,
                center=True,
            )

        out = out_wave.reshape(-1).detach().cpu().numpy().astype(np.float32)
        target_len = waveform.numel()
        if out.shape[0] > target_len:
            out = out[:target_len]
        elif out.shape[0] < target_len:
            out = np.pad(out, (0, target_len - out.shape[0]))

        self.chunks_processed += 1
        self.total_process_time += time.perf_counter() - t0
        return out

    def process_file(
        self,
        samples: np.ndarray,
        *,
        chunked: bool = False,
    ) -> np.ndarray:
        """
        Enhance a full recording (offline).

        Uses a single peak normalization for the whole file so level does not
        jump at chunk boundaries. By default runs one STFT over the entire file;
        use ``chunked=True`` for very long files (contiguous chunks + stateful h0).
        """
        x = torch.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1))
        x = _normalize_waveform(x)
        self.reset()

        if not chunked or x.numel() <= self.chunk_samples:
            out = self._infer(x, update_state=False)
        else:
            out = self._process_file_chunked(x)

        if out.shape[0] > x.numel():
            out = out[: x.numel()]
        elif out.shape[0] < x.numel():
            out = np.pad(out, (0, x.numel() - out.shape[0]))
        return out

    def _process_file_chunked(self, x: torch.Tensor) -> np.ndarray:
        """Contiguous non-overlapping chunks; short crossfade only at seams."""
        parts: list[np.ndarray] = []
        n = x.numel()
        pos = 0
        seam = min(self.crossfade_samples, self.hop_length)

        while pos < n:
            end = min(pos + self.chunk_samples, n)
            chunk_len = end - pos
            chunk = x[pos:end]
            if chunk.numel() < self.chunk_samples:
                chunk = torch.cat(
                    [chunk, torch.zeros(self.chunk_samples - chunk.numel())]
                )
            out = self._infer(chunk, update_state=True)[:chunk_len]

            if parts and seam > 0:
                fade = np.linspace(0.0, 1.0, seam, dtype=np.float32)
                blend = parts[-1][-seam:] * (1.0 - fade) + out[:seam] * fade
                parts[-1] = np.concatenate([parts[-1][:-seam], blend])
                out = out[seam:]

            if out.size:
                parts.append(out)
            pos = end

        if not parts:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(parts)

    def _update_live_scale(self, peak: float) -> None:
        """Slow-release, fast-attack peak tracker with a noise floor."""
        target = max(peak, _LIVE_PEAK_FLOOR)
        if target >= self._live_norm_scale:
            self._live_norm_scale += _LIVE_ATTACK * (target - self._live_norm_scale)
        else:
            self._live_norm_scale += _LIVE_RELEASE * (target - self._live_norm_scale)
        self._live_norm_scale = max(self._live_norm_scale, _LIVE_PEAK_FLOOR)

    def _scale_for_model(self, waveform: torch.Tensor) -> torch.Tensor:
        peak = waveform.abs().max().item()
        if peak > 0:
            self._update_live_scale(peak)
        return waveform / self._live_norm_scale

    def _restore_output_level(self, out: np.ndarray) -> np.ndarray:
        return out * self._live_norm_scale

    def _seam_crossfade(self, chunk: np.ndarray) -> np.ndarray:
        """Blend chunk start with the tail of the previous emitted chunk."""
        n = min(self.crossfade_samples, chunk.shape[0])
        if self._prev_tail is None or n == 0:
            self._prev_tail = chunk.copy()
            return chunk

        fade = np.linspace(0.0, 1.0, n, dtype=np.float32)
        chunk = chunk.copy()
        chunk[:n] = self._prev_tail[-n:] * (1.0 - fade) + chunk[:n] * fade
        self._prev_tail = chunk.copy()
        return chunk

    def _process_pending_chunks(self) -> None:
        while self._pending.numel() >= self.chunk_samples:
            wave = self._pending[: self.chunk_samples]
            self._pending = self._pending[self.chunk_samples :]
            out = self._restore_output_level(self._infer(wave.clone()))
            out = self._seam_crossfade(out)
            if self._emit_carry.size:
                self._emit_carry = np.concatenate([self._emit_carry, out])
            else:
                self._emit_carry = out

    def push(self, samples: np.ndarray) -> None:
        """Feed microphone samples into the live processing buffer."""
        if samples.size == 0:
            return

        chunk = torch.from_numpy(np.asarray(samples, dtype=np.float32).reshape(-1))
        scaled = self._scale_for_model(chunk)
        self._pending = torch.cat([self._pending, scaled])
        self._process_pending_chunks()

    def pull(self, n_samples: int) -> np.ndarray:
        """Return up to ``n_samples`` of enhanced audio (may be shorter if starving)."""
        if n_samples <= 0:
            return np.zeros(0, dtype=np.float32)

        n = min(n_samples, self._emit_carry.size)
        if n == 0:
            return np.zeros(0, dtype=np.float32)

        out = self._emit_carry[:n].copy()
        self._emit_carry = self._emit_carry[n:]
        return out

    def drain(self) -> np.ndarray:
        """Return all samples waiting in the playout buffer."""
        if self._emit_carry.size == 0:
            return np.zeros(0, dtype=np.float32)
        out = self._emit_carry.copy()
        self._emit_carry = np.zeros(0, dtype=np.float32)
        return out

    def flush(self) -> np.ndarray:
        """Process remaining live-buffer samples."""
        parts = []
        if self._emit_carry.size:
            parts.append(self.drain())

        if self._pending.numel() == 0:
            return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

        n_valid = self._pending.numel()
        pad = self.chunk_samples - n_valid
        wave = torch.cat([self._pending, torch.zeros(pad, dtype=torch.float32)])
        self._pending = torch.zeros(0, dtype=torch.float32)

        out = self._restore_output_level(self._infer(wave))
        out = self._seam_crossfade(out)[:n_valid]
        parts.append(out)
        return np.concatenate(parts)

    @property
    def mean_rtf(self) -> float:
        if self.chunks_processed == 0:
            return 0.0
        chunk_dur = self.chunk_samples / self.sample_rate
        mean_proc = self.total_process_time / self.chunks_processed
        return chunk_dur / mean_proc if mean_proc > 0 else 0.0
