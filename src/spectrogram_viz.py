"""Log-magnitude spectrograms for live monitoring."""

from __future__ import annotations

import numpy as np
import torch

from src.utils import vorbis_window


def mel_forward(f: np.ndarray) -> np.ndarray:
    return 2595 * np.log10(1 + f / 700)


def mel_inverse(m: np.ndarray) -> np.ndarray:
    return (10 ** (m / 2595) - 1) * 700


def waveform_to_log_spec(
    waveform: np.ndarray,
    *,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    max_frames: int | None = None,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute log-power spectrogram (dB) for display.

    Returns
    -------
    spec_db : (n_freq, n_frames) float32
    times : (n_frames,) seconds
    freqs : (n_freq,) Hz
    """
    wave = np.asarray(waveform, dtype=np.float32).reshape(-1)
    if wave.size < hop_length:
        n_freq = n_fft // 2 + 1
        return (
            np.full((n_freq, 1), -80.0, dtype=np.float32),
            np.array([0.0], dtype=np.float32),
            np.linspace(0, sample_rate / 2, n_freq, dtype=np.float32),
        )

    x = torch.from_numpy(wave).unsqueeze(0)
    window = vorbis_window(n_fft)

    with torch.inference_mode():
        spec = torch.stft(
            x,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            return_complex=True,
            normalized=True,
            center=True,
        )

    power = spec.abs().pow(2).squeeze(0).cpu().numpy()
    spec_db = (10.0 * np.log10(power + eps)).astype(np.float32)

    if max_frames is not None and spec_db.shape[1] > max_frames:
        spec_db = spec_db[:, -max_frames:]

    n_frames = spec_db.shape[1]
    times = np.arange(n_frames, dtype=np.float32) * hop_length / sample_rate
    freqs = np.linspace(0, sample_rate / 2, spec_db.shape[0], dtype=np.float32)
    return spec_db, times, freqs


def log_spec_to_rgb(
    spec_db: np.ndarray,
    sample_rate: int,
    *,
    hop_length: int = 512,
    title: str = "",
    vmin: float = -80.0,
    vmax: float | None = None,
) -> np.ndarray:
    """Render log-power spectrogram to an RGB uint8 image."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if vmax is None:
        vmax = max(-20.0, float(spec_db.max()))

    n_frames = spec_db.shape[1]
    freqs = np.linspace(0, sample_rate / 2, spec_db.shape[0])
    t_end = n_frames * hop_length / sample_rate
    t_start = 0.0

    fig, ax = plt.subplots(figsize=(6.4, 2.6), dpi=100)
    ax.imshow(
        spec_db,
        aspect="auto",
        origin="lower",
        cmap="magma",
        vmin=vmin,
        vmax=vmax,
        extent=[t_start, t_end, freqs[0], freqs[-1]],
    )
    ax.set_yscale("function", functions=(mel_forward, mel_inverse))
    ax.set_ylim(0, sample_rate // 2)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if title:
        ax.set_title(title, fontsize=10)
    fig.tight_layout()

    fig.canvas.draw()
    h, w = fig.canvas.get_width_height()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
    plt.close(fig)
    return buf.reshape(h, w, 4)[:, :, :3]
