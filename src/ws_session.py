"""WebSocket streaming session: enhancement + spectrogram columns."""

from __future__ import annotations

import numpy as np

from src.live_session import LiveStreamCore
from src.spectrogram_viz import waveform_to_log_spec
from src.streaming import StreamingEnhancer

# Browser capture block (~85 ms @ 48 kHz); must be a power of two for ScriptProcessor.
WEB_IO_BLOCKSIZE = 4096


class WebStreamSession:
    """Process mic chunks over WebSocket; emit audio + STFT columns for canvas UI."""

    def __init__(
        self,
        enhancer: StreamingEnhancer,
        *,
        history_sec: float = 2.5,
        n_fft: int,
        hop_length: int,
        io_blocksize: int = WEB_IO_BLOCKSIZE,
    ):
        self.enhancer = enhancer
        self.sr = enhancer.sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_freq = n_fft // 2 + 1
        self.history_samples = int(history_sec * self.sr)

        self._core = LiveStreamCore(enhancer, io_blocksize=io_blocksize, enhanced=True)
        self.enhanced = True

        self.mic_viz = np.zeros(0, dtype=np.float32)
        self.play_viz = np.zeros(0, dtype=np.float32)

    def set_enhanced(self, value: bool) -> None:
        self.enhanced = value
        self._core.set_enhanced(value)

    def reset(self) -> None:
        self._core.reset()
        self.enhanced = True
        self.mic_viz = np.zeros(0, dtype=np.float32)
        self.play_viz = np.zeros(0, dtype=np.float32)

    @staticmethod
    def _resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
        if sr_from == sr_to or y.size == 0:
            return y
        n_out = max(1, int(round(y.size * sr_to / sr_from)))
        x_old = np.linspace(0.0, 1.0, y.size, endpoint=False)
        x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
        return np.interp(x_new, x_old, y).astype(np.float32)

    def _append_viz(self, buf: np.ndarray, block: np.ndarray) -> np.ndarray:
        if block.size == 0:
            return buf
        buf = np.concatenate([buf, block])
        if buf.size > self.history_samples:
            buf = buf[-self.history_samples :]
        return buf

    def _spec_columns_for_block(
        self, wave: np.ndarray, block_samples: int
    ) -> list[list[float]]:
        if wave.size < self.n_fft or block_samples <= 0:
            return []
        spec_db, _, _ = waveform_to_log_spec(
            wave,
            sample_rate=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop,
        )
        n_new = max(1, int(np.ceil(block_samples / self.hop)))
        n_take = min(spec_db.shape[1], n_new + 1)
        start = spec_db.shape[1] - n_take
        cols = spec_db[:, start:]
        return [cols[:, i].tolist() for i in range(cols.shape[1])]

    def process(
        self, y: np.ndarray, *, enhanced: bool, input_sr: int | None = None
    ) -> dict:
        if self.enhanced != enhanced:
            self.set_enhanced(enhanced)

        y = np.asarray(y, dtype=np.float32).reshape(-1)
        if input_sr and input_sr != self.sr:
            y = self._resample(y, input_sr, self.sr)

        if y.size == 0:
            return {
                "audio": [],
                "spec_in_cols": [],
                "spec_out_cols": [],
                "warmup": self._core.warmup_done,
                "enhanced": self.enhanced,
                "sr": self.sr,
                "n_freq": self.n_freq,
            }

        self._core.push_input(y)
        out = self._core.read_output(y.size)

        self.mic_viz = self._append_viz(self.mic_viz, y)
        self.play_viz = self._append_viz(self.play_viz, out)

        spec_in = self._spec_columns_for_block(self.mic_viz, y.size)
        spec_out = self._spec_columns_for_block(self.play_viz, out.size)

        return {
            "audio": out.tolist(),
            "spec_in_cols": spec_in,
            "spec_out_cols": spec_out,
            "warmup": self._core.warmup_done,
            "enhanced": self.enhanced and self._core.warmup_done,
            "sr": self.sr,
            "n_freq": self.n_freq,
            "out_q": self._core.out_queue_len,
        }
