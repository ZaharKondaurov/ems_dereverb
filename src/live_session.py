"""
Shared live enhancement pipeline (demo_mic + web).

Decouples capture from inference via a background worker, matching PortAudio-style I/O.
"""

from __future__ import annotations

import queue
import threading
from collections import deque
from typing import Deque

import numpy as np

from src.streaming import StreamingEnhancer

# Match demo_mic: limit only when peaks exceed ~full scale.
_OUTPUT_CEILING = 0.99


class LatencyMatchedDelayLine:
    """Ring buffer: read output is always ``delay_samples`` behind the latest input."""

    def __init__(self, delay_samples: int, capacity_extra: int = 8192):
        self.delay = delay_samples
        cap = delay_samples + capacity_extra
        self._buf = np.zeros(cap, dtype=np.float32)
        self._cap = cap
        self._write = 0

    def write(self, samples: np.ndarray) -> None:
        samples = np.asarray(samples, dtype=np.float32).reshape(-1)
        for s in samples:
            self._buf[self._write % self._cap] = s
            self._write += 1

    def read(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)
        end = self._write - self.delay
        start = end - n
        if end <= 0:
            return out
        for i in range(n):
            pos = start + i
            if pos < 0:
                out[i] = 0.0
            else:
                out[i] = self._buf[pos % self._cap]
        return out

    def __len__(self) -> int:
        return max(0, self._write - self.delay)


class LiveStreamCore:
    """
    Push mic blocks, pull enhanced playout (same logic as ``demo_mic.LiveDemo``).
    """

    def __init__(
        self,
        enhancer: StreamingEnhancer,
        *,
        io_blocksize: int,
        enhanced: bool = True,
    ):
        self.enhancer = enhancer
        self.io_blocksize = io_blocksize
        self.enhanced = enhanced
        self._lock = threading.Lock()

        self._out_queue: Deque[float] = deque()
        self._in_queue: queue.Queue[np.ndarray] = queue.Queue()

        delay_samples = enhancer.chunk_samples + enhancer.hop_samples
        self._dry_delay = LatencyMatchedDelayLine(delay_samples)
        self._min_out_buffer = max(enhancer.chunk_samples // 2, io_blocksize * 4)

        self._warmup_done = False
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()

    @property
    def warmup_done(self) -> bool:
        return self._warmup_done

    @property
    def out_queue_len(self) -> int:
        return len(self._out_queue)

    def set_enhanced(self, value: bool) -> None:
        with self._lock:
            if self.enhanced == value:
                return
            self.enhanced = value
            # Bypass does not consume the queue; drop stale enhanced audio on any toggle.
            self._out_queue.clear()

    def reset(self) -> None:
        self.enhancer.reset()
        delay_samples = self.enhancer.chunk_samples + self.enhancer.hop_samples
        self._dry_delay = LatencyMatchedDelayLine(delay_samples)
        with self._lock:
            while True:
                try:
                    self._in_queue.get_nowait()
                except queue.Empty:
                    break
            self._out_queue.clear()
            self._warmup_done = False

    def shutdown(self) -> None:
        self._worker_stop.set()
        self._worker.join(timeout=2.0)
        tail = self.enhancer.flush()
        if tail.size:
            self._out_queue.extend(tail.tolist())

    def _inference_worker(self) -> None:
        max_backlog = 32
        while not self._worker_stop.is_set():
            try:
                mono = self._in_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            batch = [mono]
            while True:
                try:
                    batch.append(self._in_queue.get_nowait())
                except queue.Empty:
                    break

            if len(batch) > max_backlog:
                batch = batch[-max_backlog:]

            for block in batch:
                self.enhancer.push(block)

            while True:
                pulled = self.enhancer.pull(self.io_blocksize * 8)
                if not pulled.size:
                    break
                self._out_queue.extend(pulled.tolist())

            if len(self._out_queue) >= self._min_out_buffer:
                self._warmup_done = True

    def push_input(self, mono: np.ndarray) -> None:
        mono = np.asarray(mono, dtype=np.float32).reshape(-1)
        self._dry_delay.write(mono)
        self._in_queue.put(mono.copy())

    @staticmethod
    def _limit_output(out: np.ndarray) -> np.ndarray:
        peak = float(np.max(np.abs(out)))
        if peak > _OUTPUT_CEILING:
            out = out * (_OUTPUT_CEILING / peak)
        return out

    def read_output(self, n: int) -> np.ndarray:
        out = np.zeros(n, dtype=np.float32)

        with self._lock:
            use_enhanced = (
                self.enhanced
                and self._warmup_done
                and len(self._out_queue) >= self._min_out_buffer
            )

        if use_enhanced:
            avail = min(n, len(self._out_queue))
            for i in range(avail):
                out[i] = self._out_queue.popleft()
            if avail < n:
                out[avail:] = self._dry_delay.read(n - avail)
        else:
            out = self._dry_delay.read(n)

        return self._limit_output(out)
