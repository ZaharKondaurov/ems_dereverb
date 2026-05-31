#!/usr/bin/env python3
"""
Live microphone demo: A/B between dry (latency-matched) and FSPEN-enhanced audio.

Controls (while running):
  e / E  — enhanced (model on)
  b / B  — bypass (dry signal, same latency)
  q / Q  — quit

Examples:
  python demo_mic.py --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_enc_ext_1986#0.pt
  python demo_mic.py --list-devices
  python demo_mic.py --file noisy.wav --out enhanced.wav
"""

from __future__ import annotations

import argparse
import os
import queue
import sys
import threading
import time
from collections import deque
from typing import Deque

import numpy as np

BASE_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, BASE_DIR)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FSPEN live mic demo (A/B bypass)")
    p.add_argument(
        "--checkpoint",
        default=os.path.join(
            "checkpoints",
            "fspen_chkp",
            "TrainConfig_48kHz_enc_ext_1986#0.pt",
        ),
        help="Path to model checkpoint (.pt)",
    )
    p.add_argument(
        "--config",
        default="TrainConfig_48kHz_enc_ext",
        help="Config class name in src.fspen_configs",
    )
    p.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Torch device for inference",
    )
    p.add_argument(
        "--chunk-ms",
        type=float,
        default=250.0,
        help="Processing chunk length in milliseconds",
    )
    p.add_argument(
        "--block-ms",
        type=float,
        default=10.0,
        help="Audio I/O block size in milliseconds",
    )
    p.add_argument(
        "--input-device",
        type=int,
        default=None,
        help="sounddevice input device index",
    )
    p.add_argument(
        "--output-device",
        type=int,
        default=None,
        help="sounddevice output device index",
    )
    p.add_argument(
        "--list-devices",
        action="store_true",
        help="List PortAudio devices and exit",
    )
    p.add_argument(
        "--file",
        type=str,
        default=None,
        help="Offline mode: process WAV file instead of microphone",
    )
    p.add_argument(
        "--out",
        type=str,
        default=None,
        help="Output WAV path (offline mode)",
    )
    p.add_argument(
        "--chunked",
        action="store_true",
        help="Offline: split long files into chunks (default: whole-file STFT)",
    )
    p.add_argument(
        "--enhanced",
        action="store_true",
        default=True,
        help="Start in enhanced mode (default)",
    )
    p.add_argument(
        "--bypass",
        action="store_true",
        help="Start in bypass mode",
    )
    return p.parse_args()


def _list_devices() -> None:
    import sounddevice as sd

    print(sd.query_devices())


def _resolve_stream_device(
    input_device: int | None, output_device: int | None
) -> tuple[int | None, int | None] | None:
    """Map CLI device indices to sounddevice ``device`` argument (input, output)."""
    if input_device is None and output_device is None:
        return None
    return (input_device, output_device)


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
        """Return ``n`` samples from the live mic, delayed by ``delay`` samples."""
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


def _keyboard_thread(state: dict) -> None:
    """Toggle enhanced/bypass from terminal (non-blocking on Unix)."""
    import select
    import termios
    import tty

    if not sys.stdin.isatty():
        return

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not state["stop"]:
            if select.select([sys.stdin], [], [], 0.1)[0]:
                ch = sys.stdin.read(1).lower()
                if ch in ("e",):
                    state["enhanced"] = True
                    state["status"] = "ENHANCED"
                elif ch in ("b",):
                    state["enhanced"] = False
                    state["status"] = "BYPASS"
                elif ch in ("q",):
                    state["stop"] = True
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class LiveDemo:
    def __init__(
        self,
        enhancer,
        sample_rate: int,
        blocksize: int,
        enhanced: bool = True,
    ):
        self.enhancer = enhancer
        self.sample_rate = sample_rate
        self.blocksize = blocksize

        self.enhanced = enhanced
        self._lock = threading.Lock()

        # Output queue fed by processing thread; playback reads fixed blocks.
        self._out_queue: Deque[float] = deque()
        self._in_queue: queue.Queue[np.ndarray] = queue.Queue()

        # Latency-matched dry path: align with analysis window + hop.
        delay_samples = enhancer.chunk_samples + enhancer.hop_samples
        self._dry_delay = LatencyMatchedDelayLine(delay_samples)
        # ~1 chunk of headroom before enhanced playback (reduces underrun clicks)
        self._min_out_buffer = max(enhancer.chunk_samples // 2, blocksize * 4)

        self._warmup_done = False
        self._stats_last = time.time()
        self._blocks_played = 0
        self._worker_stop = threading.Event()
        self._worker = threading.Thread(target=self._inference_worker, daemon=True)
        self._worker.start()

    def set_enhanced(self, value: bool) -> None:
        with self._lock:
            self.enhanced = value

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
                pulled = self.enhancer.pull(self.blocksize * 8)
                if not pulled.size:
                    break
                self._out_queue.extend(pulled.tolist())

            if len(self._out_queue) >= self._min_out_buffer:
                self._warmup_done = True

    def _processing_step(self, indata: np.ndarray) -> None:
        mono = indata[:, 0].astype(np.float32).copy()
        self._dry_delay.write(mono)
        self._in_queue.put(mono)

    def _read_playback_block(self) -> np.ndarray:
        n = self.blocksize
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
                # Underrun: use latency-matched dry instead of silence (avoids gaps)
                out[avail:] = self._dry_delay.read(n - avail)
        else:
            out = self._dry_delay.read(n)

        # Soft limiter
        peak = np.max(np.abs(out))
        if peak > 0.99:
            out = out * (0.99 / peak)
        return out

    def audio_callback(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            print(f"\n[audio] {status}", file=sys.stderr)

        self._processing_step(indata)
        outdata[:, 0] = self._read_playback_block()
        if outdata.shape[1] > 1:
            outdata[:, 1] = outdata[:, 0]

        self._blocks_played += 1
        now = time.time()
        if now - self._stats_last >= 1.0:
            self._print_stats()
            self._stats_last = now

    def _print_stats(self) -> None:
        mode = "ENHANCED" if self.enhanced else "BYPASS"
        warmup = "ready" if self._warmup_done else "warming up…"
        rtf = self.enhancer.mean_rtf
        lat_ms = self.enhancer.algorithmic_latency_sec * 1000
        q_out = len(self._out_queue)
        q_dry = len(self._dry_delay)
        line = (
            f"\r[{mode}] {warmup} | latency≈{lat_ms:.0f}ms | "
            f"RTF={rtf:.2f} | out_q={q_out} dry_q={q_dry}   "
        )
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        self._worker_stop.set()
        self._worker.join(timeout=2.0)
        tail = self.enhancer.flush()
        if tail.size:
            self._out_queue.extend(tail.tolist())


def run_mic(args: argparse.Namespace) -> None:
    import sounddevice as sd
    import torch

    from src.streaming import StreamingEnhancer, load_enhancer

    if not os.path.isfile(args.checkpoint):
        print(
            f"Checkpoint not found: {args.checkpoint}\n"
            "Pass --checkpoint PATH to a trained .pt file.",
            file=sys.stderr,
        )
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU.", file=sys.stderr)
        device = "cpu"

    print(f"Loading model from {args.checkpoint} …")
    model, configs = load_enhancer(args.checkpoint, device=device, config_name=args.config)
    sr = configs.sample_rate
    chunk_samples = max(int(sr * args.chunk_ms / 1000), configs.hop_length * 4)
    blocksize = max(int(sr * args.block_ms / 1000), 64)

    enhancer = StreamingEnhancer(
        model, configs, device=device, chunk_samples=chunk_samples
    )

    enhanced = not args.bypass
    demo = LiveDemo(enhancer, sr, blocksize, enhanced=enhanced)

    state = {
        "enhanced": enhanced,
        "status": "ENHANCED" if enhanced else "BYPASS",
        "stop": False,
    }
    kb = threading.Thread(target=_keyboard_thread, args=(state,), daemon=True)
    kb.start()

    def sync_mode_from_state() -> None:
        while not state["stop"]:
            demo.set_enhanced(state["enhanced"])
            time.sleep(0.05)

    sync_thread = threading.Thread(target=sync_mode_from_state, daemon=True)
    sync_thread.start()

    print(
        f"\nFSPEN live demo @ {sr} Hz | chunk={chunk_samples / sr * 1000:.0f} ms | "
        f"block={blocksize / sr * 1000:.1f} ms"
    )
    print("Wear headphones to avoid feedback.")
    print("Keys: [E] enhanced  [B] bypass  [Q] quit\n")

    stream_device = _resolve_stream_device(args.input_device, args.output_device)
    if stream_device is not None:
        in_id, out_id = stream_device
        if in_id is not None:
            in_info = sd.query_devices(in_id)
            print(f"Audio in:  [{in_id}] {in_info['name']}")
        if out_id is not None:
            out_info = sd.query_devices(out_id)
            print(f"Audio out: [{out_id}] {out_info['name']}")
        print()

    try:
        with sd.Stream(
            samplerate=sr,
            blocksize=blocksize,
            dtype="float32",
            channels=1,
            device=stream_device,
            callback=demo.audio_callback,
        ):
            while not state["stop"]:
                time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        demo.finish()
        print("\nStopped.")


def run_offline(args: argparse.Namespace) -> None:
    import torch
    import torchaudio

    from src.streaming import StreamingEnhancer, load_enhancer

    if not os.path.isfile(args.checkpoint):
        print(f"Checkpoint not found: {args.checkpoint}", file=sys.stderr)
        sys.exit(1)
    if args.file is None:
        print("--file is required for offline mode", file=sys.stderr)
        sys.exit(1)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model, configs = load_enhancer(args.checkpoint, device=device, config_name=args.config)
    sr = configs.sample_rate
    chunk_samples = max(int(sr * args.chunk_ms / 1000), configs.hop_length * 4)
    enhancer = StreamingEnhancer(
        model, configs, device=device, chunk_samples=chunk_samples
    )

    wav, file_sr = torchaudio.load(args.file)
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if file_sr != sr:
        wav = torchaudio.functional.resample(wav, file_sr, sr)
    mono = wav.reshape(-1).numpy().astype(np.float32)

    out = enhancer.process_file(mono, chunked=args.chunked)

    out_path = args.out or args.file.replace(".wav", "_enhanced.wav")
    out_t = torch.from_numpy(out).unsqueeze(0)
    torchaudio.save(out_path, out_t, sr)
    print(f"Saved {out_path} ({len(out) / sr:.2f}s, mean RTF={enhancer.mean_rtf:.2f})")


def main() -> None:
    args = _parse_args()
    if args.list_devices:
        _list_devices()
        return
    if args.file:
        run_offline(args)
    else:
        import torch
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch._logging.set_logs(graph_code=False)
        run_mic(args)


if __name__ == "__main__":
    main()
