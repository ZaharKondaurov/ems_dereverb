#!/usr/bin/env python3
"""
FSPEN live web demo (FastAPI + WebSocket + canvas spectrograms).

Local:
  pip install -r requirements-web.txt
  python web_app.py --config TrainConfig_48kHz_overlap \\
    --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_overlap_1986#0.pt --chunk-ms 500

Hugging Face Space: see README_SPACE.md (Docker sdk, port 7860).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.streaming import StreamingEnhancer, load_enhancer  # noqa: E402
from src.ws_session import WebStreamSession  # noqa: E402

STATIC_DIR = BASE_DIR / "static"
DEFAULT_CHECKPOINT = (
    BASE_DIR / "checkpoints" / "fspen_chkp" / "TrainConfig_48kHz_overlap_1986#0.pt"
)

# Set at startup
_enhancer: Optional[StreamingEnhancer] = None
_configs = None
_sr = 48_000
_chunk_ms = 500.0
_history_sec = 2.5
_device = "cpu"
_config_name = "TrainConfig_48kHz_overlap"


def _load_model(checkpoint: str, device: str, config_name: str) -> None:
    global _enhancer, _configs, _sr
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    print(f"Loading {checkpoint} on {device} …")
    model, configs = load_enhancer(checkpoint, device=device, config_name=config_name)
    chunk_samples = max(int(configs.sample_rate * _chunk_ms / 1000), configs.hop_length * 4)
    _configs = configs
    _sr = configs.sample_rate
    _enhancer = StreamingEnhancer(model, configs, device=device, chunk_samples=chunk_samples)
    print("Model ready.")


def _new_session(enhanced: bool = True) -> WebStreamSession:
    assert _enhancer is not None and _configs is not None
    s = WebStreamSession(
        _enhancer,
        history_sec=_history_sec,
        n_fft=_configs.n_fft,
        hop_length=_configs.hop_length,
    )
    s.enhanced = enhanced
    return s


@asynccontextmanager
async def lifespan(app: FastAPI):
    ckpt = os.environ.get("FSPEN_CHECKPOINT", str(DEFAULT_CHECKPOINT))
    dev = os.environ.get("FSPEN_DEVICE", _device)
    cfg = os.environ.get("FSPEN_CONFIG", _config_name)
    _load_model(ckpt, dev, cfg)
    yield


app = FastAPI(title="FSPEN Live", lifespan=lifespan)


@app.get("/")
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        STATIC_DIR / "favicon.svg",
        media_type="image/svg+xml",
    )


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    session = _new_session(enhanced=True)
    try:
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            mtype = msg.get("type")

            if mtype == "reset":
                session.reset()
                await ws.send_json({"type": "status", "message": "reset"})
                continue

            if mtype == "config":
                session.set_enhanced(bool(msg.get("enhanced", True)))
                await ws.send_json(
                    {
                        "type": "status",
                        "enhanced": session.enhanced,
                        "flush_playback": True,
                    }
                )
                continue

            if mtype == "audio":
                samples = msg.get("data", [])
                y = np.asarray(samples, dtype=np.float32)
                input_sr = int(msg.get("sr", _sr))
                result = await asyncio.to_thread(
                    session.process,
                    y,
                    enhanced=session.enhanced,
                    input_sr=input_sr,
                )
                await ws.send_json({"type": "result", **result})
    except WebSocketDisconnect:
        pass


def main() -> None:
    global _chunk_ms, _history_sec, _device, _config_name

    p = argparse.ArgumentParser(description="FSPEN web demo (FastAPI)")
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--config", default=_config_name)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--chunk-ms", type=float, default=500.0)
    p.add_argument("--history-sec", type=float, default=2.5)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    _chunk_ms = args.chunk_ms
    _history_sec = args.history_sec
    _device = args.device
    _config_name = args.config
    os.environ["FSPEN_CHECKPOINT"] = args.checkpoint
    os.environ["FSPEN_DEVICE"] = args.device
    os.environ["FSPEN_CONFIG"] = args.config

    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    main()
