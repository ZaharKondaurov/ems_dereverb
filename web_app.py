#!/usr/bin/env python3
"""
FSPEN live web demo (FastAPI + WebSocket + canvas spectrograms).

Local:
  pip install -r requirements-web.txt
  python web_app.py --preset fspen_48khz_overlap --chunk-ms 500

Hugging Face Space: see README_SPACE.md (Docker sdk, port 7860).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from src.streaming import StreamingEnhancer, load_enhancer  # noqa: E402
from src.web_models import (  # noqa: E402
    DEFAULT_PRESET_ID,
    enhance_wav_bytes,
    get_preset,
    list_presets,
)
from src.ws_session import WebStreamSession  # noqa: E402

STATIC_DIR = BASE_DIR / "static"
# Bump when static UI changes (cache-bust query string in index.html).
UI_ASSET_VERSION = "3"
_NO_CACHE_HEADERS = {
    "Cache-Control": "no-store, no-cache, must-revalidate",
    "Pragma": "no-cache",
}


class NoCacheMiddleware(BaseHTTPMiddleware):
    """Avoid stale index.html / app.js in the browser during development."""

    async def dispatch(self, request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path == "/" or path.startswith("/static/"):
            for k, v in _NO_CACHE_HEADERS.items():
                response.headers[k] = v
        return response


_enhancer: Optional[StreamingEnhancer] = None
_configs = None
_sr = 48_000
_chunk_ms = 500.0
_history_sec = 2.5
_device = "cpu"
_preset_id = DEFAULT_PRESET_ID
_config_name = ""
_checkpoint_path = ""

_model_lock = threading.Lock()


class ModelSwitchRequest(BaseModel):
    preset_id: str
    chunk_ms: float = Field(default=500.0, gt=0, le=5000)


def _load_model(
    checkpoint: str,
    device: str,
    config_name: str,
    chunk_ms: float,
    *,
    preset_id: str,
) -> None:
    global _enhancer, _configs, _sr, _chunk_ms, _config_name, _checkpoint_path, _device, _preset_id

    ckpt = str(Path(checkpoint).expanduser())
    if not os.path.isfile(ckpt):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt}")

    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    preset = get_preset(preset_id)
    if preset.config != config_name:
        raise ValueError(
            f"Preset {preset_id} expects config {preset.config}, got {config_name}"
        )

    # torch.set_num_threads(1)
    # torch.set_num_interop_threads(1)
    print(f"Loading [{preset.label}] {ckpt} on {device} …")
    model, configs = load_enhancer(ckpt, device=device, config_name=config_name)
    chunk_samples = max(
        int(configs.sample_rate * chunk_ms / 1000), configs.hop_length * 4
    )

    _device = device
    _configs = configs
    _sr = configs.sample_rate
    _chunk_ms = chunk_ms
    _config_name = config_name
    _checkpoint_path = ckpt
    _preset_id = preset_id
    _enhancer = StreamingEnhancer(
        model, configs, device=device, chunk_samples=chunk_samples
    )
    print(f"Model ready ({preset.model_class}, {preset.eval_fn}).")


def _load_preset(preset_id: str, device: str, chunk_ms: float) -> None:
    preset = get_preset(preset_id)
    _load_model(
        str(preset.checkpoint_path(BASE_DIR)),
        device,
        preset.config,
        chunk_ms,
        preset_id=preset_id,
    )


def _new_session(enhanced: bool = True) -> WebStreamSession:
    assert _enhancer is not None and _configs is not None
    s = WebStreamSession(
        _enhancer,
        history_sec=_history_sec,
        n_fft=_configs.n_fft,
        hop_length=_configs.hop_length,
    )
    s.set_enhanced(enhanced)
    return s


def _current_rtf() -> float:
    if _enhancer is None:
        return 0.0
    return round(float(_enhancer.mean_rtf), 3)


def _current_model_info() -> dict[str, Any]:
    preset = get_preset(_preset_id)
    return {
        "preset_id": _preset_id,
        "preset_label": preset.label,
        "config": _config_name,
        "checkpoint": _checkpoint_path,
        "checkpoint_name": Path(_checkpoint_path).name,
        "model_class": preset.model_class,
        "eval_fn": preset.eval_fn,
        "chunk_ms": _chunk_ms,
        "sample_rate": _sr,
        "device": _device,
        "rtf": _current_rtf(),
    }


def _catalog_payload() -> dict[str, Any]:
    return {
        "presets": list_presets(BASE_DIR),
        "current": _current_model_info(),
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    preset_id = os.environ.get("FSPEN_PRESET", DEFAULT_PRESET_ID)
    dev = os.environ.get("FSPEN_DEVICE", _device)
    chunk = float(os.environ.get("FSPEN_CHUNK_MS", str(_chunk_ms)))
    with _model_lock:
        _load_preset(preset_id, dev, chunk)
    yield


app = FastAPI(title="FSPEN Live", lifespan=lifespan)
app.add_middleware(NoCacheMiddleware)


@app.get("/")
async def index():
    return FileResponse(
        STATIC_DIR / "index.html",
        headers=_NO_CACHE_HEADERS,
    )


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse(
        STATIC_DIR / "favicon.svg",
        media_type="image/svg+xml",
    )


@app.get("/api/catalog")
async def api_catalog():
    return JSONResponse(_catalog_payload())


@app.get("/api/model")
async def api_model():
    return JSONResponse(_current_model_info())


@app.post("/api/model")
async def api_set_model(body: ModelSwitchRequest):
    try:
        preset = get_preset(body.preset_id)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e

    ckpt = preset.checkpoint_path(BASE_DIR)
    if not ckpt.is_file():
        raise HTTPException(404, f"Checkpoint not found: {ckpt}")

    try:
        with _model_lock:
            await asyncio.to_thread(
                _load_preset,
                body.preset_id,
                _device,
                body.chunk_ms,
            )
    except FileNotFoundError as e:
        raise HTTPException(404, str(e)) from e
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    except Exception as e:
        raise HTTPException(500, str(e)) from e

    return JSONResponse({"ok": True, "current": _current_model_info()})


@app.post("/api/process")
async def api_process_file(
    file: UploadFile = File(...),
    chunked: bool = Query(False),
):
    if _enhancer is None or _configs is None:
        raise HTTPException(503, "Model not loaded")

    raw = await file.read()
    if not raw:
        raise HTTPException(400, "Empty file")

    name = file.filename or "upload.wav"

    def _run() -> tuple[bytes, dict[str, Any]]:
        with _model_lock:
            return enhance_wav_bytes(
                _enhancer,
                file_bytes=raw,
                filename=name,
                sample_rate=_sr,
                chunked=chunked,
            )

    try:
        wav_bytes, meta = await asyncio.to_thread(_run)
    except Exception as e:
        raise HTTPException(500, f"Processing failed: {e}") from e

    out_name = Path(name).stem + "_enhanced.wav"
    headers = {
        "Content-Disposition": f'attachment; filename="{out_name}"',
        "X-FSPEN-Meta": json.dumps(meta),
    }
    return Response(content=wav_bytes, media_type="audio/wav", headers=headers)


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
                if _enhancer is None:
                    await ws.send_json({"type": "error", "message": "Model not loaded"})
                    continue
                samples = msg.get("data", [])
                y = np.asarray(samples, dtype=np.float32)
                input_sr = int(msg.get("sr", _sr))

                def _run() -> dict:
                    with _model_lock:
                        return session.process(
                            y,
                            enhanced=session.enhanced,
                            input_sr=input_sr,
                        )

                result = await asyncio.to_thread(_run)
                await ws.send_json({"type": "result", **result})
    except WebSocketDisconnect:
        pass


def main() -> None:
    global _history_sec, _device, _chunk_ms

    p = argparse.ArgumentParser(description="FSPEN web demo (FastAPI)")
    p.add_argument(
        "--preset",
        default=DEFAULT_PRESET_ID,
        help="Model preset id (see /api/catalog)",
    )
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--chunk-ms", type=float, default=512.0)
    p.add_argument("--history-sec", type=float, default=2.5)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=7860)
    args = p.parse_args()

    _history_sec = args.history_sec
    _device = args.device
    _chunk_ms = args.chunk_ms

    os.environ["FSPEN_PRESET"] = args.preset
    os.environ["FSPEN_DEVICE"] = args.device
    os.environ["FSPEN_CHUNK_MS"] = str(args.chunk_ms)

    import uvicorn

    uvicorn.run(
        "web_app:app",
        host=args.host,
        port=args.port,
        reload=False,
    )


if __name__ == "__main__":
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    try:
        torch._logging.set_logs(graph_code=False)
    except Exception:
        pass
    main()
