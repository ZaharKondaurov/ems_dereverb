# Hugging Face Space

Create a Space with **Docker** SDK (Gradio is not used for the live spectrogram UI).

## Space README (frontmatter)

```yaml
---
title: FSPEN Live
emoji: 🎤
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---
```

## Checkpoints

Place these files under `checkpoints/fspen_chkp/`:

| Preset | File |
|--------|------|
| FSPEN+48kHz | `TrainConfig_48khz_baseline.pt` |
| FSPEN+48kHz+overlap | `TrainConfig_48kHz_overlap.pt` |
| FSPEN+48kHz+SBLE | `TrainConfig_48kHz_enc_ext.pt` |
| FSPEN+48kHz+SBDC+overlap | `TrainConfig_48kHz_enc_ext_lay_1_overlap.pt` |

## Environment

| Variable | Default |
|----------|---------|
| `FSPEN_PRESET` | `fspen_48khz_overlap` |
| `FSPEN_DEVICE` | `cpu` |
| `FSPEN_CHUNK_MS` | `500` |

## Local test

```bash
pip install -r requirements-web.txt
pip install torch torchaudio
python web_app.py --preset fspen_48khz_overlap --chunk-ms 500
```

Open http://127.0.0.1:7860

## UI

- **Model** dropdown — four presets (config + checkpoint + `model_eval` / `model_eval_old`)
- **Live** — mic, spectrograms, Enhanced/Bypass, Monitor on/off, **RTF** in status
- **File** — upload, process, download WAV, **RTF** after processing
