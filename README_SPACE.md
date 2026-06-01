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

## Files

- `Dockerfile` — builds and runs `web_app.py` on port **7860**
- Put the checkpoint under `checkpoints/fspen_chkp/` (Git LFS) or set `FSPEN_CHECKPOINT` to a downloaded path in the Dockerfile

## Environment

| Variable | Default |
|----------|---------|
| `FSPEN_CHECKPOINT` | `checkpoints/fspen_chkp/TrainConfig_48kHz_overlap_1986#0.pt` |
| `FSPEN_DEVICE` | `cpu` |
| `FSPEN_CONFIG` | `TrainConfig_48kHz_overlap` |

## Local test (same as Space)

```bash
pip install -r requirements-web.txt
pip install torch torchaudio
python web_app.py --config TrainConfig_48kHz_overlap \\
  --checkpoint checkpoints/fspen_chkp/TrainConfig_48kHz_overlap_1986#0.pt --chunk-ms 500
```

Open http://127.0.0.1:7860
