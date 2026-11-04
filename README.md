# GuitStrum Recognizer

Real-time guitar-chord recognition service for GuitStrum.

- FastAPI + WebSockets (`/ws/recognize`), served by uvicorn on port 8000.
- Streams raw browser audio, detects strum onsets, extracts mel-spectrogram
  features, and classifies the chord with a trained SVM (`chord_model.pkl`).
- `GET /health` returns a liveness check.

The `.pkl` model files are stored via Git LFS — install `git-lfs` before cloning.

## Run locally

```bash
docker build -t guitstrum-recognizer .
docker run -p 8000:8000 guitstrum-recognizer
# or: uvicorn main:app --reload
```

## Deploy (production, free)

Runs on a free Oracle Cloud "Always Free" VM behind Caddy, which provides
automatic HTTPS so the browser can reach it over `wss://`. See
[DEPLOY.md](DEPLOY.md) for the full runbook.
