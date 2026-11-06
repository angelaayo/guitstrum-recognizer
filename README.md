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

Deployed on Render's free tier as a Docker web service (see `render.yaml`).
Render provides HTTPS/`wss://` automatically and injects `$PORT`, which the
Dockerfile binds. In Render: New → Blueprint → connect this repo → Apply.
