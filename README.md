---
title: GuitStrum Recognizer
emoji: 🎸
colorFrom: purple
colorTo: blue
sdk: docker
app_port: 8000
pinned: false
---

# GuitStrum Recognizer

Real-time guitar-chord recognition service for [GuitStrum](https://github.com/).

- FastAPI + WebSockets (`/ws/recognize`), served by uvicorn on port 8000.
- Streams raw browser audio, detects strum onsets, extracts mel-spectrogram
  features, and classifies the chord with a trained SVM (`chord_model.pkl`).
- `GET /health` returns a liveness check.

Deployed as a Docker Space. The `.pkl` model files are stored via Git LFS.
