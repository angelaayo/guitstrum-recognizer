from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import onnxruntime as ort

TARGET_SR = 22050
N_SAMPLES = 43844  # Basic Pitch's fixed ~2s input window at 22050 Hz.
CAPTURE_SECONDS = 2.0  # Audio collected after a strum before predicting.
DETECTION_WINDOW_SECONDS = 0.5

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "ok"}

NOTE_MODEL = ort.InferenceSession("models/nmp.onnx", providers=["CPUExecutionProvider"])
NOTE_INPUT = NOTE_MODEL.get_inputs()[0].name

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_note_profile(y):
    # y is already 22050 Hz mono; shape to the model's fixed window.
    x = y[:N_SAMPLES] if len(y) >= N_SAMPLES else np.pad(y, (0, N_SAMPLES - len(y)))
    x = x.reshape(1, N_SAMPLES, 1).astype(np.float32)
    outputs = NOTE_MODEL.run(None, {NOTE_INPUT: x})
    note = next(o for o in outputs if o.shape[-1] == 88)[0]  # (frames, 88)
    return note.mean(axis=0)  # 88 values; bin i -> MIDI note 21+i (A0)


@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    await websocket.accept()
    try:
        browser_sr = int(websocket.query_params.get("sampleRate", 48000))
    except (TypeError, ValueError):
        browser_sr = 48000

    check_interval_samples = int(browser_sr * 0.05)
    print(f"Client connected — browser sample rate: {browser_sr} Hz")

    audio_buffer = np.array([], dtype=np.float32)
    samples_since_last_check = 0
    total_samples_received = 0
    cooldown_until = 0  # so we don't retrigger on the same strum's fade out

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, chunk])
            samples_since_last_check += len(chunk)
            total_samples_received += len(chunk)

            # cap the buffer so it doesn't grow forever
            max_buffer_samples = int(
                browser_sr * (CAPTURE_SECONDS + DETECTION_WINDOW_SECONDS + 1)
            )
            if len(audio_buffer) > max_buffer_samples:
                audio_buffer = audio_buffer[-max_buffer_samples:]

            if samples_since_last_check < check_interval_samples:
                continue
            samples_since_last_check = 0

            detection_window_samples = int(browser_sr * DETECTION_WINDOW_SECONDS)
            if len(audio_buffer) < detection_window_samples:
                continue

            recent = audio_buffer[-detection_window_samples:]

            recent_rms = np.sqrt(np.mean(recent ** 2))
            if recent_rms < 0.01:
                continue
            onset_env = librosa.onset.onset_strength(y=recent, sr=browser_sr)

            if(
                onset_env.max() > max(np.median(onset_env) * 5, 0.1)
                and total_samples_received >= cooldown_until
            ):
                print("Strum detected, capturing window with onset near the start...")

                onset_frame = int(np.argmax(onset_env))
                onset_in_recent = librosa.frames_to_samples(onset_frame)
                recent_start = len(audio_buffer) - len(recent)

                onset_position = recent_start + onset_in_recent
                pre_roll = int(browser_sr * 0.1)
                needed_total = int(browser_sr * CAPTURE_SECONDS)

                # keep receiving audio until we've accumulated a full window
                # that starts just before the detected onset
                while (len(audio_buffer) - (onset_position - pre_roll)) < needed_total:
                    data = await websocket.receive_bytes()
                    chunk = np.frombuffer(data, dtype=np.float32)
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                    total_samples_received += len(chunk)

                start = max(0, onset_position - pre_roll)
                window = audio_buffer[start : start + needed_total]

                y_resampled = librosa.resample(window, orig_sr=browser_sr, target_sr=TARGET_SR)

                print(f"Peak amplitude: {np.abs(y_resampled).max():.4f}, "
                      f"Clipped samples: {np.sum(np.abs(y_resampled) >= 0.99)}")

                profile = get_note_profile(y_resampled)
                top = np.argsort(profile)[::-1][:3]
                print("Top notes (midi, strength):",
                      [(21 + int(i), round(float(profile[i]), 3)) for i in top])

                await websocket.send_json({
                    "notes": [round(float(v), 4) for v in profile],
                })

                cooldown_until = total_samples_received + browser_sr * 3
                audio_buffer = audio_buffer[start + needed_total :]  # drop what we just used

            else:
                print(
                    f"No trigger — max: {onset_env.max():.2f}, "
                    f"median*3: {np.median(onset_env) * 5:.2f}, "
                    f"total received: {total_samples_received}, cooldown_until: {cooldown_until}"
                )
    except WebSocketDisconnect:
        print("Client disconnected")
