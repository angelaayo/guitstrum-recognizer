from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import io
import soundfile as sf
import time

TARGET_SR = 22050
# BROWSER_SR = 44100 # Must match audioContext.sampleRate in the browser.
MODEL_WINDOW_SECONDS = 5  # Fixed: the SVC was trained on five-second features.
CAPTURE_SECONDS = 2.0  # Audio collected after a strum before predicting.
DETECTION_WINDOW_SECONDS = 0.5
# CHECK_INTERVAL_SAMPLES = int(BROWSER_SR * 0.05)  # Check every 50 ms.
CONFIDENCE_FLOOR = 0.55  # dont report a guess under this accuracy percent

app = FastAPI()

model = joblib.load("chord_model.pkl")
encoder = joblib.load("chord_encoder.pkl")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def getChord(y, sr):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    features = S_db_mel.reshape(1, -1)

    probs = model.predict_proba(features)[0]
    best_idx = np.argmax(probs)
    chord = encoder.classes_[best_idx]
    confidence = float(probs[best_idx])

    return chord, confidence



@app.post("/recognize")
async def recognize(audio: UploadFile = File(...)):
    #ensuring our audio is processed in the right format
    raw_bytes = await audio.read()
    audio_segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
    wav_buffer = io.BytesIO()
    audio_segment.export(wav_buffer, format="wav")
    wav_buffer.seek(0)

    #based on our training on mel spectograms all our audio files need to be
    #the same length of 5 seconds

    target_duaration = 5
    y,sr = librosa.load(wav_buffer)
    #trim off silence in audio
    print(f"Raw audio: {len(y)/sr:.2f}s, sample rate: {sr}")
    y_trimmed, _ = librosa.effects.trim(y, top_db=34)
    print(f"After trim: {len(y_trimmed)/sr:.2f}s")

    target_samples = target_duaration * sr
    if len(y_trimmed) > target_samples:
        y_fixed = y_trimmed[:target_samples]
    else:
        padding = target_samples - len(y_trimmed)
        y_fixed = np.pad(y_trimmed, (0, padding), mode="constant")
    # import soundfile as sf
    # sf.write("debug_final.wav", y_trimmed, sr)
    chord = getChord(y_fixed, sr)

    return {"detectedChord": chord}


#overall process that got things working:
# Live mic at 48 kHz
# → ignore quiet room noise
# → detect real strum
# → cut a five-second window around that strum
# → resample correctly to 22.05 kHz
# → compute the same Mel spectrogram features used in training
# → predict chord
@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    await websocket.accept()

    try:
        browser_sr = int(websocket.query_params.get("sampleRate", 44100))
    except (TypeError, ValueError):
        browser_sr = 44100

    check_interval_samples = int(browser_sr * 0.05)  # recomputed per connection now

    print(f"Client connected — browser sample rate: {browser_sr}Hz")

    audio_buffer = np.array([], dtype=np.float32)
    samples_since_last_check = 0
    total_samples_received = 0
    cooldown_until = 0

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer = np.concatenate([audio_buffer, chunk])
            samples_since_last_check += len(chunk)
            total_samples_received += len(chunk)

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

            if (
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

                while (len(audio_buffer) - (onset_position - pre_roll)) < needed_total:
                    data = await websocket.receive_bytes()
                    chunk = np.frombuffer(data, dtype=np.float32)
                    audio_buffer = np.concatenate([audio_buffer, chunk])
                    total_samples_received += len(chunk)

                start = max(0, onset_position - pre_roll)
                window = audio_buffer[start : start + needed_total]

                y_resampled = librosa.resample(window, orig_sr=browser_sr, target_sr=TARGET_SR)

                target_samples = MODEL_WINDOW_SECONDS * TARGET_SR
                if len(y_resampled) > target_samples:
                    y_fixed = y_resampled[:target_samples]
                else:
                    padding = max(0, target_samples - len(y_resampled))
                    y_fixed = np.pad(y_resampled, (0, padding), mode="constant")

                print(f"Peak amplitude: {np.abs(y_fixed).max():.4f}, "
                      f"Clipped samples: {np.sum(np.abs(y_fixed) >= 0.99)}")

                chord, confidence = getChord(y_fixed, TARGET_SR)
                print(f"Predicted: {chord} ({confidence:.2%} confidence)")

                if confidence >= CONFIDENCE_FLOOR:
                    await websocket.send_json({"detectedChord": chord, "confidence": confidence})
                else:
                    print("Below confidence floor - treating as noise, not sending")

                cooldown_until = total_samples_received + browser_sr * 3
                audio_buffer = audio_buffer[start + needed_total :]

            else:
                print(
                    f"No trigger — max: {onset_env.max():.2f}, "
                    f"median*5: {np.median(onset_env) * 5:.2f}, "
                    f"total received: {total_samples_received}, cooldown_until: {cooldown_until}"
                )
    except WebSocketDisconnect:
        print("Client disconnected")