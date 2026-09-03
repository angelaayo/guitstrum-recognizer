from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect
from pydub import AudioSegment
from fastapi.middleware.cors import CORSMiddleware
import librosa
import numpy as np
import joblib
import io

TARGET_SR = 22050
BROWSER_SR = 44100
BUFFER_SECONDS = 5

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
    S = librosa.feature.melspectogram(y=y, sr=sr, n_mels=128)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    features = S_db_mel.reshape(1, -1)
    prediction = model.predict(features)
    chord = encoder.inverse_transform(prediction)[0]

    return chord



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
    S = librosa.feature.melspectrogram(y=y_fixed, sr=sr, n_mels=128)
    S_db_mel = librosa.amplitude_to_db(S, ref=np.max)
    features = S_db_mel.reshape(1, -1)
    prediction = model.predict(features)
    chord = encoder.inverse_transform(prediction)[0]

    return {"detectedChord": chord}

@app.websocket("/ws/recognize")
async def ws_recognize(websocket: WebSocket):
    await websocket.accept()
    print("Client connected")

    audio_buffer = np.array([], dtype=np.float32)

    try:
        while True:
            data = await websocket.receive_bytes()
            chunk = np.frombuffer(data, dtype=np.float32)
            audio_buffer - np.concatenate([audio_buffer, chunk])

            #trying to aquire out 5 second window

            if len(audio_buffer) >= BROWSER_SR * BUFFER_SECONDS:
                window = audio_buffer[: BROWSER_SR * BUFFER_SECONDS]
                audio_buffer = audio_buffer[BROWSER_SR * BUFFER_SECONDS:]
                try:
                    y_resampled = librosa.resample(window, orig_sr=BROWSER_SR, target_sr=TARGET_SR)
                    target_samples = BUFFER_SECONDS * TARGET_SR
                    if len(y_resampled) > target_samples:
                        y_fixed = y_resampled[:target_samples]
                    else:
                        padding = max(0, target_samples - len(y_resampled))
                        y_fixed = np.pad(y_resampled, (0, padding), mode="constant")
                    chord  = getChord(y_fixed, TARGET_SR)
                    print(f"Predicted: {chord}")
                    await websocket.send_json({"detectedChord": chord})
                except Exception as e:
                    print(f"Prediction error: {e}")
    except WebSocketDisconnect:
        print("Client disconnected")