from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import librosa
import numpy as np

app = FastAPI(title="Audio Feature Extraction API")

# -------------------------------
# Schema for response
# -------------------------------
class AudioFeatures(BaseModel):
    duration: float
    rms: float
    mfcc1: float
    mfcc2: float
    mfcc3: float
    mfcc4: float
    mfcc5: float
    mfcc6: float
    mfcc7: float
    mfcc8: float
    mfcc9: float
    mfcc10: float
    mfcc11: float
    mfcc12: float
    mfcc13: float


# -------------------------------
# Endpoint
# -------------------------------
@app.post("/extract_features", response_model=AudioFeatures)
async def extract_features(file: UploadFile = File(...)):
    # Load audio
    y, sr = librosa.load(file.file, sr=16000)

    # Duration
    duration = librosa.get_duration(y=y, sr=sr)

    # RMS
    rms = np.mean(librosa.feature.rms(y=y))

    # MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfcc, axis=1)

    # Prepare response
    features = {
        "duration": float(duration),
        "rms": float(rms),
        **{f"mfcc{i+1}": float(mfcc_means[i]) for i in range(13)}
    }

    return features
