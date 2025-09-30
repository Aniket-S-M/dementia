from fastapi import FastAPI ,UploadFile, File
from pydantic import BaseModel, Field
from typing import Optional
import librosa
import numpy as np
from prediction import predict_audio, predict_cognitive, predict_combined
from fastapi.middleware.cors import CORSMiddleware




app = FastAPI(title="Dementia Prediction API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------
# Schemas
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




class CognitiveData(BaseModel):
    age: int
    gender: str
    education_years: int
    memory_score: float
    attention_score: float
    language_score: float
    executive_score: float
    visuospatial_score: float
    reaction_time_ms: float
    mood_score: int


class CombinedInput(BaseModel):
    audio: Optional[AudioFeatures] = None
    cognitive: Optional[CognitiveData] = None


# -------------------------------
# API Endpoints
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

@app.post("/predict")
def predict(input_data: CombinedInput):
    if input_data.audio and input_data.cognitive:
        prediction = predict_combined(input_data.audio.dict(), input_data.cognitive.dict())
        source = "combined"
    elif input_data.audio:
        prediction = predict_audio(input_data.audio.dict())
        source = "audio"
    elif input_data.cognitive:
        prediction = predict_cognitive(input_data.cognitive.dict())
        source = "cognitive"
    else:
        return {"error": "No valid input provided"}

    return {"prediction": prediction, "source": source}


# -------------------------------
# Run server
# -------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apiendpoint:app", host="0.0.0.0", port=8000, reload=True)
