from fastapi import FastAPI, UploadFile, File
import os
import sys

# Update path to import from backend
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from lid_model import LIDModel
from spoof_detector import SpoofDetector

app = FastAPI(title="Multilingual LID + Spoof Detection API")

# Load models
lid_model = LIDModel()
spoof_model = SpoofDetector()

@app.get("/")
async def root():
    return {"message": "Welcome to the LID + Spoof Detection API!"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        language = lid_model.predict(temp_path)
        spoofed = spoof_model.predict(temp_path)
        result = {
            "language": language,
            "spoofed": spoofed
        }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        os.remove(temp_path)

    return result
