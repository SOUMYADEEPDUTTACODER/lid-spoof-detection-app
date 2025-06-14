from fastapi import FastAPI, UploadFile, File
import os
from lid_model import LIDModel
from spoof_detector import SpoofDetector

# Initialize app and models
app = FastAPI(title="Multilingual LID + Spoof Detection API")
lid_model = LIDModel("./models/robust_lid_model.pkl")
spoof_model = SpoofDetector("./models/spoof_detector_mlp.pkl")

@app.get("/")
async def root():
    return {"message": "Welcome to the LID + Spoof Detection API!"}

@app.post("/predict/")
async def predict_audio(file: UploadFile = File(...)):
    temp_path = "temp.wav"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    try:
        lang = lid_model.predict(temp_path)
        spoofed = spoof_model.predict(temp_path)
        result = {
            "language": lang,
            "spoofed": spoofed
        }
    except Exception as e:
        result = {"error": str(e)}
    finally:
        os.remove(temp_path)

    return result