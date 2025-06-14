Multilingual LID + Spoof Detection System

A production-ready FastAPI + HTML/JS project for:
- 🔤 Spoken **Language Identification** (LID)
- 🔐 **Replay spoof detection** (genuine vs. replayed speech)
- 🌐 Clean web frontend for uploading `.wav` files

---

## 📁 Project Structure

project-root/
├── backend/
│ ├── app.py # FastAPI backend
│ ├── lid_model.py # Wav2Vec2-based LID logic
│ ├── spoof_detector.py # Binary spoof detection logic
│ ├── models/
│ │ ├── robust_lid_model.pkl
│ │ └── spoof_detector_mlp.pkl
│ └── requirements.txt
├── frontend/
│ ├── index.html # Frontend UI
│ ├── style.css
│ └── script.js
└── README.md

yaml
Copy
Edit

---

## 🚀 How to Run

### ✅ 1. Set up backend

```bash
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
🔗 Open: http://127.0.0.1:8000/docs to test the API

✅ 2. Run frontend
Open frontend/index.html in any browser (no server needed).

🎯 Features
Multilingual LID: English, Hindi, Spanish, Tamil, Mandarin

Spoof detection using audio augmentation and reverb simulation

Wav2Vec2 embeddings from Hugging Face

Clean UI for audio upload

RESTful JSON API with /predict/ route

🧪 Sample Output
json
Copy
Edit
{
  "language": "tamil",
  "spoofed": false
}

Models
You must include the following pre-trained models in backend/models/:

robust_lid_model.pkl

spoof_detector_mlp.pkl

🌐 Deployment Options
 Hugging Face Spaces (Gradio frontend)

 Render / Railway (backend)

 GitHub Pages (frontend)

 License
MIT © 2025 YourName