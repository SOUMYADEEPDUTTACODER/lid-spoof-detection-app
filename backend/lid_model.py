import joblib
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

class LIDModel:
    def __init__(self, model_path="models/robust_lid_model.pkl"):
        self.model = joblib.load(model_path)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53",force_download=True)
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
        self.languages = ["english", "hindi", "spanish", "tamil", "mandarin"]

    def extract_embedding(self, file_path):
        audio, _ = librosa.load(file_path, sr=16000)
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            features = self.wav2vec(**inputs).last_hidden_state
        return torch.mean(features, dim=1).squeeze().numpy().reshape(1, -1)

    def predict(self, file_path):
        emb = self.extract_embedding(file_path)
        lang_idx = self.model.predict(emb)[0]
        return self.languages[lang_idx]

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return acc, cm

