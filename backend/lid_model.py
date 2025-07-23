import os
import joblib
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.metrics import accuracy_score, confusion_matrix

class LIDModel:
    def __init__(self, model_path=None):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = model_path or os.path.join(base_path, "../models/robust_lid_model.pkl")

        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"LID model not found at: {model_path}")

        self.model = joblib.load(model_path)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            "facebook/wav2vec2-large-xlsr-53")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()
        self.languages = ["english", "hindi", "spanish", "tamil", "mandarin"]

    def extract_embedding(self, file_path):
        audio, sr = librosa.load(file_path, sr=16000)
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            features = self.wav2vec(**inputs).last_hidden_state

        mean_embedding = features.mean(dim=1).squeeze(0).numpy()
        return mean_embedding.reshape(1, -1)

    def predict(self, file_path):
        embedding = self.extract_embedding(file_path)
        lang_index = self.model.predict(embedding)[0]
        return self.languages[lang_index]

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        return acc, cm
