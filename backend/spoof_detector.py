import os
import joblib
import torch
import librosa
import numpy as np
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix

class SpoofDetector:
    def __init__(self):
        base_path = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(base_path, "../models/spoof_detector_mlp.pkl")
        self.model = joblib.load(model_path)
        self.extractor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        self.wav2vec = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53").eval()

    def extract_embedding(self, file_path):
        audio, _ = librosa.load(file_path, sr=16000)
        inputs = self.extractor(audio, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            features = self.wav2vec(**inputs).last_hidden_state
        return torch.mean(features, dim=1).squeeze().numpy().reshape(1, -1)

    def predict(self, file_path):
        emb = self.extract_embedding(file_path)
        return bool(self.model.predict(emb)[0])  # True = Spoof, False = Real

    def evaluate(self, X, y):
        y_pred = self.model.predict(X)
        report = classification_report(y, y_pred, target_names=["Real", "Spoofed"], output_dict=True)
        cm = confusion_matrix(y, y_pred)
        return report, cm
