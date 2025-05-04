# models/voice_from_wav.py

import sys
import json
import torch
import torch.nn as nn
import numpy as np
import joblib

WAV_PATH = "audio_file"

# STEP 1: Extract AU features from WAV
# You must replace this block with your real feature extraction code
def extract_features(wav_path):
    # Dummy example – replace with py-feat/OpenFace or your extractor
    # Example: return [20 AU feature floats]
    return [
        2.84338, 1.09432, 0.758829, 0.647394, 0.680064, 0.901901, 0.789843, 0, 0, 0,
        0.829708, 0, 0, 0, 0, 0, 0, 1, 1, 0
    ]

raw_input = extract_features(WAV_PATH)

# STEP 2: Your model
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1])
        return out

scaler = joblib.load("scaler.pkl")
input_size = 20
model = SimpleLSTM(input_size=input_size)
model.load_state_dict(torch.load("model_simplelstm.pth", map_location=torch.device("cpu")))
model.eval()

# Preprocess and predict
au_features = np.array(raw_input).reshape(1, -1)
au_scaled = scaler.transform(au_features)
x_tensor = torch.tensor(au_scaled, dtype=torch.float32).unsqueeze(2)

with torch.no_grad():
    output = model(x_tensor)
    prediction = torch.argmax(output, dim=1).item()

print(prediction)
