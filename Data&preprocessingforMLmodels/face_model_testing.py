import sys
import json
import torch
import torch.nn as nn
import numpy as np
import joblib
import pandas as pd
from pathlib import Path

# Configuration
REQUESTED_AUS = [
    'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
    'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r',
    'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c',
    'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c'
]

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

def prepare_features_from_openface(all_features_path):
    """Extract and prepare AU features from OpenFace output"""
    try:
        # Load OpenFace output
        df = pd.read_csv(all_features_path)
        
        # Verify all requested AUs exist
        missing_aus = set(REQUESTED_AUS) - set(df.columns)
        if missing_aus:
            raise ValueError(f"Missing AUs in data: {missing_aus}")
        
        # Get most recent frame with good confidence
        valid_frame = df[df['confidence'] > 0.9].iloc[-1] if 'confidence' in df else df.iloc[-1]
        au_features = valid_frame[REQUESTED_AUS].values.astype(float)
        
        return au_features
    
    except Exception as e:
        print(f"Error processing features: {e}")
        return None

def predict_emotion(au_features, model_path="models/model_simplelstm.pth", 
                   scaler_path="models/scaler.pkl"):
    """Run prediction on prepared AU features"""
    try:
        # Load model assets
        scaler = joblib.load(scaler_path)
        input_size = len(REQUESTED_AUS)
        
        # Initialize model
        model = SimpleLSTM(input_size=input_size)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Preprocess features
        au_scaled = scaler.transform(au_features.reshape(1, -1))
        x_tensor = torch.tensor(au_scaled, dtype=torch.float32).unsqueeze(2)
        
        # Predict
        with torch.no_grad():
            output = model(x_tensor)
            prediction = torch.argmax(output, dim=1).item()
        
        return prediction
    
    except Exception as e:
        print(f"Prediction error: {e}")
        return None

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <path_to_openface_output.csv>")
        sys.exit(1)
    
    # 1. Prepare features
    input_path = "all_features.csv"
    au_features = prepare_features_from_openface(input_path)
    
    if au_features is None:
        sys.exit(1)
    
    print("Extracted AU features:")
    print(dict(zip(REQUESTED_AUS, au_features)))
    
    # 2. Run prediction
    prediction = predict_emotion(au_features)
    
    if prediction is not None:
        print(f"\nPrediction result: {prediction}")
        # For your voice_from_wav.py integration:
        # print(prediction)  # Uncomment for pure numeric output