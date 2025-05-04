import os
import subprocess
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import joblib
from pathlib import Path


# Define your model architecture
class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_classes=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=False)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, F, T] -> [B, T, F]
        out, _ = self.lstm(x)
        out = self.classifier(out[:, -1])  # Last time step
        return out


def extract_action_units(video_path, openface_path):
    from datetime import datetime
    from pathlib import Path
    import subprocess
    import pandas as pd

    video_path = Path(video_path)
    output_dir = Path("csvfiles")
    output_dir.mkdir(exist_ok=True)

    # Generate a unique output file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"{video_path.stem}_{timestamp}"
    csv_path = output_dir / f"{base_name}.csv"

    # Call OpenFace FeatureExtraction
    cmd = [
        str(openface_path),
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-of", base_name,
        "-aus"
    ]
    subprocess.run(cmd, capture_output=True, text=True)

    # Read the resulting CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    # Define columns to drop
    drop_columns = [
        'frame', 'face_id', 'timestamp', 'confidence', 'success',
        'AU07_r', 'AU23_r', 'AU45_r', 'AU01_c', 'AU02_c', 'AU05_c', 'AU06_c', 'AU07_c',
        'AU09_c', 'AU10_c', 'AU14_c', 'AU17_c', 'AU20_c', 'AU25_c', 'AU26_c'
    ]

    # Drop only columns that exist
    drop_columns_present = [col for col in drop_columns if col in df.columns]
    df.drop(columns=drop_columns, inplace=True)

    df_filtered_path = output_dir / f"{base_name}_aus.csv"
    df.to_csv(df_filtered_path, index=False)

    return df



# Predict using the trained model
def predict_from_aus(au_row, model_path, scaler_path):
    input_size = 20
    model = SimpleLSTM(input_size=input_size)
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cpu")))
    model.eval()

    scaler = joblib.load(scaler_path)
    au_scaled = scaler.transform(au_row.values.reshape(1, -1))
    tensor_input = torch.tensor(au_scaled, dtype=torch.float32).unsqueeze(2)

    with torch.no_grad():
        output = model(tensor_input)
        prediction = torch.argmax(output, dim=1).item()
    return prediction
