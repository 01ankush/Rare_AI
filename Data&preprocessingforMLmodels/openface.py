# import os
# import subprocess
# import pandas as pd
# import numpy as np

# def extract_action_units(video_path, openface_path):
#     """
#     Extract facial action units from a video file using OpenFace
    
#     Args:
#         video_path (str): Path to input video file
#         openface_path (str): Path to OpenFace FeatureExtraction executable
        
#     Returns:
#         pd.DataFrame: DataFrame containing the extracted AU features
#     """
#     # Prepare output directory
#     output_dir = os.path.dirname(video_path)
#     video_name = os.path.splitext(os.path.basename(video_path))[0]
#     output_csv = os.path.join(output_dir, f"final_au.csv")
    
#     # Command to run OpenFace FeatureExtraction
#     cmd = [
#         openface_path,
#         "-f", video_path,
#         "-out_dir", output_dir,
#         "-of", video_name,
#         "-aus"
#     ]
    
#     # Run OpenFace
#     try:
#         subprocess.run(cmd, check=True)
#         print("OpenFace processing completed successfully.")
#     except subprocess.CalledProcessError as e:
#         print(f"Error running OpenFace: {e}")
#         return None
    
#     # Check if output file was created
#     if not os.path.exists(output_csv):
#         print("Output CSV file not found.")
#         return None
    
#     # Load and process the output data
#     try:
#         # Read the OpenFace output
#         df = pd.read_csv(output_csv)
        
#         # Select the requested AU columns
#         requested_aus = [
#             'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
#             'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 
#             'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c',
#             'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c'
#         ]
        
#         # Check if all requested columns exist
#         available_aus = [col for col in requested_aus if col in df.columns]
#         missing_aus = set(requested_aus) - set(available_aus)
        
#         if missing_aus:
#             print(f"Warning: The following AUs were not found in the output: {missing_aus}")
        
#         # Select only the available requested columns
#         au_df = df[['frame', 'face_id', 'timestamp'] + available_aus].copy()
        
#         return au_df
    
#     except Exception as e:
#         print(f"Error processing output file: {e}")
#         return None

# # Example usage
# if __name__ == "__main__":
#     # Path to your video file
#     video_file = "sad_video.mp4"
    
#     # Path to OpenFace FeatureExtraction executable
#     # (typically located in OpenFace/bin/FeatureExtraction on Windows or OpenFace/build/bin/FeatureExtraction on Linux)
#     openface_exe = r"C:\Users\HP\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe"
    
#     # Extract action units
#     au_data = extract_action_units(video_file, openface_exe)
    
#     if au_data is not None:
#         # Display the first few rows
#         print(au_data.head())
        
#         # Save to CSV
#         output_path = "_aus_processed.csv"
#         au_data.to_csv(output_path, index=False)
#         print(f"Results saved to {output_path}")



import os
import subprocess
import pandas as pd
from pathlib import Path

def extract_action_units(video_path, openface_path):
    """
    Extract facial action units from a video file using OpenFace
    
    Args:
        video_path (str): Path to input video file
        openface_path (str): Path to OpenFace FeatureExtraction executable
        
    Returns:
        tuple: (full_features_df, requested_aus_df) containing all features and filtered AUs
    """
    # Convert to Path objects for better path handling
    video_path = Path(video_path)
    openface_path = Path(openface_path)
    output_dir = video_path.parent
    
    # OpenFace's output naming convention
    openface_output_csv = output_dir / f"{video_path.stem}.csv"
    all_features_output = output_dir / "all_features.csv"
    requested_aus_output = output_dir / "requested_aus.csv"
    
    # Command to run OpenFace
    cmd = [
        str(openface_path),
        "-f", str(video_path),
        "-out_dir", str(output_dir),
        "-of", video_path.stem,
        "-aus"
    ]
    
    print(f"Running OpenFace on {video_path.name}...")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("OpenFace Error:")
            print(result.stderr)
            return None, None
        
        print("OpenFace processing completed successfully.")
    except Exception as e:
        print(f"Error running OpenFace: {e}")
        return None, None
    
    # Check for output files (try both possible naming conventions)
    if not openface_output_csv.exists():
        openface_output_csv = output_dir / f"{video_path.stem}_features.csv"
        if not openface_output_csv.exists():
            print(f"Error: No output CSV found in {output_dir}")
            print("Files found:")
            for f in output_dir.glob(f"{video_path.stem}*"):
                print(f" - {f.name}")
            return None, None
    
    try:
        # 1. Save complete OpenFace output
        full_df = pd.read_csv(openface_output_csv)
        full_df.to_csv(all_features_output, index=False)
        print(f"Saved all features to {all_features_output}")
        
        # 2. Filter requested AUs
        requested_aus = [
            'AU01_r', 'AU02_r', 'AU04_r', 'AU05_r', 'AU06_r',
            'AU09_r', 'AU10_r', 'AU12_r', 'AU14_r', 'AU15_r', 
            'AU17_r', 'AU20_r', 'AU25_r', 'AU26_r', 'AU04_c',
            'AU12_c', 'AU15_c', 'AU23_c', 'AU28_c', 'AU45_c'
        ]
        
        # Find available columns
        available_columns = [col for col in requested_aus if col in full_df.columns]
        missing_columns = set(requested_aus) - set(available_columns)
        
        if missing_columns:
            print(f"Warning: Missing AUs: {missing_columns}")
        
        # Always keep these core columns
        core_columns = ['frame', 'face_id', 'timestamp', 'confidence', 'success']
        selected_columns = [col for col in core_columns if col in full_df.columns] + available_columns
        
        au_df = full_df[selected_columns].copy()
        au_df.to_csv(requested_aus_output, index=False)
        print(f"Saved requested AUs to {requested_aus_output}")
        
        return full_df, au_df
    
    except Exception as e:
        print(f"Error processing data: {e}")
        return None, None

if __name__ == "__main__":
    # Path configuration
    video_file = Path("sad_video.mp4")
    openface_exe = Path(r"C:\Users\HP\Desktop\OpenFace_2.2.0_win_x64\FeatureExtraction.exe")
    
    # Verify paths exist
    if not video_file.exists():
        print(f"Error: Video file not found at {video_file}")
        exit(1)
        
    if not openface_exe.exists():
        print(f"Error: OpenFace executable not found at {openface_exe}")
        exit(1)
    
    # Process video
    all_features, requested_aus = extract_action_units(video_file, openface_exe)
    
    if requested_aus is not None:
        print("\nFirst 5 rows of requested AUs:")
        print(requested_aus.head())