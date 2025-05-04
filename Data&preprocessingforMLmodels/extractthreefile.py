import os
import zipfile
from io import BytesIO

source_dir = r'C:\Users\HP\Desktop\DaicWoiz'
target_dir = r'C:\Users\HP\Desktop\DataofML'

os.makedirs(target_dir, exist_ok=True)

for i in range(300, 493):
    folder_name = f"{i}_P"
    zip_path = os.path.join(source_dir, f"{folder_name}.zip")
    
    if not os.path.isfile(zip_path):
        print(f"Skipping missing archive: {zip_path}")
        continue

    with zipfile.ZipFile(zip_path, 'r') as zip_in:
        file_prefix = f"{i}"
        required_files = {
            f"{file_prefix}_CLNF_AUs.txt",
            f"{file_prefix}_COVAREP.csv",
            f"{file_prefix}_FORMANT.csv"
        }

        # Prepare output zip file path
        output_zip_path = os.path.join(target_dir, f"{folder_name}.zip")

        with zipfile.ZipFile(output_zip_path, 'w') as zip_out:
            for file_info in zip_in.infolist():
                if os.path.basename(file_info.filename) in required_files:
                    data = zip_in.read(file_info.filename)
                    zip_out.writestr(os.path.basename(file_info.filename), data)
                    print(f"Added {file_info.filename} to {output_zip_path}")
