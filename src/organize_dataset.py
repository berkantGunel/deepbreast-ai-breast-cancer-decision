import os
import shutil
from tqdm import tqdm

RAW_DIR = r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\raw\breast-histopathology-images"
PROCESSED_DIR = r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\processed"

os.makedirs(os.path.join(PROCESSED_DIR, "benign"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "malignant"), exist_ok=True)

for patient_folder in os.listdir(RAW_DIR):
    patient_path = os.path.join(RAW_DIR, patient_folder)
    
    if not os.path.isdir(patient_path):
        continue 

    for label in ["0", "1"]:
        label_path = os.path.join(patient_path, label)
        if not os.path.exists(label_path):
            continue

        for file in tqdm(os.listdir(label_path), desc=f"Processing {patient_folder}/{label}"):
            if file.endswith(".png"):
                src_path = os.path.join(label_path, file)

                if label == "0":
                    dst_dir = os.path.join(PROCESSED_DIR, "benign")
                else:
                    dst_dir = os.path.join(PROCESSED_DIR, "malignant")

                dst_path = os.path.join(dst_dir, f"{patient_folder}_{file}")
                shutil.copy(src_path, dst_path)
