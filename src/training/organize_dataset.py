"""Script that flattens the Kaggle BreakHis folder hierarchy into benign and
malignant directories expected by the training pipeline."""

import os
import shutil
from tqdm import tqdm

#Paths for raw dataset and processed output
RAW_DIR = r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\raw\breast-histopathology-images"
PROCESSED_DIR = r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\processed"

#Create target folders for the two classes
os.makedirs(os.path.join(PROCESSED_DIR, "benign"), exist_ok=True)
os.makedirs(os.path.join(PROCESSED_DIR, "malignant"), exist_ok=True)

#Iterate over patient-level folders
for patient_folder in os.listdir(RAW_DIR):
    patient_path = os.path.join(RAW_DIR, patient_folder)
    
    #Skip non-directories
    if not os.path.isdir(patient_path):
        continue 
    
    #The dataset encodes labels as "0" = benign, "1" = malignant
    for label in ["0", "1"]:
        label_path = os.path.join(patient_path, label)
        if not os.path.exists(label_path):
            continue
    #Process all PNG files under each label
        for file in tqdm(os.listdir(label_path), desc=f"Processing {patient_folder}/{label}"):
            if file.endswith(".png"):
                src_path = os.path.join(label_path, file)
                #Choose destination folder based on label
                if label == "0":
                    dst_dir = os.path.join(PROCESSED_DIR, "benign")
                else:
                    dst_dir = os.path.join(PROCESSED_DIR, "malignant")

                dst_path = os.path.join(dst_dir, f"{patient_folder}_{file}")
                shutil.copy(src_path, dst_path)
