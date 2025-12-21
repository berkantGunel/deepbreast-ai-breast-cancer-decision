"""
Debug CBIS-DDSM - simple test
"""
import pandas as pd
from pathlib import Path
import os

CSV_DIR = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography\csv")
JPEG_DIR = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography\jpeg")

# Load CSV
df = pd.read_csv(CSV_DIR / "mass_case_description_train_set.csv")
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Get first cropped path
path = df['cropped_image_file_path'].iloc[0]
print("CSV cropped path:")
print(path)
print()

# List first 5 jpeg folders
print("First 5 JPEG folders:")
folders = sorted(os.listdir(JPEG_DIR))[:5]
for f in folders:
    print(f"  {f}")
    # List contents
    contents = os.listdir(JPEG_DIR / f)
    for c in contents:
        print(f"    -> {c}")
