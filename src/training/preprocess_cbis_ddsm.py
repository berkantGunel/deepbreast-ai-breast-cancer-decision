"""
CBIS-DDSM Preprocessing v3 - Using meta.csv for UID matching
"""
import pandas as pd
from pathlib import Path
import shutil
import random
from tqdm import tqdm
from collections import defaultdict

CSV_DIR = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography\csv")
JPEG_DIR = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography\jpeg")
OUTPUT_DIR = Path(r"C:\Users\MSI\Python\BreastCancerPrediction_BCP\data\mammography\processed")

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15

BIRADS_TO_CLASS = {2: 'benign', 3: 'benign', 4: 'suspicious', 5: 'malignant'}

def load_metadata():
    """Load case descriptions and meta.csv"""
    print("Loading metadata...")
    
    # Load case descriptions
    dfs = []
    for csv_file in ['mass_case_description_train_set.csv',
                     'mass_case_description_test_set.csv',
                     'calc_case_description_train_set.csv',
                     'calc_case_description_test_set.csv']:
        df = pd.read_csv(CSV_DIR / csv_file)
        df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()
        dfs.append(df)
    
    cases = pd.concat(dfs, ignore_index=True)
    print(f"  Case descriptions: {len(cases)}")
    
    # Load meta.csv
    meta = pd.read_csv(CSV_DIR / "meta.csv")
    print(f"  Meta entries: {len(meta)}")
    
    return cases, meta


def get_jpeg_folders():
    """Get all JPEG folder names as a set."""
    folders = set()
    for f in JPEG_DIR.iterdir():
        if f.is_dir() and f.name.startswith('1.3.6.1'):
            folders.add(f.name)
    print(f"  JPEG folders: {len(folders)}")
    return folders


def extract_patient_info(row):
    """Extract patient ID and view info from row."""
    patient_id = row.get('patient_id', '')
    side = row.get('left_or_right_breast', '')
    view = row.get('image_view', '')
    return f"{patient_id}_{side}_{view}".upper()


def main():
    print("="*60)
    print("CBIS-DDSM Preprocessing v3")
    print("="*60)
    
    # Create output directories
    for split in ['train', 'val', 'test']:
        for cls in ['benign', 'suspicious', 'malignant']:
            (OUTPUT_DIR / split / cls).mkdir(parents=True, exist_ok=True)
    
    # Load data
    cases, meta = load_metadata()
    jpeg_folders = get_jpeg_folders()
    
    # Filter cases by valid BI-RADS
    cases = cases[cases['assessment'].isin([2, 3, 4, 5])].copy()
    cases['class'] = cases['assessment'].map(BIRADS_TO_CLASS)
    print(f"\n  Valid cases: {len(cases)}")
    
    # Build lookup: SeriesInstanceUID -> folder path
    uid_to_folder = {}
    for folder_name in jpeg_folders:
        uid_to_folder[folder_name] = JPEG_DIR / folder_name
    
    # Filter meta for ROI/cropped images
    meta_cropped = meta[meta['SeriesDescription'].str.contains('cropped', case=False, na=False)]
    print(f"  Cropped series in meta: {len(meta_cropped)}")
    
    # Try to match using patient ID patterns
    print("\nMatching images using patient IDs...")
    
    class_images = defaultdict(list)
    matched = 0
    unmatched = 0
    
    for idx, row in tqdm(cases.iterrows(), total=len(cases)):
        cls = row['class']
        patient_id = row.get('patient_id', '')
        
        # Get the cropped image path from case description
        crop_path = row.get('cropped_image_file_path', '')
        if pd.isna(crop_path):
            unmatched += 1
            continue
        
        # Extract UIDs from the path - there are usually 2 UIDs in the path
        parts = str(crop_path).replace('\\', '/').split('/')
        uids = [p for p in parts if p.startswith('1.3.6.1.4.1.9590')]
        
        found = False
        for uid in uids:
            if uid in uid_to_folder:
                folder = uid_to_folder[uid]
                # Get smallest image (ROI crop)
                images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
                if images:
                    img = min(images, key=lambda x: x.stat().st_size)
                    class_images[cls].append(img)
                    matched += 1
                    found = True
                    break
        
        if not found:
            unmatched += 1
    
    print(f"\n  Matched: {matched}")
    print(f"  Unmatched: {unmatched}")
    
    # If matching still fails, use alternative approach
    if matched < 100:
        print("\n⚠️ Low match rate, trying alternative approach...")
        print("   Matching JPEG folders to patient IDs in meta.csv...")
        
        class_images = defaultdict(list)
        
        # Build patient_id -> class mapping
        patient_class = {}
        for _, row in cases.iterrows():
            pid = row['patient_id']
            cls = row['class']
            patient_class[pid] = cls
        
        # Find cropped images from meta
        for _, meta_row in tqdm(meta_cropped.iterrows(), total=len(meta_cropped)):
            uid = meta_row['SeriesInstanceUID']
            
            if uid in uid_to_folder:
                folder = uid_to_folder[uid]
                images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
                
                if images:
                    # Try to match to patient
                    for pid, cls in patient_class.items():
                        # Check if patient ID appears in any metadata
                        if pid in str(meta_row.values):
                            img = min(images, key=lambda x: x.stat().st_size)
                            class_images[cls].append(img)
                            break
    
    # Print class distribution
    print("\nClass distribution:")
    for cls in ['benign', 'suspicious', 'malignant']:
        print(f"  {cls}: {len(class_images[cls])}")
    
    total_images = sum(len(v) for v in class_images.values())
    
    if total_images == 0:
        print("\n❌ No images matched! Using fallback method...")
        # Fallback: just use all images and split by folder size pattern
        print("   Collecting all ROI crops from JPEG folders...")
        
        all_images = []
        for folder_name in tqdm(jpeg_folders):
            folder = JPEG_DIR / folder_name
            images = list(folder.glob('*.jpg')) + list(folder.glob('*.png'))
            if len(images) >= 2:
                # Get smaller image (ROI)
                img = min(images, key=lambda x: x.stat().st_size)
                all_images.append(img)
        
        print(f"   Found {len(all_images)} ROI crops")
        
        # Since we can't match to classes, we'll need dicom_info.csv
        print("\n   Loading dicom_info.csv for class matching...")
        dicom_info = pd.read_csv(CSV_DIR / "dicom_info.csv")
        
        # Match images to classes
        print("   Matching to cases...")
        for img in tqdm(all_images):
            folder_uid = img.parent.name
            
            # Find in dicom_info
            matching = dicom_info[dicom_info['file_path'].str.contains(folder_uid, na=False)]
            
            if len(matching) > 0:
                # Get patient ID from file path
                file_path = matching['file_path'].iloc[0]
                
                for pid, cls in patient_class.items():
                    if pid in str(file_path):
                        class_images[cls].append(img)
                        break
    
    # Final check and copy
    total_images = sum(len(v) for v in class_images.values())
    print(f"\nTotal images to copy: {total_images}")
    
    if total_images == 0:
        print("❌ Still no matches found. Please check dataset structure.")
        return
    
    # Copy files
    print("\nCopying files...")
    random.seed(42)
    
    stats = {s: {c: 0 for c in ['benign', 'suspicious', 'malignant']} 
             for s in ['train', 'val', 'test']}
    
    for cls, images in class_images.items():
        random.shuffle(images)
        n = len(images)
        
        train_end = int(n * TRAIN_RATIO)
        val_end = int(n * (TRAIN_RATIO + VAL_RATIO))
        
        for i, img in enumerate(images):
            if i < train_end:
                split = 'train'
            elif i < val_end:
                split = 'val'
            else:
                split = 'test'
            
            dst = OUTPUT_DIR / split / cls / f"{stats[split][cls]:04d}.jpg"
            shutil.copy2(img, dst)
            stats[split][cls] += 1
    
    # Summary
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    total = 0
    for split in ['train', 'val', 'test']:
        split_total = sum(stats[split].values())
        total += split_total
        print(f"\n{split.upper()}: {split_total}")
        for cls in ['benign', 'suspicious', 'malignant']:
            print(f"  {cls}: {stats[split][cls]}")
    
    print(f"\n✅ Total copied: {total}")


if __name__ == "__main__":
    main()
