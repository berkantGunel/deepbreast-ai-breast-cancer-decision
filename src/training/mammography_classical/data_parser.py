"""
DMID Mammography Dataset Parser
Info.txt dosyasını parse eder ve yapılandırılmış veri oluşturur
"""

import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import pandas as pd


@dataclass
class Lesion:
    """Tek bir lezyon bilgisi"""
    abnormality_type: str  # CIRC, SPIC, MISC, ARCH, CALC, ASYM, NORM
    pathology: str  # B, M, N
    x: Optional[int] = None
    y: Optional[int] = None
    radius: Optional[int] = None


@dataclass 
class MammogramRecord:
    """Bir mamografi görüntüsünün tüm bilgileri"""
    image_id: str  # IMG001, IMG002, ...
    tissue_type: str  # F, G, D
    lesions: List[Lesion] = field(default_factory=list)
    
    @property
    def is_normal(self) -> bool:
        """Görüntü normal mi?"""
        return len(self.lesions) == 0 or all(l.abnormality_type == 'NORM' for l in self.lesions)
    
    @property
    def has_malignant(self) -> bool:
        """Malignant lezyon var mı?"""
        return any(l.pathology == 'M' for l in self.lesions)
    
    @property
    def has_benign(self) -> bool:
        """Benign lezyon var mı?"""
        return any(l.pathology == 'B' for l in self.lesions)
    
    @property
    def primary_pathology(self) -> str:
        """Ana patoloji sınıfı (en kötü durumu döndürür)"""
        if self.has_malignant:
            return 'M'
        elif self.has_benign:
            return 'B'
        elif self.is_normal:
            return 'N'
        else:
            return 'N'  # Default
    
    @property
    def abnormality_types(self) -> List[str]:
        """Tüm anormallik türleri"""
        return list(set(l.abnormality_type for l in self.lesions if l.abnormality_type != 'NORM'))
    
    @property
    def primary_abnormality(self) -> str:
        """Ana anormallik türü (en riskli olanı döndürür)"""
        risk_order = ['SPIC', 'ARCH', 'MISC', 'CALC', 'ASYM', 'CIRC', 'NORM']
        types = self.abnormality_types
        if not types:
            return 'NORM'
        for risk_type in risk_order:
            if risk_type in types:
                return risk_type
        return types[0]


def parse_info_line(line: str) -> Optional[Tuple[str, str, Lesion]]:
    """
    Tek bir Info.txt satırını parse eder
    Format: IMG001  G  MISC  M  1567  3644  295
    """
    line = line.strip()
    if not line:
        return None
    
    # Tab ve boşlukları normalize et
    parts = re.split(r'\s+', line)
    parts = [p.strip() for p in parts if p.strip()]
    
    if len(parts) < 2:
        return None
    
    image_id = parts[0]  # IMG001
    
    # Tissue type (F, G, D veya -)
    tissue_type = parts[1] if len(parts) > 1 else 'G'
    if tissue_type == '-':
        tissue_type = 'G'  # Default
    
    # Abnormality type
    abnormality_type = parts[2] if len(parts) > 2 else 'NORM'
    
    # NORM durumu - lezyon yok
    if abnormality_type == 'NORM':
        lesion = Lesion(abnormality_type='NORM', pathology='N')
        return (image_id, tissue_type, lesion)
    
    # Pathology
    pathology = parts[3] if len(parts) > 3 else 'N'
    
    # Coordinates (optional)
    x = int(parts[4]) if len(parts) > 4 and parts[4].isdigit() else None
    y = int(parts[5]) if len(parts) > 5 and parts[5].isdigit() else None
    
    # Radius (may have '-' or be missing)
    radius = None
    if len(parts) > 6:
        try:
            radius = int(parts[6]) if parts[6] != '-' else None
        except ValueError:
            radius = None
    
    lesion = Lesion(
        abnormality_type=abnormality_type,
        pathology=pathology,
        x=x,
        y=y,
        radius=radius
    )
    
    return (image_id, tissue_type, lesion)


def parse_info_file(info_path: Path) -> Dict[str, MammogramRecord]:
    """
    Info.txt dosyasını parse eder ve tüm kayıtları döndürür
    """
    records: Dict[str, MammogramRecord] = {}
    
    with open(info_path, 'r', encoding='utf-8') as f:
        for line in f:
            result = parse_info_line(line)
            if result is None:
                continue
            
            image_id, tissue_type, lesion = result
            
            if image_id not in records:
                records[image_id] = MammogramRecord(
                    image_id=image_id,
                    tissue_type=tissue_type,
                    lesions=[]
                )
            
            records[image_id].lesions.append(lesion)
    
    return records


def create_dataset_dataframe(records: Dict[str, MammogramRecord]) -> pd.DataFrame:
    """
    MammogramRecord'lardan DataFrame oluşturur
    """
    data = []
    
    for image_id, record in records.items():
        row = {
            'image_id': image_id,
            'tissue_type': record.tissue_type,
            'tissue_name': {
                'F': 'Fatty',
                'G': 'Fatty-Glandular',
                'D': 'Dense-Glandular'
            }.get(record.tissue_type, 'Unknown'),
            'is_normal': record.is_normal,
            'primary_abnormality': record.primary_abnormality,
            'primary_pathology': record.primary_pathology,
            'pathology_name': {
                'B': 'Benign',
                'M': 'Malignant',
                'N': 'Normal'
            }.get(record.primary_pathology, 'Unknown'),
            'num_lesions': len([l for l in record.lesions if l.abnormality_type != 'NORM']),
            'has_malignant': record.has_malignant,
            'has_benign': record.has_benign,
            'abnormality_types': ','.join(record.abnormality_types) if record.abnormality_types else 'NORM',
        }
        
        # İlk lezyon koordinatları (varsa)
        if record.lesions and record.lesions[0].x is not None:
            row['lesion_x'] = record.lesions[0].x
            row['lesion_y'] = record.lesions[0].y
            row['lesion_radius'] = record.lesions[0].radius
        
        data.append(row)
    
    df = pd.DataFrame(data)
    df = df.sort_values('image_id').reset_index(drop=True)
    
    return df


def get_dataset_statistics(df: pd.DataFrame) -> Dict:
    """
    Dataset istatistiklerini döndürür
    """
    stats = {
        'total_images': len(df),
        'tissue_distribution': df['tissue_type'].value_counts().to_dict(),
        'pathology_distribution': df['primary_pathology'].value_counts().to_dict(),
        'abnormality_distribution': df['primary_abnormality'].value_counts().to_dict(),
        'normal_count': df['is_normal'].sum(),
        'abnormal_count': (~df['is_normal']).sum(),
        'malignant_count': df['has_malignant'].sum(),
        'benign_count': df['has_benign'].sum(),
    }
    return stats


if __name__ == "__main__":
    from config import INFO_FILE, PROCESSED_DIR
    
    print("Parsing Info.txt...")
    records = parse_info_file(INFO_FILE)
    print(f"Parsed {len(records)} mammogram records")
    
    print("\nCreating DataFrame...")
    df = create_dataset_dataframe(records)
    
    print("\nDataset Statistics:")
    stats = get_dataset_statistics(df)
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Save to CSV
    csv_path = PROCESSED_DIR / "dataset_info.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved dataset info to {csv_path}")
    
    print("\nSample records:")
    print(df.head(10).to_string())
