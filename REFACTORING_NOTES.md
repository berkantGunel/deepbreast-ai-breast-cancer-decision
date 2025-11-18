# Yeniden Organize Edilmiş src/ Klasör Yapısı

Proje `src/` klasörü, kodun daha anlaşılır ve bakımı kolay olması için mantıksal alt klasörlere ayrılmıştır.

## 📂 Yeni Yapı

```
src/
├── core/                 # Temel bileşenler
│   ├── model.py              → CNN mimarisi (BreastCancerCNN)
│   ├── data_loader.py        → Veri yükleme ve DataLoader'lar
│   └── xai_visualizer.py     → Grad-CAM implementasyonu
│
├── training/             # Eğitim ve değerlendirme
│   ├── train_model.py        → Model eğitim scripti
│   ├── evaluate_model.py     → Model değerlendirme scripti
│   └── organize_dataset.py   → Veri organizasyonu scripti
│
├── ui/                   # Streamlit arayüzü
│   ├── app.py                → Ana Streamlit uygulaması
│   ├── predict.py            → Tahmin paneli
│   ├── analysis_panel.py     → Grad-CAM analiz paneli
│   ├── performance.py        → Performans metrikleri gösterimi
│   └── about.py              → Hakkında sayfası
│
└── scripts/              # Bağımsız test scriptleri
    └── test_xai.py           → XAI/Grad-CAM test scripti
```

## 🔄 Değişiklikler

### Önceki Yapı → Yeni Yapı

**Core Modüller:**

- `src/model.py` → `src/core/model.py`
- `src/data_loader.py` → `src/core/data_loader.py`
- `src/xai_visualizer.py` → `src/core/xai_visualizer.py`

**Training Modülleri:**

- `src/train_model.py` → `src/training/train_model.py`
- `src/evaluate_model.py` → `src/training/evaluate_model.py`
- `src/organize_dataset.py` → `src/training/organize_dataset.py`

**UI Modülleri:**

- `src/app.py` → `src/ui/app.py`
- `src/predict.py` → `src/ui/predict.py`
- `src/analysis_panel.py` → `src/ui/analysis_panel.py`
- `src/performance.py` → `src/ui/performance.py`
- `src/about.py` → `src/ui/about.py`

**Scripts:**

- `src/test_xai.py` → `src/scripts/test_xai.py`

## 🚀 Kullanım

### Streamlit Uygulaması

Proje kökünde wrapper `app.py` dosyası mevcut:

```bash
streamlit run app.py
```

veya doğrudan:

```bash
python -m streamlit run app.py
```

### Eğitim

```bash
python src/training/train_model.py
```

### Değerlendirme

```bash
python src/training/evaluate_model.py
```

### Veri Organizasyonu

```bash
python src/training/organize_dataset.py
```

### XAI Test

```bash
python src/scripts/test_xai.py
```

## 📝 Import Değişiklikleri

Tüm import ifadeleri yeni yapıya uygun şekilde güncellenmiştir:

**Önceki:**

```python
from model import BreastCancerCNN
from data_loader import train_loader
from xai_visualizer import generate_gradcam
```

**Yeni:**

```python
from src.core.model import BreastCancerCNN
from src.core.data_loader import train_loader
from src.core.xai_visualizer import generate_gradcam
```

## ✅ Test Edildi

- ✅ Streamlit uygulaması başarıyla çalışıyor (`http://localhost:8503`)
- ✅ Import yapıları doğru şekilde güncellendi
- ✅ Tüm modüller birbirleriyle uyumlu çalışıyor
- ✅ Kodun çalışırlığı korundu

## 💡 Faydalar

1. **Daha İyi Organizasyon**: Her modül kendi amacına uygun klasörde
2. **Kolay Navigasyon**: Dosyaları bulmak ve anlamak daha kolay
3. **Bakım Kolaylığı**: İlgili dosyalar birlikte gruplandırılmış
4. **Ölçeklenebilirlik**: Yeni özellikler eklemek daha kolay
5. **Modülerlik**: Her klasör bağımsız bir modül olarak çalışabilir

## 🔧 Notlar

- Tüm scriptler proje kökünden (`BreastCancerPrediction_BCP/`) çalıştırılmalıdır
- Python path ayarlaması otomatik olarak yapılmaktadır (her scriptte)
- Kök dizindeki `app.py` wrapper olarak `src/ui/app.py`'yi çağırır
- Lint uyarıları mevcuttur ancak kod çalışırlığını etkilememektedir
