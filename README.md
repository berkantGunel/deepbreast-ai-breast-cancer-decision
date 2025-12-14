# 🩺 DeepBreast AI - Breast Cancer Detection

<div align="center">

**AI-powered breast cancer detection system using deep learning for histopathology image analysis.**

[![Version](https://img.shields.io/badge/version-2.1.0-blue?style=for-the-badge)](https://github.com/berkantGunel/deepbreast-ai-breast-cancer-decision)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.0-3178C6?style=for-the-badge&logo=typescript&logoColor=white)](https://typescriptlang.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

[Features](#-features) • [Installation](#️-installation) • [Usage](#️-usage) • [API](#-api-endpoints) • [Model](#-model-architecture) • [Screenshots](#-screenshots)

</div>

---

## 📋 Table of Contents

- [About The Project](#-about-the-project)
- [Features](#-features)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Installation](#️-installation)
- [Usage](#️-usage)
- [API Endpoints](#-api-endpoints)
- [Model Architecture](#-model-architecture)
- [Dataset](#-dataset)
- [Performance](#-performance)
- [Screenshots](#-screenshots)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)
- [Acknowledgments](#-acknowledgments)

---

## 🎯 About The Project

**DeepBreast AI** is a comprehensive medical image analysis system designed to assist pathologists in detecting breast cancer from histopathology images. The system leverages state-of-the-art deep learning techniques to classify tissue samples as **benign** or **malignant** with high accuracy.

### Why This Project?

- 🏥 **Medical Impact**: Early detection of breast cancer significantly improves survival rates
- 🤖 **AI-Assisted Diagnosis**: Reduces human error and speeds up the diagnostic process
- 🔍 **Explainable AI**: Grad-CAM visualizations help understand model decisions
- 📊 **Production Ready**: Full-stack application with modern web technologies

---

## 🚀 Features

### Core Features

| Feature                             | Description                                              |
| ----------------------------------- | -------------------------------------------------------- |
| 🧠 **Deep Learning Classification** | CNN-based model trained on histopathology images         |
| 🔥 **Enhanced Grad-CAM**            | Multiple XAI methods: Grad-CAM, Grad-CAM++, Score-CAM    |
| ⚡ **Real-time Prediction**         | Instant analysis with confidence scores                  |
| 📈 **Performance Metrics**          | Detailed accuracy, precision, recall, and F1 metrics     |
| 📊 **Training History**             | Visualize model training progress over epochs            |
| 📜 **Analysis History**             | Track and review past predictions with local storage     |

### Technical Features

| Feature                | Description                                          |
| ---------------------- | ---------------------------------------------------- |
| 🌐 **REST API**        | FastAPI backend with automatic OpenAPI documentation |
| 💻 **Modern Frontend** | React 18 + TypeScript with modern UI design          |
| 🎨 **Tailwind CSS**    | Beautiful glassmorphism UI components                |
| 🔄 **Hot Reload**      | Development servers with live reload                 |
| 📱 **Mobile Friendly** | Responsive design works on all devices               |
| 🧭 **Lucide Icons**    | Modern, consistent iconography throughout the app    |

---

## 🛠 Tech Stack

### Backend

| Technology                                                                                      | Purpose                   |
| ----------------------------------------------------------------------------------------------- | ------------------------- |
| ![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)    | Core programming language |
| ![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat&logo=fastapi&logoColor=white) | REST API framework        |
| ![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white) | Deep learning framework   |
| ![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)    | Image processing          |
| ![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat&logo=numpy&logoColor=white)       | Numerical computing       |

### Frontend

| Technology                                                                                               | Purpose              |
| -------------------------------------------------------------------------------------------------------- | -------------------- |
| ![React](https://img.shields.io/badge/React-61DAFB?style=flat&logo=react&logoColor=black)                | UI library           |
| ![TypeScript](https://img.shields.io/badge/TypeScript-3178C6?style=flat&logo=typescript&logoColor=white) | Type-safe JavaScript |
| ![Vite](https://img.shields.io/badge/Vite-646CFF?style=flat&logo=vite&logoColor=white)                   | Build tool           |
| ![Tailwind](https://img.shields.io/badge/Tailwind-06B6D4?style=flat&logo=tailwindcss&logoColor=white)    | CSS framework        |
| ![Axios](https://img.shields.io/badge/Axios-5A29E4?style=flat&logo=axios&logoColor=white)                | HTTP client          |
| ![Recharts](https://img.shields.io/badge/Recharts-22B5BF?style=flat)                                     | Chart visualization  |

---

## 📁 Project Structure

```
BreastCancerPrediction_BCP/
│
├── 📂 deepbreastai/              # React + Vite Frontend
│   ├── 📂 src/
│   │   ├── 📂 components/        # Reusable UI components
│   │   │   ├── Sidebar.tsx       # Navigation sidebar
│   │   │   └── Navbar.tsx        # Top navigation bar
│   │   ├── 📂 pages/             # Application pages
│   │   │   ├── Home.tsx          # Landing page (redesigned)
│   │   │   ├── Predict.tsx       # Image upload & prediction
│   │   │   ├── Analysis.tsx      # Enhanced Grad-CAM visualization
│   │   │   ├── History.tsx       # Analysis history tracking
│   │   │   ├── Metrics.tsx       # Performance dashboard
│   │   │   └── About.tsx         # Project information
│   │   ├── 📂 services/          # API integration
│   │   │   └── api.ts            # Axios HTTP client
│   │   ├── App.tsx               # Main application component
│   │   ├── main.tsx              # Application entry point
│   │   └── index.css             # Global styles
│   ├── package.json              # Node.js dependencies
│   ├── vite.config.ts            # Vite configuration
│   ├── tailwind.config.js        # Tailwind CSS config
│   └── tsconfig.json             # TypeScript configuration
│
├── 📂 src/                       # Python Backend
│   ├── 📂 api/                   # FastAPI Application
│   │   ├── main.py               # API entry point & CORS
│   │   ├── 📂 endpoints/         # API route handlers
│   │   │   ├── predict.py        # /api/predict endpoint
│   │   │   ├── gradcam.py        # /api/gradcam endpoint
│   │   │   └── metrics.py        # /api/metrics endpoint
│   │   └── 📂 utils/             # Utility functions
│   │       ├── model_loader.py   # Model loading & caching
│   │       └── image_utils.py    # Image preprocessing
│   │
│   ├── 📂 core/                  # Core ML Components
│   │   ├── model.py              # CNN architecture definition
│   │   ├── data_loader.py        # Dataset & DataLoader
│   │   └── xai_visualizer.py     # Grad-CAM implementation
│   │
│   ├── 📂 training/              # Training Scripts
│   │   ├── train_model.py        # Model training loop
│   │   ├── evaluate_model.py     # Model evaluation
│   │   └── organize_dataset.py   # Data preparation
│   │
│   └── 📂 ui/                    # Legacy Streamlit UI
│       ├── app.py                # Main Streamlit app
│       └── ...                   # Other UI components
│
├── 📂 models/                    # Trained Models & Results
│   ├── best_model.pth            # Best model weights (not in repo)
│   ├── eval_results.json         # Evaluation metrics
│   └── train_history.json        # Training history
│
├── 📂 data/                      # Dataset (not in repo)
│   ├── 📂 raw/                   # Original images
│   └── 📂 processed/             # Preprocessed images
│
├── 📂 reports/                   # Documentation
│   └── DeepBreast_Model_Report.pdf
│
├── requirements.txt              # Python dependencies
├── .gitignore                    # Git ignore rules
└── README.md                     # This file
```

---

## ⚙️ Installation

### Prerequisites

- **Python 3.11+** with pip
- **Node.js 18+** with npm
- **CUDA 11.8+** (optional, for GPU acceleration)
- **Git**

### 1. Clone the Repository

```bash
git clone https://github.com/berkantGunel/deepbreast-ai-breast-cancer-decision.git
cd deepbreast-ai-breast-cancer-decision
```

### 2. Backend Setup (Python)

```bash
# Create and activate virtual environment
python -m venv venv

# Windows
.\venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Verify PyTorch CUDA (optional)
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### 3. Frontend Setup (React)

```bash
# Navigate to frontend directory
cd deepbreastai

# Install Node.js dependencies
npm install

# Return to project root
cd ..
```

### 4. Download Model Weights

Place your trained model file `best_model.pth` in the `models/` directory:

```
models/
├── best_model.pth      # Your trained model
├── eval_results.json   # Evaluation metrics
└── train_history.json  # Training history
```

---

## 🖥️ Usage

### Start the Application

#### 1. Start Backend Server

```bash
# From project root, with venv activated
python -m uvicorn src.api.main:app --reload --port 8000
```

The API will be available at: `http://localhost:8000`  
API Documentation: `http://localhost:8000/docs`

#### 2. Start Frontend Server

```bash
# In a new terminal
cd deepbreastai
npm run dev
```

The frontend will be available at: `http://localhost:5173`

### Mobile Access (Same Network)

```bash
# Start frontend with network access
npm run dev -- --host 0.0.0.0

# Access from mobile using your PC's IP
# Example: http://192.168.1.100:5173
```

### Quick Start (Both Servers)

**Terminal 1 - Backend:**

```bash
.\venv\Scripts\activate
python -m uvicorn src.api.main:app --reload --port 8000
```

**Terminal 2 - Frontend:**

```bash
cd deepbreastai
npm run dev
```

---

## 📊 API Endpoints

### Base URL: `http://localhost:8000`

| Endpoint                | Method | Description      | Request                           | Response                                         |
| ----------------------- | ------ | ---------------- | --------------------------------- | ------------------------------------------------ |
| `/api/health`           | GET    | Health check     | -                                 | `{ "status": "healthy" }`                        |
| `/api/predict`          | POST   | Classify image   | `multipart/form-data` with `file` | `{ "prediction": "Benign", "confidence": 95.5 }` |
| `/api/gradcam`          | POST   | Generate heatmap | `multipart/form-data` with `file` | `{ "gradcam_image": "base64..." }`               |
| `/api/metrics`          | GET    | Model metrics    | -                                 | `{ "accuracy": 89.5, "precision": 80.7, ... }`   |
| `/api/training-history` | GET    | Training data    | -                                 | `[{ "epoch": 1, "loss": 0.5, ... }]`             |

### Example API Calls

```bash
# Health Check
curl http://localhost:8000/api/health

# Predict Image
curl -X POST -F "file=@image.png" http://localhost:8000/api/predict

# Get Metrics
curl http://localhost:8000/api/metrics
```

---

## 🧠 Model Architecture

### Network Design

```
Input (50x50x3 RGB Image)
         │
    ┌────┴────┐
    │  Conv2D │ 32 filters, 3x3, ReLU
    │ MaxPool │ 2x2
    └────┬────┘
         │
    ┌────┴────┐
    │  Conv2D │ 64 filters, 3x3, ReLU
    │ MaxPool │ 2x2
    └────┬────┘
         │
    ┌────┴────┐
    │  Conv2D │ 128 filters, 3x3, ReLU
    │ MaxPool │ 2x2
    └────┬────┘
         │
    ┌────┴────┐
    │ Flatten │
    │  Dense  │ 256 units, ReLU, Dropout(0.5)
    │  Dense  │ 2 units, Softmax
    └────┬────┘
         │
Output (Benign / Malignant)
```

### Training Configuration

| Parameter             | Value                |
| --------------------- | -------------------- |
| **Optimizer**         | Adam                 |
| **Learning Rate**     | 0.001                |
| **Loss Function**     | CrossEntropyLoss     |
| **Batch Size**        | 32                   |
| **Epochs**            | 50                   |
| **Early Stopping**    | Patience: 10         |
| **Data Augmentation** | Rotation, Flip, Zoom |

---

## 📚 Dataset

### Breast Histopathology Images (BreakHis-inspired)

The model is trained on histopathology image patches:

| Class         | Description            | Samples |
| ------------- | ---------------------- | ------- |
| **Benign**    | Non-cancerous tissue   | ~20,000 |
| **Malignant** | Cancerous tissue (IDC) | ~8,000  |

### Data Split

| Set        | Percentage | Purpose               |
| ---------- | ---------- | --------------------- |
| Training   | 70%        | Model training        |
| Validation | 15%        | Hyperparameter tuning |
| Test       | 15%        | Final evaluation      |

### Image Specifications

- **Size**: 50x50 pixels
- **Format**: PNG
- **Color**: RGB (3 channels)
- **Source**: Breast histopathology slides at 40x magnification

---

## 📈 Performance

### Evaluation Metrics

| Metric        | Value  | Description                          |
| ------------- | ------ | ------------------------------------ |
| **Accuracy**  | 89.59% | Overall correct predictions          |
| **Precision** | 80.66% | True positives / Predicted positives |
| **Recall**    | 83.26% | True positives / Actual positives    |
| **F1-Score**  | 81.94% | Harmonic mean of precision & recall  |
| **AUC-ROC**   | ~0.91  | Area under ROC curve                 |

### Confusion Matrix

|                      | Predicted Benign | Predicted Malignant |
| -------------------- | ---------------- | ------------------- |
| **Actual Benign**    | 18,314 (TN)      | 1,571 (FP)          |
| **Actual Malignant** | 1,317 (FN)       | 6,552 (TP)          |

### Training Progress

The model converges after approximately 30-40 epochs with early stopping preventing overfitting.

---

## 🗺 Roadmap

### Completed ✅
- [x] CNN Model Training (v1.0 Baseline: 89.32%)
- [x] Transfer Learning with ResNet18 (v2.0: 92.86% accuracy)
- [x] Enhanced Grad-CAM (Grad-CAM++, Score-CAM) - v2.1
- [x] FastAPI Backend with XAI endpoints
- [x] React Frontend with modern UI
- [x] Performance Metrics Dashboard
- [x] Test-Time Augmentation (Implemented but disabled due to recall drop)
- [x] **UI Redesign** - All pages redesigned with glassmorphism (v2.1)
- [x] **Analysis History Page** - Track past predictions (v2.1)
- [x] **Modern Icon System** - Lucide React icons (v2.1)

### In Progress 🚧
- [ ] Batch Prediction API
- [ ] Saliency Maps & Advanced XAI
- [ ] Dark/Light Mode Toggle

### Planned 📋
- [ ] Progressive Web App (PWA)
- [ ] Enhanced PDF Reports
- [ ] SQLite + Real-time Statistics
- [ ] Caching System
- [ ] Model Versioning System
- [ ] Docker Containerization
- [ ] CI/CD Pipeline
- [ ] Cloud Deployment (AWS/GCP)

### Research Ideas 💡
- [ ] Model Ensemble Methods
- [ ] Multi-class Classification (4+ tumor types)
- [ ] Attention Mechanisms
- [ ] Vision Transformers (ViT)

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/AmazingFeature`)
3. **Commit** your changes (`git commit -m 'Add AmazingFeature'`)
4. **Push** to the branch (`git push origin feature/AmazingFeature`)
5. **Open** a Pull Request

---

## 👨‍💻 Author

<div align="center">

**Berkant Günel**

[![GitHub](https://img.shields.io/badge/GitHub-100000?style=for-the-badge&logo=github&logoColor=white)](https://github.com/berkantGunel)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-0077B5?style=for-the-badge&logo=linkedin&logoColor=white)](https://linkedin.com/in/berkantgunel)

_Software Engineering_  
_Graduation Project - 2025_

</div>

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

```
MIT License

Copyright (c) 2025 Berkant Günel

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- [BreakHis Dataset](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) - Histopathology images
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [FastAPI](https://fastapi.tiangolo.com/) - Modern Python web framework
- [React](https://reactjs.org/) - Frontend library
- [Tailwind CSS](https://tailwindcss.com/) - CSS framework
- [Grad-CAM Paper](https://arxiv.org/abs/1610.02391) - Explainable AI technique

---

<div align="center">

**⭐ Star this repository if you find it helpful! ⭐**

Made with ❤️ for better healthcare

</div>

## 🚧 Future Works

### Planned Features & Improvements

#### 🔥 High Priority

1. **✅ Transfer Learning with Pre-trained Models** _(COMPLETED)_

   - ResNet18 with ImageNet pre-trained weights
   - **Accuracy improvement: 89.32% → 92.86%** (+3.54%)
   - Faster convergence: 5 epochs vs 10 epochs
   - Model: `models/best_model_resnet18.pth`

2. **⚠️ Test-Time Augmentation (TTA)** _(IMPLEMENTED BUT DISABLED)_

   - **Status**: Code complete, disabled by default
   - **Issue**: Recall dropped -10.4% (86.45% → 76.05%)
   - **Reason**: Over-smoothing + class imbalance + aggressive augmentations
   - **Available**: Optional via `use_tta=true` parameter
   - **Recommendation**: Use standard prediction (92.86% accuracy)

3. **Attention Mechanism**

   - Self-attention layers for better feature focus
   - Improved interpretability
   - Enhanced Grad-CAM visualization

4. **Batch Prediction (Bulk Analysis)**
   - Upload and analyze multiple images at once
   - Batch reporting and export
   - Time-saving for pathologists

#### 📊 XAI & Visualization

5. **Grad-CAM++ & Advanced XAI**

   - Grad-CAM++, Score-CAM, LayerCAM implementations
   - More accurate heatmaps
   - Better interpretability

6. **Saliency Maps**
   - Pixel-level importance visualization
   - Complementary to Grad-CAM
   - Enhanced explainability

#### 🎨 Frontend & UX

7. **Dark/Light Mode Toggle**

   - User preference theme switching
   - Better accessibility
   - Reduced eye strain

8. **Progressive Web App (PWA)**

   - Installable on mobile devices
   - Offline support with caching
   - Native app-like experience

9. **Enhanced PDF Reports**
   - Include Grad-CAM visualizations
   - Patient information forms
   - Similar case examples from dataset
   - Digital signature support

#### 📈 Performance & Backend

10. **Real-time Statistics Dashboard**

    - SQLite database integration
    - Daily/weekly/monthly analytics
    - Usage tracking and insights

11. **Caching System**

    - Redis/in-memory caching for predictions
    - 10x faster repeated queries
    - Reduced server load

12. **Model Versioning**
    - Multiple model versions (v1.0, v1.1, v2.0)
    - A/B testing capability
    - Rollback support

#### 🐳 Deployment

13. **Docker Containerization** _(Final Step)_
    - Multi-stage Docker build
    - docker-compose for easy deployment
    - CI/CD pipeline integration
    - Cloud-ready (AWS, GCP, Azure)

---

🎯 Önerilen İlerleme Sırası
İşte mantıklı bir sıralama - her adım bir sonraki için temel oluşturuyor:

📅 Faz 1: Model İyileştirmeleri (Temel - 3-4 gün)

1. Transfer Learning 🔥 (1-2 gün)

En büyük performans artışı
Diğer özellikler için daha iyi model
Başlamadan önce: mevcut modeli yedekle 2. Test-Time Augmentation (4-6 saat)

Transfer Learning'e kolayca eklenebilir
Performansı +%2-3 artırır
Kod olarak basit 3. Attention Mechanism (1 gün)

Transfer Learning üzerine eklenebilir
XAI özelliklerini güçlendirir
Grad-CAM için faydalı
📅 Faz 2: XAI & Görselleştirme (Orta - 2-3 gün) 4. Grad-CAM++ (6-8 saat)

Mevcut Grad-CAM kodunu geliştirir
Kütüphane kullanarak kolay
Attention Mechanism ile uyumlu çalışır 5. Saliency Maps (4-6 saat)

Grad-CAM'e tamamlayıcı
Frontend'de yan yana gösterebiliriz
Basit implementasyon
📅 Faz 3: Backend & Database (Orta - 2 gün) 6. SQLite + Gerçek Zamanlı İstatistikler (1 gün)

Her tahmin kaydedilecek
Caching için gerekli altyapı
Batch prediction için veritabanı lazım 7. Caching Sistemi (6-8 saat)

SQLite üzerine eklenebilir
Aynı görüntü tekrar yüklenirse cache'ten döner
API hızını 10x artırır
📅 Faz 4: Frontend Geliştirmeleri (Kolay - 2 gün) 8. Batch Prediction (6-8 saat)

Database hazır olmalı (Faz 3'ten)
Frontend + Backend birlikte
Kullanıcı deneyimini çok artırır 9. Dark/Light Mode (3-4 saat)

Tailwind ile çok kolay
Context API kullanacağız
localStorage ile kaydet 10. PWA (Progressive Web App) (4-6 saat)

manifest.json + service worker
Offline support
Mobil cihaza kurulabilir hale gelir
📅 Faz 5: Raporlama (Orta - 1 gün) 11. PDF Rapor Geliştirme (1 gün)

Grad-CAM++, Saliency Maps ekle
Database'den istatistikler çek
Profesyonel template
📅 Faz 6: Production Hazırlığı (İleri - 1-2 gün) 12. Model Versiyonlama (4-6 saat)

Transfer Learning modelini v2.0 olarak kaydet
API'ye version parametresi ekle
Database'e model_version kolonu 13. Docker Containerization (1 gün)

EN SON ADIM
Tüm özellikler tamamlanmış olmalı
Multi-stage build
docker-compose ile tek komutta çalıştır






1. 🔥 Model Uncertainty (MC Dropout)
Klinik kullanım için kritik - "Bu tahminden ne kadar eminiz?" sorusuna cevap verir.

2. 📄 PDF Rapor Çıktısı
Doktorların kullanabileceği profesyonel raporlar üretir.

3. 📁 DICOM Format Desteği
Tıbbi görüntüleme standardı - hastane sistemleriyle uyumluluk sağlar.

4. 🗄️ Batch Upload & Geçmiş Kayıtları
Pratik kullanım için önemli - SQLite ile basit veritabanı.


https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
mamagrofi dataset
