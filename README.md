# 🩺 DeepBreast AI - Breast Cancer Detection

AI-powered breast cancer detection system using deep learning for histopathology image analysis.

![Version](https://img.shields.io/badge/version-2.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.11-green)
![React](https://img.shields.io/badge/React-18-61DAFB)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688)

## 🚀 Features

- **Deep Learning Model**: CNN-based classification for breast cancer detection
- **Grad-CAM Visualization**: Explainable AI with heatmap overlays
- **Modern Web Interface**: React + TypeScript frontend with Streamlit-like design
- **REST API**: FastAPI backend for inference and metrics
- **Real-time Prediction**: Upload histopathology images for instant analysis

## 📁 Project Structure

```
BreastCancerPrediction_BCP/
├── deepbreastai/          # React + Vite frontend
│   ├── src/
│   │   ├── components/    # Sidebar, UI components
│   │   ├── pages/         # Home, Predict, Analysis, Metrics, About
│   │   ├── services/      # API client (axios)
│   │   └── types/         # TypeScript interfaces
│   └── package.json
├── src/
│   ├── api/               # FastAPI backend
│   │   ├── endpoints/     # predict, gradcam, metrics
│   │   └── utils/         # model loader, image utils
│   ├── model.py           # CNN architecture
│   ├── train_model.py     # Training script
│   └── evaluate_model.py  # Evaluation script
├── models/                # Trained model weights
├── data/                  # Dataset (not in repo)
└── requirements.txt       # Python dependencies
```

## 🛠️ Installation

### Backend (Python)

```bash
# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Start API server
python -m uvicorn src.api.main:app --reload --port 8000
```

### Frontend (React)

```bash
cd deepbreastai

# Install dependencies
npm install

# Start development server
npm run dev
```

## 🖥️ Usage

1. Start the backend API server (port 8000)
2. Start the frontend dev server (port 5173)
3. Open http://localhost:5173 in your browser
4. Navigate to **Predict** page and upload a histopathology image
5. View results with confidence score and Grad-CAM visualization

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check |
| `/api/predict` | POST | Image classification |
| `/api/gradcam` | POST | Grad-CAM heatmap |
| `/api/metrics` | GET | Model performance metrics |
| `/api/training-history` | GET | Training history data |

## 🧠 Model Architecture

- **Base**: Custom CNN / ResNet-based architecture
- **Input**: 50x50 RGB histopathology patches
- **Output**: Binary classification (Benign / Malignant)
- **Training**: Cross-entropy loss, Adam optimizer

## 📈 Performance

| Metric | Value |
|--------|-------|
| Accuracy | ~89.5% |
| Precision | ~80.7% |
| Recall | ~83.3% |
| F1-Score | ~81.9% |

## 🔗 Branches

- `v2-fastapi` (default): Current version - FastAPI + React
- `main`: Legacy version - Streamlit interface

## 👨‍💻 Author

**Berkant Günel**

## 📄 License

This project is for educational purposes (graduation project).
