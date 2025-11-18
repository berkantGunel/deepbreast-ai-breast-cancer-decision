🧠 DeepBreast: AI-Based Breast Cancer Decision
📘 Overview
This project focuses on developing a deep learning-based Convolutional Neural Network (CNN) to classify histopathological breast tissue images as benign or malignant.
The model automatically learns hierarchical image features and assists in AI-supported early diagnosis of breast cancer.
py -3.10 -m streamlit run app.py

🧩 Key Features
CNN architecture optimized for histopathological image classification
Automated data preprocessing and augmentation
Real-time training progress visualization with tqdm
GPU acceleration support (CUDA)
Model checkpoint saving (best_model.pth)

🗂️ Project Structure
BREASTCANCERPREDICTION_BCP/
│
├── data/
│ ├── raw/ # Original dataset (too large, not uploaded)
│ └── processed/ # Organized dataset (benign/malignant folders)
│
├── models/
│ └── best_model.pth # Best-performing trained model
│
├── src/
│ ├── core/ # Core components
│ │ ├── model.py # CNN architecture
│ │ ├── data_loader.py # Dataset loading and splitting
│ │ └── xai_visualizer.py # Grad-CAM implementation
│ │
│ ├── training/ # Training and evaluation
│ │ ├── train_model.py # Training and validation loop
│ │ ├── evaluate_model.py # Model evaluation script
│ │ └── organize_dataset.py # Dataset organization script
│ │
│ ├── ui/ # Streamlit interface
│ │ ├── app.py # Main Streamlit application
│ │ ├── predict.py # Prediction panel
│ │ ├── analysis_panel.py # Grad-CAM analysis panel
│ │ ├── performance.py # Performance metrics dashboard
│ │ └── about.py # About page
│ │
│ └── scripts/ # Standalone test scripts
│ └── test_xai.py # XAI testing script
│
├── app.py # Streamlit entry point (wrapper)
├── notebooks/ # Jupyter notebooks for experiments
├── requirements.txt # Dependencies
└── README.md # Project description

⚙️ Installation
1️⃣ Clone the repository
git clone https://github.com/<your-username>/deepbreast-ai-breast-cancer-decision.git
cd deepbreast-ai-breast-cancer-decision

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Prepare dataset
Download the Breast Histopathology Images dataset from Kaggle:
https://www.kaggle.com/datasets/paultimothymooney/breast-histopathology-images
Then organize it
python src/training/organize_dataset.py

🚀 Training
Run model training:
python src/training/train_model.py

📊 Evaluation
Run model evaluation:
python src/training/evaluate_model.py

🧪 Testing
Test XAI/Grad-CAM visualization:
python src/scripts/test_xai.py

📊 Results
Metric Value
Best Validation Accuracy ~90%
Loss Function CrossEntropyLoss
Optimizer Adam (lr=0.001)

🎯 Future Work
Add web-based UI for image upload & prediction
Integrate Grad-CAM visualization for explainability
Perform hyperparameter tuning for higher accuracy
In the future, this framework can be extended to analyze mammogram images for early-stage cancer detection using transfer learning models such as ResNet50 or EfficientNet.
(Maybe https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images)

👨‍💻 Author

Berkant Günel
Software Engineering – Nişantaşı University

💻 Streamlit Web Interface

The project now includes a fully interactive Streamlit interface designed for medical-grade AI presentation and usability.
Users can upload histopathology images, view predictions, interpret Grad-CAM visualizations, and export performance reports in PDF format.

Interface Sections:
Section Description
🧭 Prediction Upload tissue images → View predicted label (Benign / Malignant), confidence score, and inference time
📊 Analysis Grad-CAM heatmaps and transparency control for model interpretability
📈 Performance Live training logs, validation curves, confusion matrix, and downloadable PDF reports
ℹ️ About Project overview, developer info, and version notes
🖼️ UI Preview (optional)
#i will add image
