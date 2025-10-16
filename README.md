🧠 DeepBreast: AI-Based Breast Cancer Decision
📘 Overview
This project focuses on developing a deep learning-based Convolutional Neural Network (CNN) to classify histopathological breast tissue images as benign or malignant.
The model automatically learns hierarchical image features and assists in AI-supported early diagnosis of breast cancer.

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
│   ├── raw/              # Original dataset (too large, not uploaded)
│   └── processed/        # Organized dataset (benign/malignant folders)
│
├── models/
│   └── best_model.pth    # Best-performing trained model
│
├── src/
│   ├── data_loader.py    # Dataset loading and splitting
│   ├── model.py          # CNN architecture
│   ├── organize_dataset.py # Dataset organization script
│   └── train_model.py    # Training and validation loop
│
├── ui/                   # (Future UI for user interaction)
├── notebooks/            # Jupyter notebooks for experiments
│
├── requirements.txt      # Dependencies
└── README.md             # Project description

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
python src/organize_dataset.py

🚀 Training
Run model training:
py 3.10 src/train_model.py

📊 Results
Metric	Value
Best Validation Accuracy	~90%
Loss Function	CrossEntropyLoss
Optimizer	Adam (lr=0.001)

🎯 Future Work
Add web-based UI for image upload & prediction
Integrate Grad-CAM visualization for explainability
Perform hyperparameter tuning for higher accuracy

👨‍💻 Author

Berkant Günel
Software Engineering – Nişantaşı University
