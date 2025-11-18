"""Static Streamlit page that communicates the project goal, dataset,
architecture, and current status to end users."""

import streamlit as st

def run_about():
    st.title("⚙️ About — DeepBreast Project")
    st.write("""
    This application is part of the **DeepBreast: AI-Based Breast Cancer Detection** project.

    **Developed by:** Berkant Günel  
    **Model:** Custom Convolutional Neural Network (CNN)  
    **Dataset:** BreakHis (Histopathological Images)  
    **Frameworks:** PyTorch, Streamlit  
    **Environment:** Python 3.10.14

    ---
    ### 💡 Project Description
    The goal of this system is to assist medical researchers by providing  
    an AI-based tool to classify histopathological images as **Benign** or **Malignant**.

    ### ⚙️ Technical Overview
    - CNN with 4 convolutional blocks  
    - Data augmentation and normalization  
    - Model trained for 10 epochs  
    - Grad-CAM for Explainable AI visualization  
    - Streamlit app for user interaction  

    ---
    ### 📈 Current Status
    Training completed successfully.  
    Best Validation Accuracy: **≈ 89.46%**
    """)
