"""Streamlit entry point that assembles navigation, layout, and shared UI
elements for the DeepBreast application."""

import sys
from pathlib import Path
# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import streamlit as st
import os
import base64

from src.ui.predict import run_prediction
from src.ui.analysis_panel import run_analysis
from src.ui.performance import run_performance
from src.ui.about import run_about

# ======================================================
# 🩺 Streamlit Config
# ======================================================
st.set_page_config(
    page_title="DeepBreast: AI-Based Breast Cancer Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

#Logo Base64 Loader
#Btw base64:Converts data (images) to ASCII characters.
def load_base64_image(path):
    """Convert an image file into base64 for inline display."""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

#Logo path (main directory -> logo_assets/deep_breast.png)
# Get project root directory (3 levels up from src/ui/app.py)
project_root = Path(__file__).parent.parent.parent
logo_path = project_root / "logo_assets" / "deep_breast.png"
if logo_path.exists():
    logo_base64 = load_base64_image(str(logo_path))
    print(f"[DEBUG] Logo loaded from: {logo_path}")
else:
    logo_base64 = ""  # fallback if missing
    print(f"[WARNING] Logo not found at: {logo_path}")

#Custom header
def app_header():
    header_html = f"""
    <style>
        .app-header {{
            background-color: #1f1f2e;
            padding: 1.2rem;
            border-radius: 10px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            margin-bottom: 1rem;
        }}
        .header-left {{
            display: flex;
            align-items: center;
        }}
        .header-left img {{
            width: 55px;
            margin-right: 15px;
            border-radius: 6px;
        }}
        .header-title {{
            color: #E2E8F0;
            margin: 0;
        }}
        .header-subtitle {{
            color: #A0AEC0;
            font-size: 14px;
            margin: 0;
        }}
        .header-right {{
            color: #CBD5E0;
            font-size: 13px;
            text-align: right;
        }}
    </style>

    <div class="app-header">
        <div class="header-left">
            <img src="data:image/png;base64,{logo_base64}">
            <div>
                <h2 class="header-title">DeepBreast</h2>
                <p class="header-subtitle">AI-Powered Breast Cancer Detection and Explainability System</p>
            </div>
        </div>
        <div class="header-right">
            Version 1.0<br><b>Developed by Berkant Günel</b>
        </div>
    </div>
    """
    st.markdown(header_html, unsafe_allow_html=True)

#Custom footer
def app_footer():
    footer_html = """
    <hr>
    <div style="text-align:center;color:gray;font-size:13px;padding-top:10px;">
        © 2025 DeepBreast | Developed for Academic Research Purposes.<br>
        <span style="font-size:12px;">Faculty of Engineering — Department of Software Engineering</span>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

#sidebar navigation
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "🔍 Prediction",
    "📊 Analysis Panel",
    "📈 Model Performance",
    "⚙️ About"
])

#main layout
app_header()

if page == "🔍 Prediction":
    run_prediction()
elif page == "📊 Analysis Panel":
    run_analysis()
elif page == "📈 Model Performance":
    run_performance()
elif page == "⚙️ About":
    run_about()

app_footer()
