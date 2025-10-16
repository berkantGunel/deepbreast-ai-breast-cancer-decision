import streamlit as st
import os
import base64

from predict import run_prediction
from analysis_panel import run_analysis
from performance import run_performance
from about import run_about

# ======================================================
# 🩺 Streamlit Config
# ======================================================
st.set_page_config(
    page_title="DeepBreast: AI-Based Breast Cancer Detection",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================================================
# 🧬 Logo Base64 Loader
# ======================================================
def load_base64_image(path):
    """Convert an image file into base64 for inline display."""
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# Logo path (main directory -> logo_assets/deep_breast.png)
logo_path = os.path.join(os.path.dirname(__file__), "..", "logo_assets", "deep_breast.png")
if os.path.exists(logo_path):
    logo_base64 = load_base64_image(logo_path)
else:
    logo_base64 = ""  # fallback if missing

# ======================================================
# 🎨 Custom Header
# ======================================================
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

# ======================================================
# 🎨 Custom Footer
# ======================================================
def app_footer():
    footer_html = """
    <hr>
    <div style="text-align:center;color:gray;font-size:13px;padding-top:10px;">
        © 2025 DeepBreast | Developed for Academic Research Purposes.<br>
        <span style="font-size:12px;">Faculty of Engineering — Department of Software Engineering</span>
    </div>
    """
    st.markdown(footer_html, unsafe_allow_html=True)

# ======================================================
# 🧭 Sidebar Navigation
# ======================================================
st.sidebar.title("🧭 Navigation")
page = st.sidebar.radio("Go to:", [
    "🔍 Prediction",
    "📊 Analysis Panel",
    "📈 Model Performance",
    "⚙️ About"
])

# ======================================================
# 🚀 Main Layout
# ======================================================
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
