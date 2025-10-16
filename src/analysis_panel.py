import streamlit as st
from PIL import Image
import io, os
import torch
from model import BreastCancerCNN
from xai_visualizer import generate_gradcam

# ==========================================================
# 🧠 Model yükleme
# ==========================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model


# ==========================================================
# 📊 Analysis Panel — Multi-Image Grad-CAM Gallery
# ==========================================================
def run_analysis():
    st.title("📊 Analysis Panel — Grad-CAM Visualization Gallery")
    st.write("""
    Upload one or more histopathology images to explore where the model focuses  
    while predicting whether tissue is **Benign** or **Malignant**.
    """)

    # ------------------------------------------------------
    # 📂 Image Upload Section (multi-upload)
    # ------------------------------------------------------
    uploaded_files = st.file_uploader(
        "📁 Upload one or more images:", type=["jpg", "jpeg", "png"], accept_multiple_files=True
    )

    if not uploaded_files:
        st.info("Please upload one or more histopathology images to visualize Grad-CAM results.")
        return

    model = load_model()

    # Grad-CAM results will be stored here
    results = []

    st.divider()
    st.subheader("🧠 Generating Grad-CAM Heatmaps")

    # ------------------------------------------------------
    # ⚙️ Process each uploaded image
    # ------------------------------------------------------
    progress = st.progress(0)
    for idx, uploaded_file in enumerate(uploaded_files):
        image = Image.open(uploaded_file).convert("RGB")

        # Temporary file for Grad-CAM generation
        temp_path = f"temp_image_{idx}.png"
        image.save(temp_path)

        gradcam_img, pred_class = generate_gradcam(model, temp_path, target_layer_name="conv4", device=device)
        os.remove(temp_path)

        pred_label = "Benign" if pred_class == 0 else "Malignant"
        results.append({
            "original": image,
            "gradcam": gradcam_img,
            "label": pred_label
        })

        progress.progress((idx + 1) / len(uploaded_files))

    st.success("✅ Grad-CAM generation completed!")
    st.divider()

    # ------------------------------------------------------
    # 🎨 Display Gallery
    # ------------------------------------------------------
    st.subheader("🩺 Grad-CAM Visualization Gallery")

    # Dynamic grid layout (2 per row)
    cols_per_row = 2
    for i in range(0, len(results), cols_per_row):
        row_items = results[i:i + cols_per_row]
        cols = st.columns(cols_per_row)

        for j, item in enumerate(row_items):
            with cols[j]:
                st.markdown(
                    f"<div style='text-align:center;font-size:15px;margin-bottom:5px;'><b>{item['label']}</b></div>",
                    unsafe_allow_html=True
                )

                alpha_key = f"alpha_{i+j}"
                if alpha_key not in st.session_state:
                    st.session_state[alpha_key] = 0.6

                alpha = st.slider(
                    f"Transparency for Image {i+j+1}",
                    0.1, 1.0,
                    st.session_state[alpha_key],
                    step=0.05,
                    key=f"slider_{i+j}"
                )
                st.session_state[alpha_key] = alpha

                blended = Image.blend(
                    item["original"].resize(item["gradcam"].size),
                    item["gradcam"],
                    alpha=alpha
                )

                st.image(blended, caption=f"Grad-CAM Heatmap (α={alpha:.2f})", width=340)

                with st.expander("🔍 View Full Size"):
                    st.image(blended, use_container_width=True)

    st.caption("🧬 DeepBreast: Multi-Image Explainable AI Visualization Panel")
