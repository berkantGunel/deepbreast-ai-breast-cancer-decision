"""Streamlit analysis panel that generates Grad-CAM overlays to explain
model focus regions for uploaded histopathology images."""

import streamlit as st
from PIL import Image
import io, os
import torch
import numpy as np
from src.core.model import BreastCancerCNN
from src.core.xai_visualizer import generate_gradcam

#model loading
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", 
                                      map_location=device, 
                                      weights_only=False))
    model.eval()
    return model


#Histopathology similarity check
def is_histopathology_like(image):
    """
    It checks whether the image resembles tissue under a microscope.
    It filters scene/object/outdoor photographs by
    examining the average color and contrast distribution.
    """
    try:
        # Resize large images for performance
        img_resized = image.copy()
        if max(img_resized.size) > 500:
            img_resized.thumbnail((500, 500), Image.Resampling.LANCZOS)
        
        img_np = np.array(img_resized)
        mean_color = np.mean(img_np, axis=(0, 1))
        std_color = np.std(img_np, axis=(0, 1))

        # Filters out images with very high contrast or abnormal RGB balance
        # Relaxed thresholds (80 and 40)
        if std_color.mean() > 80 or mean_color[0] < 40:
            return False
        return True
    except Exception as e:
        st.warning(f"Image validation warning: {str(e)}")
        return True  # Accept image on error


#Analysis Panel (Explainable AI - Grad-CAM)
def run_analysis():
    st.title("📊 Analysis Panel — Explainable AI (Grad-CAM)")
    st.write("""
    This panel visualizes which parts of the histopathology image  
    the model focused on during classification using Grad-CAM heatmaps.
    """)

    #model loading
    model = load_model()

    #Check Image from Prediction Page
    if "last_uploaded_image" in st.session_state:
        image_data = st.session_state["last_uploaded_image"]
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        st.markdown("### 📎 Image from Prediction Page")
        st.image(image, width=300)
    else:
        image = None

    # Alternative: Upload New Image
    uploaded_file = st.file_uploader(
        "📁 Or upload another image for Grad-CAM visualization:", 
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🩺 Uploaded Image", width=300)

    if image is None:
        st.warning("⚠️ No image available. Please upload or select one from Prediction page.")
        return

    # Histopathology similarity check
    if not is_histopathology_like(image):
        st.error("🚫 The provided image does not appear to be a histopathology tissue sample.")
        st.info("Please upload microscope-level biopsy images similar to the BreakHis dataset.")
        return
    
    # Grad-CAM Visualization
    st.subheader("🎯 Grad-CAM Visualization")
    
    # Initialize session state for Grad-CAM result
    if 'gradcam_result' not in st.session_state:
        st.session_state.gradcam_result = None
    if 'gradcam_base_image' not in st.session_state:
        st.session_state.gradcam_base_image = None
    
    if st.button("🧠 Generate Grad-CAM Heatmap"):
        try:
            with st.spinner("Generating Grad-CAM heatmap..."):
                temp_path = "temp_analysis_image.png"
                image.save(temp_path)
                gradcam_img, _ = generate_gradcam(
                    model, temp_path, 
                    target_layer_name="conv4", 
                    device=device
                )
                print(f"[DEBUG] Grad-CAM size: {gradcam_img.size}")
                os.remove(temp_path)
                
                # Store in session state
                st.session_state.gradcam_result = gradcam_img
                st.session_state.gradcam_base_image = image
                st.success("✅ Grad-CAM heatmap generated!")
        except Exception as e:
            st.error(f"❌ Grad-CAM generation error: {str(e)}")
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # Display with transparency slider (only if Grad-CAM exists)
    if st.session_state.gradcam_result is not None:
        alpha = st.slider(
            "Adjust heatmap transparency:", 
            0.1, 1.0, 0.6, step=0.05,
            key="gradcam_alpha"
        )
        try:
            gradcam_img = st.session_state.gradcam_result
            base_img = st.session_state.gradcam_base_image
            
            # Ensure RGBA and same size
            gradcam_rgba = gradcam_img.convert("RGBA")
            base_rgba = base_img.resize(gradcam_rgba.size).convert("RGBA")
            blended = Image.blend(base_rgba, gradcam_rgba, alpha=alpha)

            st.image(
                blended, 
                caption=f"🔥 Grad-CAM Overlay (α={alpha:.2f})", 
                width=400
            )

            with st.expander("🔍 View Full Size"):
                st.image(blended, use_container_width=True)
        except Exception as e:
            st.error(f"❌ Display error: {str(e)}")

    st.caption("🧬 DeepBreast: AI-Based Breast Cancer Detection — Explainable Visualization Module")
