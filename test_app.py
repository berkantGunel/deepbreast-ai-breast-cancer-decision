"""Minimal test app to isolate the issue"""
import streamlit as st
from PIL import Image

st.title("🔍 Minimal Test - Image Upload")

st.write("Testing basic image upload without model...")

uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.success("✅ File uploaded successfully!")
    st.write(f"Filename: {uploaded_file.name}")
    st.write(f"Size: {uploaded_file.size} bytes")
    
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.write(f"Image dimensions: {image.size}")
        
        st.image(image, caption="Uploaded Image", width=300)
        st.success("✅ Image displayed successfully!")
        
        if st.button("🔍 Test Button"):
            st.success("✅ Button works!")
            st.balloons()
            
    except Exception as e:
        st.error(f"❌ Error: {e}")
        import traceback
        st.code(traceback.format_exc())
