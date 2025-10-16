import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import time, io
from model import BreastCancerCNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", map_location=device))
    model.eval()
    return model


def run_prediction():
    st.title("🔍 Breast Cancer Prediction")
    st.write("Upload a histopathology image to predict whether it’s **Benign** or **Malignant**.")

    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    uploaded_file = st.file_uploader("📁 Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        # ----------------------------
        # 🖼️ Küçük önizleme
        # ----------------------------
        st.image(image, caption="🩺 Uploaded Image", width=320)

        # ----------------------------
        # 🔍 Gri arka planlı Full View
        # ----------------------------
        with st.expander("🔍 View Full Size"):
            st.markdown(
                """
                <div style="
                    background-color:#2e2e2e;
                    padding:15px;
                    border-radius:10px;
                    border:1px solid #444;
                    text-align:center;">
                """,
                unsafe_allow_html=True
            )
            st.image(image, caption="Full Resolution View", use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # ----------------------------
        # 📦 Session State'e Kaydet
        # ----------------------------
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        st.session_state["last_uploaded_image"] = img_bytes.getvalue()

        # ----------------------------
        # 🧠 Tahmin
        # ----------------------------
        if st.button("🔍 Predict"):
            start = time.time()
            img_tensor = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)[0]
                pred = torch.argmax(probs).item()
                conf = probs[pred].item() * 100
            end = time.time()

            classes = ["Benign", "Malignant"]
            result = classes[pred]

            # Session’a tahmin ve güveni kaydet
            st.session_state["last_prediction"] = result
            st.session_state["last_confidence"] = conf

            # Sonuç gösterimi
            st.subheader("🔎 Prediction Result")
            if result == "Benign":
                st.success(f"✅ {result}")
            else:
                st.error(f"⚠️ {result}")

            st.info(f"Confidence: **{conf:.2f}%**")
            st.info(f"Prediction Time: **{end - start:.3f}s**")
