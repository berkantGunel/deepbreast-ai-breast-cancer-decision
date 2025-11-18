"""Streamlit prediction module responsible for loading the trained CNN,
validating uploaded images, and presenting inference feedback."""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import time, io
from src.core.model import BreastCancerCNN
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Model Yükleme
@st.cache_resource
def load_model():
    model = BreastCancerCNN().to(device)
    model.load_state_dict(torch.load("models/best_model.pth", 
                                      map_location=device, 
                                      weights_only=False))
    model.eval()
    return model

#Histopathology Benzerlik Kontrolü
def is_histopathology_like(image):
    """
    Görselin mikroskop altı dokuya benzeyip benzemediğini kontrol eder.
    Ortalama renk ve kontrast dağılımını inceleyerek
    sahne / obje / dış ortam fotoğraflarını filtreler.
    """
    try:
        # Büyük görselleri küçült (performans için)
        img_resized = image.copy()
        if max(img_resized.size) > 500:
            img_resized.thumbnail((500, 500), Image.Resampling.LANCZOS)
        
        img_np = np.array(img_resized)
        mean_color = np.mean(img_np, axis=(0, 1))
        std_color = np.std(img_np, axis=(0, 1))

        # Çok yüksek kontrastlı veya RGB dengesi anormal görselleri eler
        # Filtreyi biraz gevşettik (80 ve 40 eşikleri)
        if std_color.mean() > 80 or mean_color[0] < 40:
            return False
        return True
    except Exception as e:
        st.warning(f"Image validation warning: {str(e)}")
        return True  # Hata durumunda görseli kabul et

#Prediction Fonksiyonu
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
        try:
            with st.spinner("Loading image..."):
                image = Image.open(uploaded_file).convert("RGB")
                print(f"[DEBUG] Image loaded successfully: {image.size}")
        except Exception as e:
            print(f"[ERROR] Image loading failed: {str(e)}")
            st.error(f"❌ Error loading image: {str(e)}")
            return

        # Histopathology similarity check (optimized)
        with st.spinner("Validating image..."):
            is_valid = is_histopathology_like(image)
            print(f"[DEBUG] Histopathology validation result: {is_valid}")
            if not is_valid:
                print("[DEBUG] Image rejected by filter")
                st.warning("⚠️ The uploaded image may not be a histopathology tissue sample.")
                st.info("For best results, upload microscope-level biopsy images similar to the BreakHis dataset.")

        #Küçük önizleme
        print("[DEBUG] About to display preview image...")
        st.image(image, caption="🩺 Uploaded Image", width=320)
        print("[DEBUG] Preview image displayed successfully")

        #Gri arka planlı Full View
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
            st.image(image, caption="Full Resolution View")
            st.markdown("</div>", unsafe_allow_html=True)

        #Session State'e Kaydet
        img_bytes = io.BytesIO()
        image.save(img_bytes, format="PNG")
        st.session_state["last_uploaded_image"] = img_bytes.getvalue()

        #Tahmin
        if st.button("🔍 Predict"):
            print("[DEBUG] Predict button clicked")
            try:
                with st.spinner("Making prediction..."):
                    print("[DEBUG] Starting prediction...")
                    start = time.time()
                    img_tensor = transform(image).unsqueeze(0).to(device)
                    print(f"[DEBUG] Tensor created: {img_tensor.shape}")
                    with torch.no_grad():
                        output = model(img_tensor)
                        print(f"[DEBUG] Model output: {output}")
                        probs = torch.softmax(output, dim=1)[0]
                        pred = torch.argmax(probs).item()
                        conf = probs[pred].item() * 100
                    end = time.time()
                    print(f"[DEBUG] Prediction complete: pred={pred}, conf={conf:.2f}%")
            except Exception as e:
                print(f"[ERROR] Prediction failed: {str(e)}")
                import traceback
                traceback.print_exc()
                st.error(f"❌ Prediction error: {str(e)}")
                return

            classes = ["Benign", "Malignant"]
            result = classes[pred]

            # Session’a tahmin ve güveni kaydet
            st.session_state["last_prediction"] = result
            st.session_state["last_confidence"] = conf

            #Confidence Threshold Uyarısı
            #modelin tahmininin yeterince güvenilir sayılması için gereken minimum olasılık eşiğidir
            if conf < 60:
                st.warning("⚠️ Low confidence detected. "
                           "The input may not be a histopathology tissue sample "
                           "or may contain artifacts.")

            #Sonuç Gösterimi
            st.subheader("🔎 Prediction Result")
            if result == "Benign":
                st.success(f"✅ {result}")
            else:
                st.error(f"⚠️ {result}")

            st.info(f"Confidence: **{conf:.2f}%**")
            st.info(f"Prediction Time: **{end - start:.3f}s**")
