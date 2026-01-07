import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -------------------------------
# CONFIGURACIÓN Y MODELO
# -------------------------------
MODEL_PATH = "./models/modelo_tb_densenet121.h5"
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    # Redimensionar
    img = image.resize((224, 224))
    img = np.array(img)

    # Verificar formato de canales
    if img.ndim == 2:
        st.warning("La imagen está en escala de grises. Convirtiendo a RGB.")
        img = np.stack((img,) * 3, axis=-1)
    elif img.shape[-1] == 4:
        st.warning("La imagen tiene canal alpha. Eliminando canal adicional.")
        img = img[:, :, :3]

    # Normalizar
    img = img / 255.0

    # Expandir dimensión batch
    img = np.expand_dims(img, axis=0)

    # Validación final
    if img.shape != (1, 224, 224, 3):
        st.error(f"La imagen no tiene el formato correcto. Shape recibido: {img.shape}")
        return None

    return img

# -------------------------------
# INTERFAZ STREAMLIT
# -------------------------------
st.set_page_config(page_title="Detección de Tuberculosis", layout="centered")
st.title("🩺 Detección de Tuberculosis con IA")
st.write("Sube una radiografía de tórax y el modelo clasificará si es TB o Normal.")

uploaded_file = st.file_uploader("Selecciona una imagen radiográfica", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Radiografía cargada", use_column_width=True)

    img_array = preprocess_image(image)

    if img_array is not None:
        prediction = float(model.predict(img_array)[0][0])
        tb_confidence = prediction
        normal_confidence = 1 - prediction
        st.markdown("### 🧪 Nivel de confianza del modelo")
        st.progress(int(normal_confidence * 100))  # o tb_confidence si es positivo
        st.write(f"🔬 Confianza en TB: {tb_confidence:.2%}")
        st.write(f"🫁 Confianza en Normal: {normal_confidence:.2%}")
        if prediction > 0.5:
            st.error("🔴 Resultado: Posible Tuberculosis")
        else:
            st.success("🟢 Resultado: Radiografía Normal")
