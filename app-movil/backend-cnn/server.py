from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
import base64
from io import BytesIO
import os
import cv2

# DICOM
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# Pillow
from PIL import Image, UnidentifiedImageError, ImageOps

# =========================
#  Flask + CORS
# =========================
app = Flask(__name__)
CORS(app)

# =========================
#  TFLite
# =========================
MODEL_PATH = "./model/model.tflite"
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# =========================
#  Configuración / Config
# =========================
MAX_IMAGE_BYTES = 50 * 1024 * 1024  # admite DICOM grandes (50MB)
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}  # ráster

# =========================
#  Utilidades base64
# =========================
def _strip_data_url(b64_str: str) -> str:
    """Quita 'data:<mime>;base64,' si viene como data URL."""
    if b64_str.startswith("data:"):
        try:
            return b64_str.split(",", 1)[1]
        except ValueError:
            return b64_str
    return b64_str

def _safe_b64decode(b64_str: str) -> bytes:
    """Decodifica base64 tolerando espacios y corrigiendo padding."""
    s = _strip_data_url(b64_str).strip().replace("\n", "").replace("\r", "")
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing
    return base64.b64decode(s, validate=False)

# =========================
#  Detección / carga DICOM
#  Detection / DICOM load
# =========================
def _is_dicom_bytes(buf: bytes) -> bool:
    """Heurística para detectar si un buffer es DICOM."""
    if len(buf) > 132 and buf[128:132] == b"DICM":
        return True
    try:
        ds = pydicom.dcmread(BytesIO(buf), stop_before_pixels=True, force=True)
        return bool(ds.file_meta) or ("SOPClassUID" in ds)
    except Exception:
        return False

def _dicom_to_rgb_array(buf: bytes) -> np.ndarray:
    """
    Carga DICOM desde bytes y devuelve np.uint8 shape [H,W,3] en RGB.
    - Aplica Rescale Slope/Intercept (pydicom lo hace vía pixel_array)
    - Aplica VOI LUT / windowing si está disponible
    - Invierte MONOCHROME1
    - Normaliza a 8-bit de forma robusta

    Loads DICOM from bytes and returns np.uint8 array with shape [H,W,3] in RGB.
    - Applies Rescale Slope/Intercept (handled by pydicom via pixel_array)
    - Applies VOI LUT / windowing if available
    - Inverts MONOCHROME1
    - Normalizes to 8-bit in a robust way

    """
    ds = pydicom.dcmread(BytesIO(buf), force=True)
    arr = ds.pixel_array

    # VOI LUT / windowing
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    if arr.ndim == 3:
        arr = arr[0]

    # MONOCHROME1
    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    arr = arr.astype(np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)  # [H,W] uint8

    # A RGB
    rgb = np.repeat(arr[..., None], 3, axis=2)  # [H,W,3]
    return rgb

# =========================
#  Carga ráster (PNG/JPG/TIFF/…)
# =========================
def _raster_to_rgb_array(buf: bytes) -> np.ndarray:
    """
    Abre imagen ráster (PNG/JPEG/TIFF/BMP/WEBP/GIF) con Pillow
    y devuelve np.uint8 [H,W,3] en RGB. Soporta TIFF 16-bit.

    Opens raster image (PNG/JPEG/TIFF/BMP/WEBP/GIF) with Pillow
    and returns np.uint8 [H,W,3] in RGB. Supports 16-bit TIFF.
    """
    try:
        img = Image.open(BytesIO(buf))
    except UnidentifiedImageError:
        raise ValueError("Unsupported or corrupted image file")

    fmt = (img.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {fmt or 'UNKNOWN'}")
    
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # TIFF 16-bit → escalar a 8-bit
    if img.mode.startswith("I;16") or (img.mode == "I"):
        arr16 = np.array(img, dtype=np.uint16)
        maxv = int(arr16.max()) if arr16.size > 0 else 0
        if maxv > 0:
            arr8 = (arr16.astype(np.float32) / maxv * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr16, dtype=np.uint8)
        img = Image.fromarray(arr8, mode="L").convert("RGB")
    else:
        if img.mode != "RGB":
            img = img.convert("RGB")

    rgb = np.array(img, dtype=np.uint8)  # [H,W,3]
    return rgb

# =========================
#  Preprocesamiento unificado
# =========================
def preprocess_for_tflite(rgb: np.ndarray, target_size=(224, 224)) -> np.ndarray:
    """
    Recibe np.uint8 [H,W,3] RGB y devuelve tensor listo para TFLite:
    - resize a target_size
    - normaliza si input dtype del modelo es float32 (divide por 255)
    - shape final: [1,H,W,3]

    Receives np.uint8 [H,W,3] RGB and returns a tensor ready for TFLite:
    - resize to target_size
    - normalize if the model's input dtype is float32 (divide by 255)
    - final shape: [1,H,W,3]
    """
    # Resize con OpenCV
    img = cv2.resize(rgb, target_size, interpolation=cv2.INTER_AREA)

    input_details = interpreter.get_input_details()[0]
    in_dtype = input_details['dtype']  # np.float32 o np.uint8
    img = img.astype(in_dtype)

    if in_dtype == np.float32:
        img = img / 255.0  # normalización típica

    img = np.expand_dims(img, axis=0)  # [1,H,W,3]
    return img

# =========================
#  Core: procesa base64 (DICOM o imagen) y ejecuta el modelo
# =========================
def process_image(image_data: str) -> float:
    """
    Decodes and preprocesses a base64 image (DICOM or raster) and performs inference using the TFLite model.
    
    Decodifica y preprocesa una imagen en base64 (DICOM o ráster) y realiza inferencia con el modelo TFLite.

    """
    # 1) base64 -> bytes
    img_bytes = _safe_b64decode(image_data)
    if len(img_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds maximum size")

    # 2) detectar DICOM vs ráster
    if _is_dicom_bytes(img_bytes):
        rgb = _dicom_to_rgb_array(img_bytes)     # [H,W,3] uint8
    else:
        rgb = _raster_to_rgb_array(img_bytes)    # [H,W,3] uint8

    # 3) preprocesado al tamaño/dtype del modelo
    input_tensor = preprocess_for_tflite(rgb, target_size=(224, 224))

    # 4) inferencia TFLite
    input_index = interpreter.get_input_details()[0]['index']
    interpreter.set_tensor(input_index, input_tensor)
    interpreter.invoke()

    output_index = interpreter.get_output_details()[0]['index']
    prediction = interpreter.get_tensor(output_index)[0]

    # Si el modelo devuelve un escalar o un array de 1 elemento
    if isinstance(prediction, (np.ndarray, list)) and np.size(prediction) == 1:
        probability = float(np.ravel(prediction)[0])
    else:
        # Si devuelve vector de clases, asume 1er valor como prob TB (ajusta según tu modelo)
        probability = float(prediction[0])

    return probability

# =========================
#  Endpoints
# =========================
@app.route('/analyze-image', methods=['POST'])
def analyze_image():
    """
    Receives a POST request with a base64 image (DICOM or raster), processes it, and returns the TB prediction.
    
    Recibe una solicitud POST con una imagen en base64 (DICOM o ráster),
    la procesa y devuelve la predicción de tuberculosis.
cH
    """
    try:
        data = request.get_json(silent=True) or {}
        image_data = data.get('image')
        if not image_data:
            return jsonify({"error": "Missing image base64"}), 400

        probability = process_image(image_data)

        threshold = 0.17
        diagnosis = "Alta probabilidad de Tuberculosis" if probability >= threshold else "Baja probabilidad de Tuberculosis"

        return jsonify({
            "probability": round(probability * 100.0, 2),
            "diagnosis": diagnosis
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/ping', methods=['GET'])
def ping():
    """
    Basic health check endpoint.
    Endpoint de prueba para verificar que el servidor está activo.
    """
    return jsonify({"message": "Servidor Flask activo ✅"}), 200

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
