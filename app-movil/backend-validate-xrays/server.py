from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
from io import BytesIO

import numpy as np
import torch
import clip
from PIL import Image, ImageOps, UnidentifiedImageError

# DICOM
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

app = Flask(__name__)
CORS(app)

# Configuración de la imagen
# Image configuration
MAX_IMAGE_BYTES = 50 * 1024 * 1024  # 10 MB
SUPPORTED_FORMATS = {"JPEG", "PNG", "WEBP", "BMP", "GIF", "TIFF"}


# Selección del dispositivo (GPU si está disponible, si no, CPU)
# Device selection (use GPU if available, otherwise CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"

# Cargar modelo CLIP y preprocesador
# Load CLIP model and preprocessing function
model, preprocess = clip.load("ViT-B/32", device=device)

# Etiquetas personalizadas para clasificación más precisa
# Custom labels for more accurate classification
labels = [
    "a chest X-ray image",
    # Random labels
    "a cartoon of lungs",
    "a drawing of a chest X-ray",
    "a CT scan image",
    "a diagram of the respiratory system",
    "a landscape photo",
    "a selfie",
    "a dog photo"
]

# Tokenizar y obtener embeddings normalizados
# Tokenize and obtain normalized embeddings
with torch.no_grad():
    text_tokens = clip.tokenize(labels).to(device)
    text_features = model.encode_text(text_tokens)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

def _strip_data_url(b64_str: str) -> str:
    """
    Strips a 'data:<mime>;base64,' prefix from a Base64 string if present.

    Removes common data-URL headers so only the raw Base64 payload remains.

    Args:
        b64_str (str): Base64 string, optionally prefixed with a data URL header
                       like 'data:image/png;base64,<...>'.

    Returns:
        str: The Base64 content without the data URL prefix.

    ------------------------------------------------------------
    Quita el prefijo 'data:<mime>;base64,' de una cadena Base64 si existe.

    Elimina encabezados típicos de data URL para dejar solo el contenido Base64.

    Parámetros:
        b64_str (str): Cadena Base64, opcionalmente con encabezado data URL
                       como 'data:image/png;base64,<...>'.

    Retorna:
        str: Contenido Base64 sin el prefijo de data URL.
    """
    if b64_str.startswith("data:"):
        try:
            return b64_str.split(",", 1)[1]
        except ValueError:
            return b64_str
    return b64_str


def _safe_b64decode(b64_str: str) -> bytes:
    """
    Decodes a Base64 string robustly:
    - Trims whitespace and newlines
    - Accepts data URLs (strips header)
    - Auto-fixes missing padding

    Args:
        b64_str (str): Base64 string (raw or data URL).

    Returns:
        bytes: Decoded binary content.

    Notes:
        Uses 'validate=False' to be tolerant with minor Base64 issues.

    ------------------------------------------------------------
    Decodifica una cadena Base64 de forma robusta:
    - Quita espacios y saltos de línea
    - Acepta data URLs (elimina el encabezado)
    - Corrige automáticamente el padding faltante

    Parámetros:
        b64_str (str): Cadena Base64 (cruda o en data URL).

    Retorna:
        bytes: Contenido binario decodificado.

    Notas:
        Usa 'validate=False' para tolerar pequeñas inconsistencias en Base64.
    """
    s = b64_str.strip().replace("\n", "").replace("\r", "")
    if s.startswith("data:"):
        s = s.split(",", 1)[1]
    missing = (-len(s)) % 4
    if missing:
        s += "=" * missing
    return base64.b64decode(s, validate=False)

def _is_dicom_bytes(buf: bytes) -> bool:
    """
    Heuristically checks whether a byte buffer is a DICOM file.

    Strategy:
      - Quick magic check for 'DICM' at offset 128.
      - Fallback: attempt partial pydicom parse (metadata only).

    Args:
        buf (bytes): Raw file contents.

    Returns:
        bool: True if the buffer likely represents a DICOM file; False otherwise.

    ------------------------------------------------------------
    Verifica heurísticamente si un buffer de bytes es un archivo DICOM.

    Estrategia:
      - Comprobación rápida del marcador 'DICM' en el offset 128.
      - Alternativa: intentar parseo parcial con pydicom (solo metadatos).

    Parámetros:
        buf (bytes): Contenido binario del archivo.

    Retorna:
        bool: True si el buffer probablemente es DICOM; False en caso contrario.
    """
    if len(buf) > 132 and buf[128:132] == b"DICM":
        return True
    try:
        ds = pydicom.dcmread(BytesIO(buf), stop_before_pixels=True, force=True)
        return bool(ds.file_meta) or ("SOPClassUID" in ds)
    except Exception:
        return False

def _dicom_bytes_to_png_b64(buf: bytes) -> str:
    """
    Converts DICOM bytes into a grayscale PNG data URL (Base64).

    Processing steps:
      - Read with pydicom; 'pixel_array' applies rescale slope/intercept.
      - Apply VOI LUT / windowing if available.
      - If multiframe, take the middle frame.
      - Invert MONOCHROME1.
      - Robust min-max normalization to 8-bit.
      - Encode as PNG and wrap as 'data:image/png;base64,<...>'.

    Args:
        buf (bytes): Raw DICOM file contents.

    Returns:
        str: PNG data URL (grayscale) representing the DICOM image.

    Raises:
        Exception: If the DICOM cannot be parsed or pixel data is invalid.

    ------------------------------------------------------------
    Convierte bytes de DICOM en un data URL de PNG en escala de grises (Base64).

    Pasos de procesamiento:
      - Lee con pydicom; 'pixel_array' aplica rescale slope/intercept.
      - Aplica VOI LUT / windowing si existe.
      - Si es multiframe, toma el frame central.
      - Invierte MONOCHROME1.
      - Normaliza de forma robusta (min-max) a 8 bits.
      - Codifica como PNG y envuelve como 'data:image/png;base64,<...>'.

    Parámetros:
        buf (bytes): Contenido binario del archivo DICOM.

    Retorna:
        str: Data URL de PNG (escala de grises) que representa la imagen DICOM.

    Lanza:
        Exception: Si no se puede parsear el DICOM o el pixel data es inválido.
    """
    ds = pydicom.dcmread(BytesIO(buf), force=True)
    arr = ds.pixel_array  # aplica rescale slope

    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    if arr.ndim == 3:
        arr = arr[arr.shape[0] // 2]

    if getattr(ds, "PhotometricInterpretation", "").upper() == "MONOCHROME1":
        arr = np.max(arr) - arr

    # normalizar a 8-bit
    arr = arr.astype(np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr8 = (arr * 255.0).clip(0, 255).astype(np.uint8)

    img = Image.fromarray(arr8, mode="L")  # escala de grises

    out = BytesIO()
    img.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"

def _raster_bytes_to_png_b64(buf: bytes) -> str:
    """
    Converts raster image bytes to a PNG data URL (Base64).

    Supported inputs:
      - PNG, JPEG, TIFF, BMP, WEBP, GIF
      - Handles 16-bit TIFF by scaling to 8-bit.

    Processing:
      - Opens with Pillow.
      - If 16-bit (I;16 or I), converts to 8-bit grayscale.
      - Encodes as PNG and returns data URL.

    Args:
        buf (bytes): Raw raster image file contents.

    Returns:
        str: PNG data URL representing the raster image.

    Raises:
        UnidentifiedImageError: If Pillow cannot identify the image.

    ------------------------------------------------------------
    Convierte bytes de imagen ráster a un data URL de PNG (Base64).

    Entradas soportadas:
      - PNG, JPEG, TIFF, BMP, WEBP, GIF
      - Maneja TIFF de 16 bits convirtiendo a 8 bits.

    Proceso:
      - Abre con Pillow.
      - Si es de 16 bits (I;16 o I), transforma a escala de grises de 8 bits.
      - Codifica como PNG y retorna el data URL.

    Parámetros:
        buf (bytes): Contenido binario de la imagen ráster.

    Retorna:
        str: Data URL de PNG que representa la imagen.

    Lanza:
        UnidentifiedImageError: Si Pillow no puede identificar la imagen.
    """
    img = Image.open(BytesIO(buf))
    if img.mode.startswith("I;16") or img.mode == "I":
        arr16 = np.array(img, dtype=np.uint16)
        maxv = int(arr16.max()) if arr16.size > 0 else 0
        if maxv > 0:
            arr8 = (arr16.astype(np.float32)/maxv*255.0).clip(0,255).astype(np.uint8)
        else:
            arr8 = np.zeros_like(arr16, dtype=np.uint8)
        img = Image.fromarray(arr8, mode="L")
    out = BytesIO()
    img.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"



def _dicom_to_pil(buf: bytes) -> Image.Image:
    """
    Loads a DICOM file from bytes and returns a PIL RGB image ready for CLIP.

    Carga un DICOM desde bytes y devuelve una imagen PIL RGB lista para CLIP.

    Pasos:
      - Aplica RescaleSlope/Intercept (pydicom lo hace vía pixel_array).
      - Aplica VOI LUT / Windowing si está disponible.
      - Maneja MONOCHROME1 (invierte si corresponde).
      - Convierte 16-bit a 8-bit (escala) y a RGB.
    """
    ds = pydicom.dcmread(BytesIO(buf), force=True)
    # Pixel data a numpy
    arr = ds.pixel_array  # aplica rescale slope/intercept si están presentes

    # VOI LUT / Windowing si existe; mantiene bit depth
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass

    # MONOCHROME1: negros/blancos invertidos → invertimos
    photometric = getattr(ds, "PhotometricInterpretation", "").upper()
    if photometric == "MONOCHROME1":
        arr = np.max(arr) - arr

    # Si es multiframe, tomamos el primero
    if arr.ndim == 3:
        arr = arr[0]

    # Normalizamos a 8-bit (0-255) de forma robusta
    arr = arr.astype(np.float32)
    vmin, vmax = float(np.min(arr)), float(np.max(arr))
    if vmax > vmin:
        arr = (arr - vmin) / (vmax - vmin)
    else:
        arr = np.zeros_like(arr, dtype=np.float32)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)

    # Creamos PIL en escala de grises y convertimos a RGB
    pil_img = Image.fromarray(arr, mode="L").convert("RGB")
    return pil_img

def decode_base64_image(img_b64: str) -> Image.Image:
    
    """
    Decodes a base64 image payload (DICOM, PNG, JPEG, TIFF, BMP, WEBP, GIF) into a PIL RGB image.

    Decodifica una imagen en base64 (DICOM, PNG, JPEG, TIFF, BMP, WEBP, GIF)
    y la devuelve como PIL RGB.

    JSON Input:
        {
            "image": "<base64-encoded string (raw or data URL)>"
        }

    Returns:
        PIL.Image.Image (RGB)

    Raises (ValueError):
        - "Invalid base64 encoding"
        - "Image exceeds maximum size (10 MB)"
        - "Unsupported or corrupted image file"
        - "Unsupported image format: XYZ"
    """

    try:
        img_bytes = _safe_b64decode(img_b64)
    except Exception:
        raise ValueError("Invalid base64 encoding")

    if len(img_bytes) > MAX_IMAGE_BYTES:
        raise ValueError("Image exceeds maximum size (10 MB)")

    # DICOM
    if _is_dicom_bytes(img_bytes):
        try:
            return _dicom_to_pil(img_bytes)
        except Exception:
            raise ValueError("Unsupported or corrupted image file")

    # — Raster (PNG/JPEG/TIFF/etc.) —
    try:
        img = Image.open(BytesIO(img_bytes))
    except UnidentifiedImageError:
        raise ValueError("Unsupported or corrupted image file")

    fmt = (img.format or "").upper()
    if fmt not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported image format: {fmt or 'UNKNOWN'}")

    # Corrige orientación y asegura RGB
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass
    if img.mode != "RGB":
        # Si es TIFF 16-bit en 'I;16', convertimos con escalado automático a 8-bit
        if img.mode.startswith("I;16"):
            # Convertimos a np.uint16 → reescalamos → 8-bit
            arr = np.array(img, dtype=np.uint16)
            arr = (arr.astype(np.float32) / arr.max() * 255.0).clip(0, 255).astype(np.uint8) if arr.max() > 0 else arr.astype(np.uint8)
            img = Image.fromarray(arr, mode="L").convert("RGB")
        else:
            img = img.convert("RGB")

    return img

@app.route("/validate-xray", methods=["POST"])
def validate_xray():
    """
    Validates whether a base64 image is a chest X-ray using CLIP.

    Valida si una imagen base64 es una radiografía de tórax usando CLIP.

    JSON Input:
        {
            "image": "<base64-encoded string (raw or data URL)>"
        }

    JSON Success (200):
        {
            "is_xray": true,
            "score": 0.87,
            "top_label": "a chest X-ray image",
            "all_probs": { "<label>": 0.123, ... }
        }

    JSON Errors:
        400: {"error": "Missing base64 image"}
        415: {"error": "<Invalid base64 encoding | Unsupported ... | Image exceeds ... | Unsupported image format: ...>"}
        500: {"error": "<unexpected error>"}
    """
    try:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image")

        if not img_b64:
            return jsonify({"error": "Missing base64 image"}), 400

        try:
            img = decode_base64_image(img_b64)
        except ValueError as ve:
            return jsonify({"error": str(ve)}), 415

        img_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            model.eval()
            img_features = model.encode_image(img_tensor)
            img_features = img_features / img_features.norm(dim=-1, keepdim=True)
            text_feats = text_features  # ya normalizado arriba

            # Temperatura aprendida por CLIP (recupera probs altas)
            logit_scale = model.logit_scale.exp()
            logits = (logit_scale * (img_features @ text_feats.T)).squeeze(0)
            probs = logits.softmax(dim=0).cpu().numpy()

        result = {label: float(p) for label, p in zip(labels, probs)}
        score = result["a chest X-ray image"]
        top_label = labels[int(probs.argmax())]

        return jsonify({
            "is_xray": bool(score >= 0.80),  # ajusta si lo deseas (0.70–0.80)
            "score": float(score),
            "top_label": top_label,
            "all_probs": result
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/preview-image", methods=["POST"])
def preview_image():
    """
    Convierte una imagen base64 (DICOM o ráster) en una vista previa PNG (data URL).
    JSON IN:  { "image": "<base64 (raw o data URL)>" }
    JSON OUT: { "data_url": "data:image/png;base64,..." }
    """
    try:
        data = request.get_json(silent=True) or {}
        img_b64 = data.get("image")
        if not img_b64:
            return jsonify({"error": "Missing image base64"}), 400

        buf = _safe_b64decode(img_b64)
        if len(buf) > MAX_IMAGE_BYTES:
            return jsonify({"error": "Image exceeds maximum size"}), 413

        if _is_dicom_bytes(buf):
            data_url = _dicom_bytes_to_png_b64(buf)
        else:
            data_url = _raster_bytes_to_png_b64(buf)

        return jsonify({ "data_url": data_url })
    except Exception as e:
        return jsonify({ "error": str(e) }), 500
    

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "5002"))
    app.run(host=host, port=port)