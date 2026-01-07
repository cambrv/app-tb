# X-Ray Image Validator using OpenAI CLIP

> ⚙️ This microservice uses OpenAI’s CLIP model to verify whether a given image is a chest X-ray or not. It runs on Flask and supports base64 image input for easy integration with mobile or web apps.

---

## Descripción (Español)

Este microservicio utiliza el modelo CLIP de OpenAI para verificar si una imagen dada corresponde a una radiografía de tórax. Recibe imágenes en base64, compara semánticamente contra distintas clases (como “selfie”, “paisaje”, etc.) y devuelve la probabilidad de que sea una radiografía.

---

## Tecnologías utilizadas

- `CLIP` (OpenAI): modelo preentrenado para visión y lenguaje
- `Flask`: servidor ligero en Python
- `torch`: backend para deep learning
- `Pillow`: procesamiento de imágenes
- `base64`: manejo de imágenes codificadas

---

## Instalación

1. **Instalar dependencias**
```bash
pip install flask flask-cors torch Pillow
pip install git+https://github.com/openai/CLIP.git
```

2. **Ejecutar el servidor**
```bash
python server.py
```

---

## Endpoints

### ✅ `POST /validate-xray`

**Entrada:**
```json
{
  "image": "<base64 string>"
}
```

**Salida:**
```json
{
  "is_xray": true,
  "score": 0.8635,
  "all_probs": {
    "a chest X-ray image": 0.86,
    "a dog photo": 0.04,
    "a selfie": 0.01,
    ...
  }
}
```
