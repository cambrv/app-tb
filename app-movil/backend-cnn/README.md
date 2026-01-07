# Diagnosis-Service: Tuberculosis Detection API (TFLite + Flask)

> ⚙️ This lightweight Flask API takes a chest X-ray (base64 encoded), processes it, and returns a probability score indicating potential tuberculosis, using a TensorFlow Lite model.

---

## Descripción (Español)

Este microservicio permite analizar radiografías de tórax codificadas en base64 y devolver la probabilidad de tuberculosis, utilizando un modelo ligero de TensorFlow Lite. Es ideal para integrarse en aplicaciones móviles o sistemas web de diagnóstico.

---

## Features / Funcionalidades

- Recibe imágenes codificadas en base64
- Preprocesamiento automático con OpenCV
- Inferencia con modelo `.tflite`
- Rápido, ligero y portable
- Compatible con Render, Replit o local

---

## Estructura del proyecto / Project structure

```
diagnosis-service/
│
├── server.py              # Código del backend Flask
├── model/
│   └── model.tflite       # Modelo TFLite entrenado
├── .env                   # Variables de entorno
├── requirements.txt       # Dependencias del proyecto
├── render.yaml            # Configuración de despliegue para Render
```

---

## ⚙️ Instalación rápida / Quick Setup

1. **Clona el repositorio**
```bash
git clone https://github.com/tuusuario/diagnosis-service.git
cd diagnosis-service
```

2. **Instala dependencias**
```bash
pip install -r requirements.txt
```

3. **Ejecuta el servidor**
```bash
python server.py
```

---

## Endpoints

### ✅ `GET /ping`
- Prueba rápida para verificar si el servidor está activo.

**Respuesta:**
```json
{
  "message": "Servidor Flask activo ✅"
}
```

### 📤 `POST /analyze-image`
- Procesa una imagen y devuelve el diagnóstico de tuberculosis.

**Body JSON:**
```json
{
  "image": "<base64-encoded image>"
}
```

**Respuesta:**
```json
{
  "probability": 86.24,
  "diagnosis": "Alta probabilidad de Tuberculosis"
}
```

---

## Requisitos técnicos / Requirements

- Python 3.10
- TensorFlow Lite Runtime
- OpenCV
- Flask + CORS

---

## Despliegue en Render

Este proyecto incluye un `render.yaml` para desplegar fácilmente en [Render](https://render.com/).

---

## Autor

**Nombre del autor:** Camily Bravo Flores
**Tema:** *Desarrollo de Aplicación Móvil para el Diagnóstico Temprano de Tuberculosis Integrando CNN y PLN*  
**Universidad:** Universidad Técnica de Machala
**Correo electrónico:** [camilybravo@gmail.com]

---
