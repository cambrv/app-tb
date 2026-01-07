# Chat-Service: RAG-based Medical Recommendation System using DeepSeek via OpenRouter

> 🔍 This microservice provides tuberculosis-related recommendations based on semantic search and natural language processing, using FAISS, Sentence Transformers, and DeepSeek via OpenRouter API.

---

## Descripción (Español)

Este servicio web está diseñado como parte de una aplicación de diagnóstico médico, enfocada en la tuberculosis. Utiliza Recuperación aumentada por generación (RAG) para buscar información relevante en un PDF médico preprocesado y genera respuestas usando el modelo DeepSeek a través de OpenRouter.

---

## Features / Funcionalidades

- Recuperación semántica con FAISS
- Respuestas generadas por IA (DeepSeek)
- Multilenguaje (modelo de embeddings multilingüe)
- Endpoint REST listo para producción
- Compatible con Render, Replit o despliegue local

---

## Estructura del proyecto / Project structure
```
chat-service/
│
├── server.py               # Código principal del servidor Flask
├── requirements.txt        # Dependencias del proyecto
├── .env                    # Variables de entorno (API keys)
├── render.yaml             # Configuración de despliegue para Render
├── resources/
│   ├── faiss_index.bin     # Índice vectorial de FAISS
│   └── faiss_chunks.pkl    # Fragmentos de texto embebidos
```

---

## ⚙️ Instalación rápida / Quick Setup

1. **Clona el repositorio**
```bash
git clone https://github.com/tu_usuario/chat-service.git
cd chat-service
```

2. **Instala dependencias**
```bash
pip install -r requirements.txt
```

3. **Crea un archivo `.env` con tu API Key**
```env
OPENROUTER_API_KEY=sk-xxxxxx
```

4. **Ejecuta el servidor**
```bash
python server.py
```

---

## Endpoints

### ✅ `GET /ping`
- Verifica que el servicio esté activo.
- Respuesta:
```json
{
  "message": "Chat backend (DeepSeek via OpenRouter) activo ✅"
}
```

### 💬 `POST /recommendation`
- Genera una recomendación médica basada en la pregunta enviada.

**Body JSON:**
```json
{
  "question": "¿Cuáles son los síntomas de la tuberculosis?",
  "custom_prompt": "Opcional: sobrescribir el prompt base"
}
```

**Respuesta:**
```json
{
  "answer": "La tuberculosis puede presentar tos persistente, fiebre, sudores nocturnos y pérdida de peso."
}
```

---

## Requisitos técnicos / Requirements

- Python 3.10
- OpenRouter API Key
- FAISS y Sentence Transformers

---

## Despliegue en Render (opcional)

Este proyecto incluye un archivo `render.yaml` para ser desplegado fácilmente en [Render](https://render.com/).

---

## Licencia

MIT License.  
Este microservicio puede reutilizarse con fines académicos o clínicos no comerciales.

---

## Autor

**Nombre del autor:** Camily Bravo Flores
**Tesis:** *Desarrollo de Aplicación Móvil para el Diagnóstico Temprano de Tuberculosis Integrando CNN y PLN*  
**Institución:** Universidad Técnica de Machala
**Correo electrónico:** [camilybravo@gmail.com]