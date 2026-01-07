# 📱 Tuberculosis Diagnostic App - Frontend (Ionic + XP)

> 🚀 This mobile application frontend is built using Ionic Framework and follows the agile methodology Extreme Programming (XP). It serves as the user interface for uploading chest X-rays, performing tuberculosis diagnosis, and receiving medical recommendations via AI models.

---

## 📌 Descripción (Español)

Este frontend fue desarrollado con el framework **Ionic** por su capacidad multiplataforma, permitiendo a los usuarios cargar radiografías, recibir diagnósticos automáticos de tuberculosis y obtener recomendaciones médicas desde un chatbot. Se aplicó la metodología **Extreme Programming (XP)**, priorizando simplicidad, retroalimentación continua y adaptabilidad.

---

## 🌟 Funcionalidades / Features

- 📸 Carga de imágenes desde galería
- 🤖 Validación de imagen como radiografía usando CLIP (OpenAI)
- 🧠 Conexión con modelo CNN en formato TFLite para diagnóstico automático
- 💬 Chatbot con procesamiento de lenguaje natural (NLP)
- 🔁 Arquitectura modular y extensible
- ⚡ Rápida, fluida y responsiva

---

## 🧱 Estructura del proyecto / Project Structure

```
tuberculosis-app/
│
├── src/
│   ├── app/                  # Lógica principal de la aplicación
│   ├── assets/               # Recursos gráficos y estáticos
│   ├── environments/         # Configuraciones de entorno
│   └── index.html            # Entrada principal
├── package.json              # Dependencias y scripts
├── ionic.config.json         # Configuración de Ionic
└── README.md                 # Documentación del frontend
```

---

## ⚙️ Instalación rápida / Quick Setup

1. **Clona el repositorio**
```bash
git clone https://github.com/cambrv/app-tb-frontend
cd app-tb-frontend
```

2. **Instala dependencias**
```bash
npm install
```

3. **Levanta la app en modo desarrollo**
```bash
ionic serve
```

---

## 🔧 Requisitos técnicos / Requirements

- Node.js 18+
- Ionic CLI 7+
- Capacitor
- Framework: Angular / Ionic
- IDE recomendado: VS Code

---

## Related GitHub Repositories

- Diagnosis Service: https://github.com/cambrv/tb-app-server
- Chat Service: https://github.com/cambrv/ai-chat-tb
- X-Ray Image Validator: https://github.com/cambrv/validate_xrays
- PDF Preprocessing: https://github.com/cambrv/preprocess_pdfs
- Dataset training: https://github.com/cambrv/tuberculosis-detection

---

## 🧬 Autor

**Nombre del autor:** Camily Bravo Flores, Derik Aranda Neira
**Tema:** Mobile App for Tuberculosis Detection Using Deep Learning and NLP-Based Recommendations 
**Universidad:** Universidad Técnica de Machala  
**Correo electrónico:** [camilybravo@gmail.com](mailto:camilybravo@gmail.com)
