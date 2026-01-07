# PDF Preprocessing and FAISS Indexing for Tuberculosis Knowledge Retrieval

> This script processes translated medical PDFs, splits them into semantic chunks, generates multilingual embeddings using Sentence Transformers, and indexes them using FAISS for later semantic search and chatbot integration (RAG).

---

## Descripción (Español)

Este script forma parte del backend de recomendación médica **tb-app-server**, repositorio disponible en este perfil. Toma archivos `.txt` previamente traducidos desde PDFs sobre tuberculosis, los divide en fragmentos, genera embeddings multilingües y construye un índice FAISS para búsqueda semántica en un chatbot con DeepSeek.

---

## Tecnologías usadas / Technologies used

- `sentence-transformers`: para generar embeddings semánticos
- `langchain`: para dividir el texto en chunks optimizados
- `faiss`: para indexar vectores y permitir búsqueda rápida
- `pymupdf` y `deep-translator` (no usados directamente aquí, pero formaron parte del flujo general de traducción PDF)

---

## Estructura del proyecto / Project structure

```
preprocess_pdfs/
│
├── pdfs/                         # Archivos PDF originales
├── preprocess_pdf/              # Archivos .txt traducidos
├── scripts/
│   ├── pdf_to_text.py                  # Extrae texto desde PDF
│   ├── translate_to_spanish.py        # Traduce los textos al español
├── faiss_index.bin          # Índice FAISS guardado
├── faiss_chunks.pkl         # Chunks usados en el índice
├── main.ipynb                   # Notebook principal
```

---

## Instalación rápida / Quick Setup

1. **Instalar dependencias**
```bash
pip install faiss-cpu sentence-transformers langchain pymupdf deep-translator
```

2. **Ejecutar el notebook `main.ipynb`**
   Asegúrate de tener los archivos `file_translated*.txt` dentro de `preprocess_pdf/`.

---

## Proceso explicado

1. **Combina los archivos .txt** traducidos.
2. **Divide el texto en chunks** de 500 caracteres con 100 de solapamiento.
3. **Genera embeddings** con `distiluse-base-multilingual-cased`.
4. **Crea el índice FAISS** y lo guarda junto con los chunks originales.

