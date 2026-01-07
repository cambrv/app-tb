from flask import Flask, request, jsonify, session
from flask_cors import CORS
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast
from datetime import datetime
import openai
import os
import pickle
import faiss
import numpy as np
import time
import csv

# Cargar variables de entorno desde .env
# Load environment variables from .env file
load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Configurar la API key de OpenRouter y su URL base
# Configure OpenRouter API key and base URL
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"

# Inicializar aplicación Flask y habilitar CORS
# Initialize Flask app and enable CORS
app = Flask(__name__)
CORS(app)

# Cargar el modelo multilingüe para embeddings
# Load multilingual embedding model
embedder = SentenceTransformer("distiluse-base-multilingual-cased")

# Inicializar el tokenizer (se usa el de GPT-2 como aproximación general)
# Initialize tokenizer (GPT-2 used as general-purpose approximation)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

def get_most_relevant_chunks(question, k=5):
    """
    Recupera los k fragmentos más relevantes usando FAISS y similitud semántica.
    Retrieve top-k most relevant chunks using FAISS and semantic similarity.
    """
    chunks_path = os.path.join(BASE_DIR, "resources", "faiss_chunks.pkl")
    index_path = os.path.join(BASE_DIR, "resources", "faiss_index.bin")

    with open(chunks_path, "rb") as f:
        all_chunks = pickle.load(f)
    faiss_index = faiss.read_index(index_path)

    question_embedding = embedder.encode([question])
    D, I = faiss_index.search(np.array(question_embedding), k)
    return [all_chunks[i] for i in I[0]]

def get_deepseek_via_openrouter(context, question, system_prompt):
    """
    Genera una respuesta usando el modelo DeepSeek a través de OpenRouter.
    Generate a response using DeepSeek model via OpenRouter.
    """
    prompt = f"""{system_prompt}

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""

    response = openai.ChatCompletion.create(
        model="deepseek/deepseek-chat-v3.1",
        messages=[
            {"role": "user", "content": prompt}
        ],
    )

    return response.choices[0].message["content"]

def evaluate_question(question, custom_prompt=None, k=5):
    """
    Evalúa una pregunta médica usando recuperación semántica + modelo LLM.
    Retorna respuesta, tiempo total, tiempo modelo, tokens y k.
    """
    total_start = time.time()

    # Recuperar chunks más relevantes
    relevant_chunks = get_most_relevant_chunks(question, k=k)
    context = "\n\n".join(relevant_chunks)

    # Prompt base si no se especifica otro
    base_prompt = """
Responde de forma clara, profesional, y amable en español con base en el siguiente contexto médico sobre la tuberculosis.

Debes responder en un párrafo de máximo 3-4 líneas. No antepongas la palabra "Respuesta". Evita repeticiones e introducciones innecesarias. No repitas la pregunta. No cites las fuentes.
"""
    system_prompt = custom_prompt if custom_prompt else base_prompt

    # Prompt completo para estimación de tokens
    full_prompt = f"""{system_prompt}

Contexto:
{context}

Pregunta:
{question}

Respuesta:"""

    token_count = contar_tokens(full_prompt)

    # Tiempo de inferencia con el modelo
    model_start = time.time()
    answer = get_deepseek_via_openrouter(context, question, system_prompt)  # <- sin model_name
    model_end = time.time()
    total_end = time.time()

    result = {
        "answer": answer,
        "time_total": round(total_end - total_start, 2),
        "time_model": round(model_end - model_start, 2),
        "tokens": token_count,
        "cached": False,
        "k": k
    }

    log_to_csv(question, result)
    return result

def contar_tokens(texto):
    """
    Cuenta una estimación del número de tokens usando el tokenizer GPT2.
    """
    return len(tokenizer.encode(texto))


def log_to_csv(question, result):
    log_file = os.path.join("logs", "benchmark_log.csv")
    os.makedirs("logs", exist_ok=True)

    file_exists = os.path.isfile(log_file)

    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(["Fecha", "Pregunta", "Respuesta", "Tokens", "Tiempo Total (s)", "Tiempo Modelo (s)", "Cache", "K"])
        writer.writerow([
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            question,
            result["answer"],
            result["tokens"],
            result["time_total"],
            result["time_model"],
            "Sí" if result["cached"] else "No",
            result["k"]
        ])

@app.route("/ping", methods=["GET"])
def ping():
    """
    Endpoint para verificar si el servidor está activo.
    Endpoint to check if the server is running.
    """
    return jsonify({"message": "Chat backend (DeepSeek via OpenRouter) activo ✅"}), 200

@app.route("/recommendation", methods=["POST"])
def recommendation():
    """
    Endpoint principal para recibir preguntas y retornar una recomendación médica.
    Main endpoint to receive questions and return medical recommendations.
    """
    try:
        data = request.json
        question = data.get("question", "")
        custom_prompt = data.get("custom_prompt", None)

        # Recuperar contexto relevante usando embeddings
        # Retrieve relevant context using embeddings
        relevant_chunks = get_most_relevant_chunks(question, k=5)
        context = "\n\n".join(relevant_chunks)

        # Prompt base si no se proporciona uno personalizado
        # Use base prompt if no custom prompt is provided
        base_prompt = """
Responde de forma clara, profesional, y amable en español con base en el siguiente contexto médico sobre la tuberculosis.

Responde en un párrafo corto (3-4 líneas). No antepongas la palabra \"Respuesta\". Evita repeticiones e introducciones innecesarias. No repitas la pregunta. No cites las fuentes de la información. Usa markdowns.
"""
        system_prompt = custom_prompt if custom_prompt else base_prompt

        # Obtener respuesta del modelo DeepSeek
        # Get response from DeepSeek model
        answer = get_deepseek_via_openrouter(context, question, system_prompt)

        return jsonify({"answer": answer})
    except Exception as e:
        # Retornar error como JSON
        # Return error as JSON
        return jsonify({"error": str(e)}), 500

# Ejecutar el servidor Flask localmente
# Run the Flask server locally
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
