
import sys
import os
import matplotlib.pyplot as plt
import pandas as pd
sys.path.append(os.path.abspath(".."))
from server import evaluate_question

# Lista de preguntas a evaluar
preguntas = [
    "¿Qué síntomas presenta la tuberculosis pulmonar?",
    "¿Cómo se contagia la tuberculosis?",
    "¿Cuánto dura el tratamiento contra la tuberculosis?",
]

# Valores de k a probar
valores_k = range(1, 6)

# Crear dataframe para registrar resultados
datos = []

print("Iniciando evaluación de múltiples preguntas...\n")

for pregunta in preguntas:
    print(f"\nPregunta: {pregunta}\n")

    ks = []
    tiempos_total = []
    tiempos_modelo = []
    tokens_prompt = []

    for k in valores_k:
        print(f"k = {k}")
        resultado = evaluate_question(pregunta, k=k)

        print(f"    Tiempo total: {resultado['time_total']} s | Modelo: {resultado['time_model']} s | Tokens: {resultado['tokens']} |")

        datos.append({
            "Pregunta": pregunta,
            "k": k,
            "Tiempo Total (ms)": round(resultado['time_total'] * 1000, 2),
            "Tiempo Modelo (ms)": round(resultado['time_model'] * 1000, 2),
            "Tokens Prompt Estimados": resultado['tokens'],
        })

        ks.append(k)
        tiempos_total.append(resultado['time_total'])
        tiempos_modelo.append(resultado['time_model'])
        tokens_prompt.append(resultado['tokens'])

    # Gráfica de Tiempo Total
    plt.figure(figsize=(7, 4))
    plt.plot(ks, [t * 1000 for t in tiempos_total], marker='o')
    plt.title(f"{pregunta} - Tiempo Total (ms)")
    plt.xlabel("Número de Chunks (k)")
    plt.ylabel("Tiempo Total (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfica de Tiempo del Modelo
    plt.figure(figsize=(7, 4))
    plt.plot(ks, [t * 1000 for t in tiempos_modelo], marker='o')
    plt.title(f"{pregunta} - Tiempo del Modelo (ms)")
    plt.xlabel("Número de Chunks (k)")
    plt.ylabel("Tiempo del Modelo (ms)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Gráfica de Tokens Estimados
    plt.figure(figsize=(7, 4))
    plt.plot(ks, tokens_prompt, marker='o')
    plt.title(f"{pregunta} - Tokens Estimados del Prompt")
    plt.xlabel("Número de Chunks (k)")
    plt.ylabel("Tokens Estimados")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

# Guardar resultados en CSV
df_resultados = pd.DataFrame(datos)
df_resultados.to_csv("evaluation.csv", index=False)
print("\nSaved in evaluation.csv")
