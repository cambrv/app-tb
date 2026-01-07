from deep_translator import GoogleTranslator

# Cargar el texto en inglés
with open("file.txt", "r", encoding="utf-8") as f:
    texto_ingles = f.read()

# Dividir en bloques de máximo 4000 caracteres
def dividir_en_bloques(texto, tamaño=4000):
    bloques = []
    inicio = 0
    while inicio < len(texto):
        fin = inicio + tamaño
        if fin < len(texto):
            # Cortar en el último punto o salto de línea antes del límite
            corte = max(texto.rfind('.', inicio, fin), texto.rfind('\n', inicio, fin))
            if corte == -1:
                corte = fin
        else:
            corte = len(texto)
        bloques.append(texto[inicio:corte].strip())
        inicio = corte
    return bloques

bloques = dividir_en_bloques(texto_ingles)

# Traducir cada bloque
traductor = GoogleTranslator(source='en', target='es')
texto_traducido = ""

for i, bloque in enumerate(bloques):
    try:
        traducido = traductor.translate(bloque)
        texto_traducido += traducido + "\n\n"
        print(f"✅ Traducido bloque {i+1}/{len(bloques)}")
    except Exception as e:
        print(f"❌ Error al traducir bloque {i+1}: {e}")
        texto_traducido += "[ERROR EN ESTE BLOQUE]\n\n"

# Guardar traducción completa
with open("file_translated3.txt", "w", encoding="utf-8") as f:
    f.write(texto_traducido)

print("✅ Traducción final guardada.")
