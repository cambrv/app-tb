# test.py
import os, math, pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model

# ---------------------------
# RUTAS / CONFIG
# ---------------------------
TEST_DIR      = '../../dataset/test'     # Debe tener subcarpetas: normal/ tb
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32
# Ajusta si usaste otro nombre:
MODEL_CANDIDATES = ['tbnet_model_hparam_best.h5', 'tbnet_model_hparam_last.h5', 'tbnet_model_optimized.h5']
HISTORY_PKL   = 'history_tbnet_40ep.pkl'  # Cambia si usaste otro nombre

# ---------------------------
# UTIL: cargar modelo
# ---------------------------
def _load_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            print(f'✅ Cargando modelo: {p}')
            return load_model(p)
    raise FileNotFoundError(f"No se encontró ninguno de: {candidates}")

model = _load_first_existing(MODEL_CANDIDATES)

# ---------------------------
# GENERADOR TEST (sin shuffle)
# ---------------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# Mapas de clases (esperado: {'normal':0, 'tb':1})
print("Class indices:", test_gen.class_indices)
idx2class = {v:k for k,v in test_gen.class_indices.items()}

# ---------------------------
# PREDICCIONES
# ---------------------------
steps = math.ceil(test_gen.samples / BATCH_SIZE)
probs = model.predict(test_gen, steps=steps, verbose=1).ravel()
y_pred = (probs >= 0.5).astype(int)
y_true = test_gen.classes

# ---------------------------
# MÉTRICAS
# ---------------------------
target_names = [idx2class[0], idx2class[1]] if 0 in idx2class and 1 in idx2class else ['class0','class1']
report = classification_report(y_true, y_pred, target_names=target_names, digits=2)
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

print("\n===== CLASSIFICATION REPORT =====\n")
print(report)
print("===== CONFUSION MATRIX =====")
print(cm)
print(f"Accuracy: {acc:.4f}")

# Guardar métricas en archivo de texto (opcional)
with open('test_classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report + "\n")
    f.write("===== CONFUSION MATRIX =====\n")
    np.savetxt(f, cm, fmt='%d')
    f.write(f"\nAccuracy: {acc:.4f}\n")

# ---------------------------
# CURVAS (loss y accuracy)
# ---------------------------
if os.path.exists(HISTORY_PKL):
    with open(HISTORY_PKL, 'rb') as f:
        hist = pickle.load(f)
else:
    # fallback: si guardaste con otro nombre
    # intenta detectar automáticamente un .pkl en la carpeta
    pkl_candidates = [p for p in os.listdir('.') if p.endswith('.pkl')]
    if not pkl_candidates:
        print("⚠️ No se encontró el historial .pkl; no se podrán graficar curvas.")
        hist = None
    else:
        print(f"⚠️ HISTORY_PKL no encontrado. Usando: {pkl_candidates[0]}")
        with open(pkl_candidates[0], 'rb') as f:
            hist = pickle.load(f)

def _get_hist_key(h, primary, alt):
    if primary in h: return primary
    if alt in h:     return alt
    return None

if hist is not None and isinstance(hist, dict):
    # Keys robustas (por si Keras guardó 'acc' en vez de 'accuracy')
    loss_k  = _get_hist_key(hist, 'loss', 'train_loss')
    vloss_k = _get_hist_key(hist, 'val_loss', 'val_loss')
    acc_k   = _get_hist_key(hist, 'accuracy', 'acc')  # Keras antiguos: 'acc'
    vacc_k  = _get_hist_key(hist, 'val_accuracy', 'val_acc')

    epochs = range(1, len(hist[loss_k]) + 1) if loss_k in hist else None

    # ---- Curva de pérdida
    if loss_k and vloss_k and epochs is not None:
        plt.figure()
        plt.plot(epochs, hist[loss_k], label='train_loss')
        plt.plot(epochs, hist[vloss_k], label='val_loss')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Curva de Pérdida (Train vs Val)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('curve_loss.png', dpi=150)
        print("💾 Guardado: curve_loss.png")
        plt.show()
    else:
        print("⚠️ No se encontraron claves de pérdida en el historial para graficar.")

    # ---- Curva de precisión
    if acc_k and vacc_k and epochs is not None:
        plt.figure()
        plt.plot(epochs, hist[acc_k], label='train_acc')
        plt.plot(epochs, hist[vacc_k], label='val_acc')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.title('Curva de Precisión (Train vs Val)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('curve_accuracy.png', dpi=150)
        print("💾 Guardado: curve_accuracy.png")
        plt.show()
    else:
        print("⚠️ No se encontraron claves de precisión en el historial para graficar.")
else:
    print("⚠️ No hay historial disponible para graficar curvas.")
