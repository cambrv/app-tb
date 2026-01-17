# test_comp.py
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
TEST_DIR      = '../../dataset/test'     
IMG_SIZE      = (224, 224)
BATCH_SIZE    = 32

# Nombres para el modelo comparativo
MODEL_CANDIDATES = ['tbnet_model_comp_best.h5', 'tbnet_model_comp_last.h5']
HISTORY_PKL      = 'history_tbnet_comp_40ep.pkl'

def _load_first_existing(candidates):
    for p in candidates:
        if os.path.exists(p):
            print(f'✅ Cargando modelo: {p}')
            return load_model(p)
    raise FileNotFoundError(f"No se encontró ninguno de: {candidates}")

model = _load_first_existing(MODEL_CANDIDATES)

# ---------------------------
# GENERADOR TEST
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
target_names = [idx2class.get(0, 'class0'), idx2class.get(1, 'class1')]
report = classification_report(y_true, y_pred, target_names=target_names, digits=2)
cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)

# ---------------------------
# GRÁFICO: MATRIZ DE CONFUSIÓN (formato publicación)
# ---------------------------
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Ajustes tipográficos generales (puedes cambiar a 'Times New Roman' si tu revista lo pide)
mpl.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
})

def plot_confusion_matrix_pub(cm, class_names, normalize=False, fname_base="comp_confusion_matrix_pub"):
    # Crear carpeta de salida opcional
    os.makedirs("figures", exist_ok=True)

    if normalize:
        with np.errstate(all='ignore'):
            cm_plot = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        title = "Matriz de confusión (normalizada)"
        fmt_cell = lambda v, raw: f"{v*100:0.1f}%"  # solo porcentaje
        suffix = "_norm"
    else:
        cm_plot = cm
        title = "Matriz de confusión"
        fmt_cell = lambda v, raw: f"{raw:d}\n({v*100:0.1f}%)"  # conteo + %
        suffix = "_abs"

    # Figura cuadrada, alto DPI para impresión
    fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=300)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")

    # Ejes y etiquetas
    ax.set_xlabel("Etiqueta predicha")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(np.arange(len(class_names)), labels=class_names, rotation=0)
    ax.set_yticks(np.arange(len(class_names)), labels=class_names)

    # Barra de color discreta y sobria
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=9)

    # Anotaciones (conteo + % para abs; solo % para norm)
    raw_totals = cm.sum(axis=1, keepdims=True).astype(float)
    thresh = cm_plot.max() * 0.5 if np.isfinite(cm_plot.max()) else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            # valor normalizado para formateo
            v_norm = (cm[i, j] / raw_totals[i, 0]) if raw_totals[i, 0] > 0 else 0.0
            text = fmt_cell(v_norm, int(cm[i, j]))
            ax.text(
                j, i, text,
                ha="center", va="center",
                color="white" if cm_plot[i, j] > thresh else "black",
                fontsize=9
            )

    # Cuadrícula ligera y bordes
    ax.set_title(title, pad=10)
    ax.set_xlim(-0.5, len(class_names)-0.5)
    ax.set_ylim(len(class_names)-0.5, -0.5)
    ax.spines[:].set_visible(False)
    ax.set_aspect("equal")
    ax.grid(False)
    fig.tight_layout()

    # Guardados en formatos de revista
    png_path = os.path.join("figures", f"{fname_base}{suffix}.png")
    pdf_path = os.path.join("figures", f"{fname_base}{suffix}.pdf")
    svg_path = os.path.join("figures", f"{fname_base}{suffix}.svg")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(svg_path, bbox_inches="tight")
    print(f"💾 Guardado: {png_path}\n💾 Guardado: {pdf_path}\n💾 Guardado: {svg_path}")
    plt.show()

# Etiquetas ordenadas por índice de clase
classes = [idx2class.get(0, "normal"), idx2class.get(1, "tb")]

# Versiones: absoluta (conteo + %) y normalizada (solo %)
plot_confusion_matrix_pub(cm, classes, normalize=False, fname_base="comp_confusion_matrix")
plot_confusion_matrix_pub(cm, classes, normalize=True,  fname_base="comp_confusion_matrix")

print("\n===== CLASSIFICATION REPORT =====\n")
print(report)
print("===== CONFUSION MATRIX =====")
print(cm)
print(f"Accuracy: {acc:.4f}")

# Guardar métricas en archivo de texto (opcional)
with open('test_comp_classification_report.txt', 'w', encoding='utf-8') as f:
    f.write("===== CLASSIFICATION REPORT =====\n")
    f.write(report + "\n")
    f.write("===== CONFUSION MATRIX =====\n")
    for row in cm:
        f.write(' '.join(map(str, row)) + '\n')
    f.write(f"\nAccuracy: {acc:.4f}\n")

# ---------------------------
# CURVAS (loss y accuracy)
# ---------------------------
def _get_hist(history_dict):
    if os.path.exists(history_dict):
        with open(history_dict, 'rb') as f:
            return pickle.load(f)
    # fallback: detectar un .pkl en carpeta
    for p in os.listdir('.'):
        if p.endswith('.pkl'):
            print(f"⚠️ {history_dict} no encontrado. Usando: {p}")
            with open(p, 'rb') as f:
                return pickle.load(f)
    return None

hist = _get_hist(HISTORY_PKL)

def _get_hist_key(h, primary, alt=None):
    if primary in h: return primary
    if alt and alt in h: return alt
    return None

if isinstance(hist, dict):
    loss_k  = _get_hist_key(hist, 'loss')
    vloss_k = _get_hist_key(hist, 'val_loss')
    acc_k   = _get_hist_key(hist, 'accuracy', 'acc')
    vacc_k  = _get_hist_key(hist, 'val_accuracy', 'val_acc')

    if loss_k and vloss_k:
        epochs = range(1, len(hist[loss_k]) + 1)
        # ---- Curva de pérdida
        plt.figure()
        plt.plot(epochs, hist[loss_k], label='train_loss')
        plt.plot(epochs, hist[vloss_k], label='val_loss')
        plt.xlabel('Épocas')
        plt.ylabel('Pérdida')
        plt.title('Curva de Pérdida (Train vs Val) - TBNet Comparativo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('comp_curve_loss.png', dpi=150)
        print("💾 Guardado: comp_curve_loss.png")
        plt.show()
    else:
        print("⚠️ No se encontraron claves de pérdida en el historial para graficar.")

    if acc_k and vacc_k:
        epochs = range(1, len(hist[acc_k]) + 1)
        # ---- Curva de precisión
        plt.figure()
        plt.plot(epochs, hist[acc_k], label='train_acc')
        plt.plot(epochs, hist[vacc_k], label='val_acc')
        plt.xlabel('Épocas')
        plt.ylabel('Precisión')
        plt.title('Curva de Precisión (Train vs Val) - TBNet Comparativo')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('comp_curve_accuracy.png', dpi=150)
        print("💾 Guardado: comp_curve_accuracy.png")
        plt.show()
    else:
        print("⚠️ No se encontraron claves de precisión en el historial para graficar.")
else:
    print("⚠️ No hay historial disponible para graficar curvas.")
