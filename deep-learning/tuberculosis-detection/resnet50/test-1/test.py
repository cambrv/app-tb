import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score

# ---------------------------
# 1. Configuración de Rutas y Parámetros
# ---------------------------
model_path = 'resnet_model.h5'
val_dir = '../../dataset/val'
test_dir = '../../dataset/test'
img_size = (224, 224)
batch_size = 32

# Crear carpeta de salida si no existe
os.makedirs("UV", exist_ok=True)

# ---------------------------
# 2. Cargar Modelo e Historial
# ---------------------------
model = load_model(model_path)

with open('resnet_log.pkl', 'rb') as f:
    history = pickle.load(f)

# Generadores de Imágenes
val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ---------------------------
# 3. Gráficas de Historial
# ---------------------------
plt.figure(figsize=(8, 6))
plt.plot(history['accuracy'], label='Entrenamiento')
plt.plot(history['val_accuracy'], label='Validación')
plt.title('Precisión por Época')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('UV/precision_por_epoca_resnet50_limpio.png', dpi=300)
plt.show()

plt.figure(figsize=(8, 6))
plt.plot(history['loss'], label='Entrenamiento')
plt.plot(history['val_loss'], label='Validación')
plt.title('Pérdida por Época')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()
plt.grid(False)
plt.tight_layout()
plt.savefig('UV/perdida_por_epoca_resnet50_limpio.png', dpi=300)
plt.show()

# ---------------------------
# 4. Optimización de Umbral
# ---------------------------
y_pred_probs_val = model.predict(val_generator).ravel()
y_true_val = val_generator.classes

thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1 = -1.0

for th in thresholds:
    preds_val = (y_pred_probs_val > th).astype(int)
    rec = recall_score(y_true_val, preds_val, zero_division=0)
    f1  = f1_score(y_true_val, preds_val, zero_division=0)
    if rec >= 0.90 and f1 > best_f1:
        best_threshold = th
        best_f1 = f1

if best_f1 < 0:
    f1s = [f1_score(y_true_val, (y_pred_probs_val > th).astype(int), zero_division=0) for th in thresholds]
    best_threshold = thresholds[int(np.argmax(f1s))]

print(f"Mejor Umbral seleccionado desde VALIDACIÓN: {best_threshold:.2f}")

# ---------------------------
# 5. Evaluación en TEST
# ---------------------------
y_pred_probs_test = model.predict(test_generator).ravel()
y_true_test = test_generator.classes
y_pred_test = (y_pred_probs_test > best_threshold).astype(int)

cm = confusion_matrix(y_true_test, y_pred_test)

plt.figure(figsize=(4, 4))
ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',      
    cbar=False,
    square=True,   
    xticklabels=[0, 1],
    yticklabels=[0, 1]
)

ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('UV/matriz_confusion_resnet50.png', dpi=300)
plt.show()
# ============================================

# ---------------------------
# 6. Métricas con Sensibilidad y Especificidad
# ---------------------------
precision = precision_score(y_true_test, y_pred_test, zero_division=0)
recall = recall_score(y_true_test, y_pred_test, zero_division=0)  # recall positivo (clase TB)
f1 = f1_score(y_true_test, y_pred_test, zero_division=0)
accuracy = accuracy_score(y_true_test, y_pred_test)
fpr, tpr, _ = roc_curve(y_true_test, y_pred_probs_test)
roc_auc = auc(fpr, tpr)

# Cálculo de Sensibilidad y Especificidad global
tn, fp, fn, tp = cm.ravel()
sensibilidad = tp / (tp + fn)
especificidad = tn / (tn + fp)

# ---------------------------
# 7. Resultados Consolidados
# ---------------------------
print("\nResultados en el conjunto de prueba:")
print(f"Mejor Umbral: {best_threshold:.4f}")
print(f"Precisión: {precision:.4f}")
print(f"Sensibilidad (Sensitivity): {sensibilidad:.4f}")
print(f"Especificidad (Specificity): {especificidad:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

# Guardar Resultados
with open("UV/resultados_resnet50.txt", "w") as f:
    f.write("Resultados en el conjunto de prueba (ResNet50)\n")
    f.write(f"Mejor Umbral (seleccionado en validación): {best_threshold:.4f}\n")
    f.write(f"Precisión: {precision:.4f}\n")
    f.write(f"Sensibilidad (Sensitivity): {sensibilidad:.4f}\n")
    f.write(f"Especificidad (Specificity): {especificidad:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Exactitud (Accuracy): {accuracy:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
    f.write("\nMatriz de Confusión:\n")
    f.write(str(cm))
