import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score, confusion_matrix

plt.style.use('default')

# Crear carpeta para guardar gráficos si no existe
os.makedirs("UV", exist_ok=True)

# ---------------------------
# 1. Configuración de Rutas
# ---------------------------
model_path = 'densenet121_model.h5'
val_dir = '../../dataset/val'
test_dir = '../../dataset/test'
img_size = (224, 224)
batch_size = 32

# ---------------------------
# 2. Cargar Modelo e Historial
# ---------------------------
model = load_model(model_path)

with open('densenet121_log.pkl', 'rb') as f:
    history = pickle.load(f)

# Generadores
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
# 3. Evaluación de Umbral
# ---------------------------
y_pred_probs = model.predict(val_generator).ravel()
y_true = val_generator.classes

thresholds = np.arange(0.1, 0.9, 0.01)
recall_values = []
f1_values = []
best_threshold = 0.5
best_f1 = 0

for threshold in thresholds:
    y_pred = (y_pred_probs > threshold).astype(int)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    recall_values.append(recall)
    f1_values.append(f1)
    if recall >= 0.90 and f1 > best_f1:
        best_threshold = threshold
        best_f1 = f1

plt.figure()
plt.plot(thresholds, recall_values, label='Recall', linestyle='--')
plt.plot(thresholds, f1_values, label='F1-Score', linestyle='-')
plt.title('Recall y F1-Score por Umbral')
plt.xlabel('Umbral')
plt.ylabel('Valor')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('UV/recall_f1_por_umbral.png')

print(f"Mejor Umbral (F1-Score y Recall >= 90%): {best_threshold:.2f} - F1-Score: {best_f1:.4f}")

# ---------------------------
# 4. Evaluación Final en Test
# ---------------------------
y_pred_probs_test = model.predict(test_generator).ravel()
y_true_test = test_generator.classes
y_pred_test = (y_pred_probs_test > best_threshold).astype(int)

# Matriz de confusión
cm = confusion_matrix(y_true_test, y_pred_test)
tn, fp, fn, tp = cm.ravel()

# ---------------------------
# 5. Métricas
# ---------------------------
precision = precision_score(y_true_test, y_pred_test)
f1 = f1_score(y_true_test, y_pred_test)
accuracy = accuracy_score(y_true_test, y_pred_test)
fpr, tpr, _ = roc_curve(y_true_test, y_pred_probs_test)
roc_auc = auc(fpr, tpr)

# Cálculos específicos
sensibilidad = tp / (tp + fn)  # verdaderos positivos 
especificidad = tn / (tn + fp) # verdaderos negativos 

# ---------------------------
# 6. Resultados
# ---------------------------
print("\nResultados en el conjunto de prueba:")
print(f"Precisión (Precision): {precision:.4f}")
print(f"Sensibilidad (Sensitivity): {sensibilidad:.4f}")
print(f"Especificidad (Specificity): {especificidad:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"Exactitud (Accuracy): {accuracy:.4f}")
print(f"AUC-ROC: {roc_auc:.4f}")

with open("resultados_FINAL.txt", "w") as f:
    f.write("Resultados en el conjunto de prueba:\n")
    f.write(f"Mejor Umbral: {best_threshold:.4f}\n")
    f.write(f"Precisión: {precision:.4f}\n")
    f.write(f"Sensibilidad: {sensibilidad:.4f}\n")
    f.write(f"Especificidad: {especificidad:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
    f.write(f"Exactitud (Accuracy): {accuracy:.4f}\n")
    f.write(f"AUC-ROC: {roc_auc:.4f}\n")
# ---------------------------
# 7. Matriz de Confusión Visual (estilo igual a la imagen)
# ---------------------------
plt.figure(figsize=(4, 4))

ax = sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',      # mismo estilo de azul
    cbar=False,
    square=True,       # hace la matriz cuadrada
    xticklabels=[0, 1],
    yticklabels=[0, 1]
)

ax.set_xlabel('Predicción')
ax.set_ylabel('Real')
plt.xticks(rotation=0)
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('UV/matriz_confusion.png', dpi=300)
plt.show()
