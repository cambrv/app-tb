import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import pandas as pd

# ---------------------------
# CONFIGURACIÓN DEL DIRECTORIO
# ---------------------------
test_dir = '../../dataset/test'
batch_size = 32
img_size = (224, 224)

# ---------------------------
# CARGAR MODELO Y HISTORIAL
# ---------------------------
model = load_model('tbnet_model_optimized.h5')
with open('history_tbnet_optimized.pkl', 'rb') as file:
    history = pickle.load(file)

# ---------------------------
# GENERADOR DE IMÁGENES DE TEST
# ---------------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir, 
    target_size=img_size, 
    batch_size=batch_size, 
    class_mode='binary', 
    shuffle=False
)

# ---------------------------
# EVALUACIÓN DEL MODELO
# ---------------------------
scores = model.evaluate(test_generator)
print(f'Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}')

# ---------------------------
# PREDICCIONES
# ---------------------------
y_true = test_generator.classes
y_pred_probs = model.predict(test_generator).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

# ---------------------------
# MATRIZ DE CONFUSIÓN
# ---------------------------
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# ---------------------------
# REPORTE DE CLASIFICACIÓN
# ---------------------------
report = classification_report(y_true, y_pred, target_names=['Normal', 'TB'])
print("\nReporte de Clasificación:")
print(report)

# ---------------------------
# CURVA ROC Y AUC
# ---------------------------
fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.show()

# ---------------------------
# CURVA PRECISIÓN-RECALL
# ---------------------------
precision, recall, _ = precision_recall_curve(y_true, y_pred_probs)
plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label='Curva Precisión-Recall')
plt.title('Curva Precisión-Recall')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.legend()
plt.show()

# ---------------------------
# GRAFICAR HISTORIAL (PRECISIÓN Y PÉRDIDA)
# ---------------------------
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Pérdida del Modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()

# ---------------------------
# GUARDAR RESULTADOS EN TXT
# ---------------------------
with open('evaluation_results_tbnet_optimized.txt', 'w') as f:
    f.write(f'Loss: {scores[0]:.4f}, Accuracy: {scores[1]:.4f}\n')
    f.write("Matriz de Confusión:\n")
    f.write(np.array2string(cm) + '\n')
    f.write("\nReporte de Clasificación:\n")
    f.write(report)
    f.write(f"\nAUC-ROC: {roc_auc:.4f}\n")