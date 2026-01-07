import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import pickle
import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
import seaborn as sns

# ---------------------------
# CONFIGURACIÓN DEL DIRECTORIO
# ---------------------------
test_dir = '../dataset/test'
batch_size = 32
img_size = (224, 224)

# ---------------------------
# CARGA DE LOS MODELOS
# ---------------------------
models = {
    'TBNet Optimized': 'modelo_tbnet.h5',
    'DenseNet121': 'modelo_densenet121.h5'
}

# ---------------------------
# GENERADOR DE IMÁGENES DE TEST
# ---------------------------
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, shuffle=False)

# ---------------------------
# FUNCIÓN DE EVALUACIÓN ADAPTATIVA
# ---------------------------
def evaluate_model(model_path):
    model = load_model(model_path)
    scores = model.evaluate(test_generator)
    y_true = test_generator.classes
    y_pred_probs = model.predict(test_generator)

    # Detección automática del tipo de salida
    if y_pred_probs.shape[-1] == 1:  # Sigmoid (1 clase)
        y_pred = (y_pred_probs > 0.5).astype('int32').flatten()
    elif y_pred_probs.shape[-1] == 2:  # Softmax (2 clases)
        y_pred_probs = y_pred_probs[:, 1]
        y_pred = (y_pred_probs > 0.5).astype('int32')

    cm = confusion_matrix(y_true, y_pred)
    print(f"Matriz de Confusión para {model_path}")
    print(cm)

    report = classification_report(y_true, y_pred, target_names=['Normal', 'TB'])
    print(f"\nReporte de Clasificación para {model_path}")
    print(report)

    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, label=f'AUC = {roc_auc:.3f}')
    plt.title(f'Curva ROC - {model_path}')
    plt.xlabel('Falsos Positivos')
    plt.ylabel('Verdaderos Positivos')
    plt.legend()
    plt.show()

# ---------------------------
# EVALUACIÓN DE TODOS LOS MODELOS
# ---------------------------
for model_name, model_path in models.items():
    print(f"\n\nEvaluando: {model_name}")
    evaluate_model(model_path)
