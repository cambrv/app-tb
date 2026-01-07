import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Dropout, Activation, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.models import Model, load_model
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import seaborn as sns
import os

# ---------------------------
# CARGAR MODELO Y DATOS
# ---------------------------
model = load_model('tbnet_model_finetuned.h5')
test_dir = '../../dataset/test'
batch_size = 32
img_size = (256, 256)

val_datagen = ImageDataGenerator(rescale=1./255)
test_generator = val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical', shuffle=False)

# ---------------------------
# EVALUACIÓN Y PREDICCIÓN
# ---------------------------
scores = model.evaluate(test_generator)
print(f'Loss: {scores[0]}, Accuracy: {scores[1]}')

# Predicciones
y_true = test_generator.classes
y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(y_pred, axis=1)

# ---------------------------
# MATRIZ DE CONFUSIÓN
# ---------------------------
cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión')
plt.show()

# ---------------------------
# REPORTES DE MÉTRICAS
# ---------------------------
report = classification_report(y_true, y_pred_classes, target_names=test_generator.class_indices.keys())
print(report)
with open('evaluation_report1.txt', 'w') as f:
    f.write(report)

# ---------------------------
# CURVAS ROC Y PRECISIÓN-RECALL
# ---------------------------
fpr, tpr, _ = roc_curve(y_true, y_pred[:, 1])
roc_auc = auc(fpr, tpr)
plt.plot(fpr, tpr, label='AUC = %.3f' % roc_auc)
plt.title('Curva ROC')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.legend()
plt.show()

precision, recall, _ = precision_recall_curve(y_true, y_pred[:, 1])
plt.plot(recall, precision)
plt.title('Curva Precisión-Recall')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.show()

# ---------------------------
# CARGAR HISTORIAL Y GRAFICAR
# ---------------------------
with open('history_tbnet.pkl', 'rb') as file:
    history = pickle.load(file)

plt.plot(history['accuracy'], label='Train Accuracy')
plt.plot(history['val_accuracy'], label='Validation Accuracy')
plt.title('Precisión del Modelo')
plt.legend()
plt.show()

plt.plot(history['loss'], label='Train Loss')
plt.plot(history['val_loss'], label='Validation Loss')
plt.title('Pérdida del Modelo')
plt.legend()
plt.show()
