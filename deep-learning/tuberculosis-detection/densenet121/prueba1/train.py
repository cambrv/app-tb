import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import precision_score, recall_score, f1_score

# ---------------------------
# PARÁMETROS
# ---------------------------
img_size = (224, 224)
batch_size = 32
epochs = 40
learning_rate = 1e-4
fine_tuning_learning_rate = 1e-5

train_dir = '../../dataset/train'
val_dir = '../../dataset/val'
test_dir = '../../dataset/test'

# ---------------------------
# GENERADORES DE IMÁGENES
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    shear_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary'
)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

# ---------------------------
# MODELO DenseNet121
# ---------------------------
base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.4)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.3)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------------------
# ENTRENAMIENTO INICIAL (TRANSFER LEARNING)
# ---------------------------
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=int(epochs * 0.6)
)

# ---------------------------
# FINE-TUNING
# ---------------------------
base_model.trainable = True
for layer in base_model.layers[:-40]: 
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=fine_tuning_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy'])

history_finetuning = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=int(epochs * 0.4)
)

# ---------------------------
# OPTIMIZAR UMBRAL
# ---------------------------
y_pred_probs = model.predict(val_generator).ravel()
y_true = val_generator.classes

best_threshold = 0.5
best_f1 = 0
for threshold in np.arange(0.1, 0.9, 0.01):
    y_pred = (y_pred_probs > threshold).astype(int)
    f1 = f1_score(y_true, y_pred)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"Mejor Umbral: {best_threshold:.2f} - F1-Score: {best_f1:.2f}")

def combinar_historiales(hist1, hist2):
    historial_completo = {}
    for clave in hist1.history.keys():
        historial_completo[clave] = hist1.history[clave] + hist2.history[clave]
    return historial_completo

historial_completo = combinar_historiales(history, history_finetuning)

# ---------------------------
# GUARDAR MODELO E HISTORIAL
# ---------------------------
model.save("modelo_tb_densenet121_optimized_ULTIMAVERSION.h5")

with open("historial_densenet121_optimized_ULTIMAVERSION.pkl", "wb") as f:
    pickle.dump(historial_completo, f)

with open("best_threshold_ULTIMAVERSION.txt", "w") as f:
    f.write(f"Best Threshold: {best_threshold:.4f}\n")

# ---------------------------
# EVALUACIÓN EN TEST
# ---------------------------
y_pred_probs_test = model.predict(test_generator).ravel()
y_true_test = test_generator.classes
y_pred_test = (y_pred_probs_test > best_threshold).astype(int)

precision = precision_score(y_true_test, y_pred_test)
recall = recall_score(y_true_test, y_pred_test)
f1 = f1_score(y_true_test, y_pred_test)

print(f"Resultados en el conjunto de prueba:")
print(f"Precisión: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

# Guardar Resultados de prueba
with open("resultados_prueba.txt", "w") as f:
    f.write(f"Resultados en el conjunto de prueba:\n")
    f.write(f"Precisión: {precision:.4f}\n")
    f.write(f"Recall: {recall:.4f}\n")
    f.write(f"F1-Score: {f1:.4f}\n")
