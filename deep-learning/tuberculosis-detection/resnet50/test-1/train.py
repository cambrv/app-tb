import os
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2
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

# ---------------------------
# GENERADORES DE IMÁGENES MEJORADOS
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    zoom_range=0.3,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    brightness_range=(0.7, 1.3),
    shear_range=0.15
)

val_datagen = ImageDataGenerator(rescale=1./255)

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

# ---------------------------
# MODELO ResNet50
# ---------------------------
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False
for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu', kernel_regularizer=l2(0.001))(x)
x = Dropout(0.5)(x)
output = Dense(1, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=output)

# ---------------------------
# ENTRENAMIENTO INICIAL
# ---------------------------
model.compile(optimizer=Adam(learning_rate=learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=int(epochs * 0.6),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)]
)

# ---------------------------
# FINE-TUNING
# ---------------------------
base_model.trainable = True
for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

model.compile(optimizer=Adam(learning_rate=fine_tuning_learning_rate),
              loss='binary_crossentropy',
              metrics=['accuracy', tf.keras.metrics.Recall(name='recall')])

history_finetuning = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=int(epochs * 0.4),
    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, mode='min', restore_best_weights=True)]
)

# ---------------------------
# COMBINAR HISTORIALES
# ---------------------------
def combinar_historiales(hist1, hist2):
    historial_completo = {}
    for clave in hist1.history.keys():
        historial_completo[clave] = hist1.history[clave] + hist2.history[clave]
    return historial_completo

historial_completo = combinar_historiales(history, history_finetuning)

# ---------------------------
# GUARDAR MODELO E HISTORIAL
# ---------------------------
model.save("modelo_tb_resnet50.h5")

with open("historial_resnet50.pkl", "wb") as f:
    pickle.dump(historial_completo, f)