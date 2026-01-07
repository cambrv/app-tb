import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, ReLU, MaxPooling2D, Add, GlobalAveragePooling2D, Dense, Dropout, Activation, GlobalMaxPooling2D, Concatenate
from tensorflow.keras.models import Model, load_model
import pickle
import numpy as np
import os

# ---------------------------
# CONFIGURACIÓN DEL DIRECTORIO
# ---------------------------
train_dir = '../../dataset/train'
val_dir = '../../dataset/val'
test_dir = '../../dataset/test'
batch_size = 32
img_size = (224, 224)

# ---------------------------
# DEFINICIÓN DEL MODELO TBNet (OPTIMIZADO)
# ---------------------------
base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Congelar el 80% de las capas
for layer in base_model.layers[:int(len(base_model.layers) * 0.8)]:
    layer.trainable = False

inputs = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='swish', kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
x = Dropout(0.5)(x)
outputs = Dense(1, activation='sigmoid')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compilando el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# GENERADORES DE IMÁGENES
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.15,
    horizontal_flip=True,
    shear_range=0.1
)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='binary')

# ---------------------------
# ENTRENAMIENTO CON TRANSFER LEARNING
# ---------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5),
        ModelCheckpoint('tbnet_model_optimized.h5', save_best_only=True, monitor='val_loss')
    ],
    verbose=1
)

# Guardar el historial
with open('history_tbnet_optimized.pkl', 'wb') as file:
    pickle.dump(history.history, file)
