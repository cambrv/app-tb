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
img_size = (256, 256)

# ---------------------------
# DEFINICIÓN DEL MODELO TBNet (MEJORADO)
# ---------------------------
def residual_block(x, filters):
    shortcut = x

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation('swish')(x)

    x = Conv2D(filters, (3, 3), padding="same")(x)
    x = BatchNormalization()(x)

    x = Add()([x, shortcut])
    x = Activation('swish')(x)

    return x

# Bloque denso
def dense_block(x, filters):
    for _ in range(4):
        conv_layer = Conv2D(filters, (3, 3), padding="same")(x)
        conv_layer = BatchNormalization()(conv_layer)
        conv_layer = Activation('swish')(conv_layer)
        x = Concatenate()([x, conv_layer])
    return x

base_model = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
base_model.trainable = False

inputs = base_model.input
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(1024, activation='swish')(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)

# Compilando el modelo
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# GENERADORES DE IMÁGENES
# ---------------------------
train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
val_generator = val_datagen.flow_from_directory(val_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')
test_generator = val_datagen.flow_from_directory(test_dir, target_size=img_size, batch_size=batch_size, class_mode='categorical')

# ---------------------------
# ENTRENAMIENTO CON TRANSFER LEARNING
# ---------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=40,
    callbacks=[EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True), ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)],
    verbose=1
)

# Guardar el historial
with open('history_tbnet.pkl', 'wb') as file:
    pickle.dump(history.history, file)

# Fine-Tuning
base_model.trainable = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

history_fine = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    callbacks=[EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)],
    verbose=1
)

# Guardar modelo
model.save('tbnet_model_finetuned.h5')

# ---------------------------
# EVALUACIÓN EN TEST
# ---------------------------
scores = model.evaluate(test_generator)
with open('evaluation_results.txt', 'w') as f:
    f.write(f'Loss: {scores[0]}, Accuracy: {scores[1]}\n')
