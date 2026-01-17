import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model
import pickle
import os

# ---------------------------
# CONFIGURACIÓN DEL DIRECTORIO
# ---------------------------
train_dir = '../../dataset/train'
val_dir   = '../../dataset/val'
test_dir  = '../../dataset/test'

# ---------------------------
# HIPERPARÁMETROS
# ---------------------------
IMG_SIZE    = (224, 224)
BATCH_SIZE  = 32
EPOCHS      = 40
OPTIMIZER   = tf.keras.optimizers.Adam(learning_rate=1e-4)
INCLUDE_TOP = False
WEIGHTS     = 'imagenet'
POOLING     = 'avg'
FREEZE_PCT  = 0.80
FINAL_ACT   = 'sigmoid'

# ---------------------------
# BASE MODEL (TRANSFER LEARNING)
# ---------------------------
base_model = tf.keras.applications.DenseNet121(
    weights=WEIGHTS,
    include_top=INCLUDE_TOP,
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

# Congelar el 80% de las capas
num_layers = len(base_model.layers)
freeze_until = int(num_layers * FREEZE_PCT)
for layer in base_model.layers[:freeze_until]:
    layer.trainable = False
for layer in base_model.layers[freeze_until:]:
    layer.trainable = True

# Capa de pooling global promedio (GAP)
x = GlobalAveragePooling2D(name='gap')(base_model.output)

x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)

# Salida binaria con sigmoide
outputs = Dense(1, activation=FINAL_ACT, name='pred')(x)

model = Model(inputs=base_model.input, outputs=outputs, name='TBNet_DenseNet121_TL')

model.compile(
    optimizer=OPTIMIZER,
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ---------------------------
# GENERADORES DE IMÁGENES
# ---------------------------
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    shear_range=0.15,
    brightness_range=[0.8, 1.2]
)

val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_directory(
    val_dir,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

# ---------------------------
# CHECKPOINT
# ---------------------------
ckpt_cb = ModelCheckpoint(
    'tbnet_model_hparam_best.h5',
    save_best_only=True,
    monitor='val_loss',
    mode='min',
    verbose=1
)

# ---------------------------
# ENTRENAMIENTO
# ---------------------------
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    callbacks=[ckpt_cb],
    verbose=1
)

# Guardar el modelo final y el historial completo
model.save('tbnet_model_hparam_last.h5')

with open('history_tbnet_40ep.pkl', 'wb') as f:
    pickle.dump(history.history, f)


assert len(history.history['loss']) == EPOCHS, f"Historial tiene {len(history.history['loss'])} épocas, se esperaban {EPOCHS}."
print("Historial guardado con las 40 épocas en 'history_tbnet_40ep.pkl'.")
