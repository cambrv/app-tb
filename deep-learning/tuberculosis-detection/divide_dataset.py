import os
import shutil
import random

# ---------------------------
# CONFIGURACIÓN
# ---------------------------
dataset_dir = 'dataset/'  # Ruta de tu dataset original
output_dir = 'dataset_split/'  # Ruta donde se guardará el dataset dividido
categories = ['normal', 'tb']  # Las dos clases

train_ratio = 0.8
val_ratio = 0.15
test_ratio = 0.05

# ---------------------------
# CREAR ESTRUCTURA DE CARPETAS
# ---------------------------
for category in categories:
    os.makedirs(os.path.join(output_dir, 'train', category), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'val', category), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'test', category), exist_ok=True)

# ---------------------------
# DIVIDIR LOS DATOS
# ---------------------------
for category in categories:
    category_path = os.path.join(dataset_dir, category)
    images = os.listdir(category_path)
    random.shuffle(images)

    train_split = int(len(images) * train_ratio)
    val_split = int(len(images) * (train_ratio + val_ratio))

    train_images = images[:train_split]
    val_images = images[train_split:val_split]
    test_images = images[val_split:]

    # Mover imágenes a las carpetas correspondientes
    for image in train_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(output_dir, 'train', category, image)
        shutil.copy(src, dst)

    for image in val_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(output_dir, 'val', category, image)
        shutil.copy(src, dst)

    for image in test_images:
        src = os.path.join(category_path, image)
        dst = os.path.join(output_dir, 'test', category, image)
        shutil.copy(src, dst)

print("✅ División completada: Train, Validation y Test.")
