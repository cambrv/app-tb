import os
import shutil
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---------------------------
# 1. Configuración
# ---------------------------
TFLITE_MODEL_PATH = "final_model.tflite"   
test_dir = "../../dataset/test"
img_size = (224, 224)
batch_size = 32   # tamaño del generador

FORCED_THRESHOLD = None   

# Carpetas de salida
output_base_dir = "UV/errores_tflite"
fp_dir = os.path.join(output_base_dir, "false_positives")
fn_dir = os.path.join(output_base_dir, "false_negatives")

os.makedirs(fp_dir, exist_ok=True)
os.makedirs(fn_dir, exist_ok=True)

# ---------------------------
# 2. Cargar modelo TFLite
# ---------------------------
interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_index = input_details[0]['index']
output_index = output_details[0]['index']

input_shape = input_details[0]['shape']
print("Input shape TFLite:", input_shape) 

# ---------------------------
# 3. Generador de test
# ---------------------------
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
)

num_samples = len(test_generator.filenames)
y_true = test_generator.classes
filenames = test_generator.filenames

# ---------------------------
# 4. Predicción con TFLite (una imagen a la vez)
# ---------------------------
y_pred_probs = np.zeros(num_samples, dtype=np.float32)

test_generator.reset()
i = 0

for batch_x, _ in test_generator:
    current_batch_size = batch_x.shape[0]

    for j in range(current_batch_size):
        if i >= num_samples:
            break

        img = batch_x[j:j+1, ...].astype(input_details[0]['dtype'])
        interpreter.set_tensor(input_index, img)
        interpreter.invoke()
        output = interpreter.get_tensor(output_index)  # (1, 1) o (1,)
        y_pred_probs[i] = float(output.ravel()[0])
        i += 1

    if i >= num_samples:
        break

print("\nPredicciones TFLite completadas.")
print(f"Total muestras: {num_samples}")

# ---------------------------
# 5. Búsqueda del mejor umbral
# ---------------------------
if FORCED_THRESHOLD is None:
    thresholds = np.arange(0.10, 0.90, 0.01)
    best_threshold = 0.5
    best_f1 = 0.0
    best_recall = 0.0
    best_precision = 0.0

    recall_values = []
    f1_values = []

    for thr in thresholds:
        y_pred_temp = (y_pred_probs > thr).astype(int)
        recall = recall_score(y_true, y_pred_temp)
        f1 = f1_score(y_true, y_pred_temp)
        recall_values.append(recall)
        f1_values.append(f1)

        # condición: recall >= 0.90 y mejor F1
        if recall >= 0.90 and f1 > best_f1:
            best_f1 = f1
            best_threshold = thr
            best_recall = recall
            best_precision = precision_score(y_true, y_pred_temp)

    print("\n=== Búsqueda de umbral ===")
    print(f"Mejor umbral (con recall >= 0.90): {best_threshold:.2f}")
    print(f"Precision: {best_precision:.4f}")
    print(f"Recall:    {best_recall:.4f}")
    print(f"F1-Score:  {best_f1:.4f}")
    THRESHOLD = best_threshold
else:
    THRESHOLD = FORCED_THRESHOLD
    print(f"\nUsando umbral forzado por ti: {THRESHOLD:.2f}")

# ---------------------------
# 6. Métricas finales con el umbral elegido
# ---------------------------
y_pred = (y_pred_probs > THRESHOLD).astype(int)

cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = cm.ravel()

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n=== Métricas finales en TEST ===")
print(f"Umbral usado: {THRESHOLD:.4f}")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# ---------------------------
# 7. Copiar falsos positivos y falsos negativos
# ---------------------------
fp_count = 0
fn_count = 0

for true_label, pred_label, prob, rel_path in zip(y_true, y_pred, y_pred_probs, filenames):
    src_path = os.path.join(test_dir, rel_path)
    file_name = os.path.basename(rel_path)

    # Falso positivo: predice 1 pero era 0
    if pred_label == 1 and true_label == 0:
        dst_path = os.path.join(fp_dir, f"FP_{prob:.4f}_" + file_name)
        shutil.copy2(src_path, dst_path)
        fp_count += 1

    # Falso negativo: predice 0 pero era 1
    elif pred_label == 0 and true_label == 1:
        dst_path = os.path.join(fn_dir, f"FN_{prob:.4f}_" + file_name)
        shutil.copy2(src_path, dst_path)
        fn_count += 1

print("\n=== Imágenes copiadas ===")
print(f"Falsos positivos: {fp_count} -> {fp_dir}")
print(f"Falsos negativos: {fn_count} -> {fn_dir}")
