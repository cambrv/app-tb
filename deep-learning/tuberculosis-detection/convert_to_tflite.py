import tensorflow as tf

model_h5_path = "models/xd.h5"  
output_tflite_path = "models/mobilenet-model.tflite" 

model = tf.keras.models.load_model(model_h5_path)
print("Modelo .h5 cargado correctamente.")

# Convertir el modelo a TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT] 
tflite_model = converter.convert()

# Guardar
with open(output_tflite_path, "wb") as f:
    f.write(tflite_model)

print(f"Modelo convertido correctamente y guardado en {output_tflite_path}")
