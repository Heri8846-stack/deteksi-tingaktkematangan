import tensorflow as tf

# 1. Muat model yang sudah dilatih
model = tf.keras.models.load_model('models/banana_ripeness_cnn.h5')

# 2. Konversi ke TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# 3. Simpan file .tflite
with open('banana_ripeness_cnn.tflite', 'wb') as f:
    f.write(tflite_model)

print("✅ Model .tflite berhasil dibuat!")