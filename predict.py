import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import sys
import os

# ==================== KONFIGURASI KELAS ====================
# Urutan HARUS SAMA dengan urutan alfabetis folder dataset!
CLASS_NAMES = ['bercak_matang', 'bercak_mentah', 'busuk', 'matang', 'mentah', 'setengah_matang']

# Skor kematangan (0-100) untuk setiap kelas
RIPENESS_SCORES = {
    'mentah': 5,
    'bercak_mentah': 10,   # masih mentah, hanya ada bercak alami
    'setengah_matang': 40,
    'matang': 85,
    'bercak_matang': 90,   # matang dengan bercak coklat alami
    'busuk': 100           # terlalu matang / rusak
}

# Estimasi kadar air (%) per kelas (berdasarkan studi pustaka)
WATER_CONTENT = {
    'mentah': 60,
    'bercak_mentah': 62,   # sedikit lebih tinggi karena bercak bisa menyerap air
    'setengah_matang': 65,
    'matang': 70,
    'bercak_matang': 72,
    'busuk': 85            # busuk biasanya lebih berair
}

# Umur simpan (hari) pada suhu ruang (25-28°C)
SHELF_LIFE = {
    'mentah': '7-10 hari',
    'bercak_mentah': '7-10 hari (bercak tidak mempengaruhi umur simpan)',
    'setengah_matang': '5-7 hari',
    'matang': '2-3 hari',
    'bercak_matang': '2-3 hari',
    'busuk': '0 hari (buang)'
}

# ==================== FUNGSI ALERT ====================
def get_detailed_alert(predictions, class_names):
    # Kelas dengan probabilitas tertinggi
    idx_max = np.argmax(predictions)
    predicted_class = class_names[idx_max]
    confidence = predictions[idx_max] * 100

    # Hitung tingkat kematangan numerik (weighted average)
    ripeness = sum(predictions[i] * RIPENESS_SCORES[class_names[i]] for i in range(len(class_names)))
    ripeness_percent = min(ripeness, 100)

    # Estimasi kadar air (weighted average)
    water = sum(predictions[i] * WATER_CONTENT[class_names[i]] for i in range(len(class_names)))

    # Deskripsi warna & status konsumsi
    if predicted_class == 'mentah':
        warna = 'Hijau'
        status_konsumsi = 'Belum layak konsumsi, masih sangat mentah.'
    elif predicted_class == 'bercak_mentah':
        warna = 'Hijau dengan bercak hitam alami (bukan busuk)'
        status_konsumsi = ('Pisang masih mentah dan aman. Bercak hitam adalah hal alami '
                           '(misalnya getah atau memar ringan), tidak berbahaya.')
    elif predicted_class == 'setengah_matang':
        warna = 'Kuning kehijauan'
        status_konsumsi = 'Dapat dikonsumsi, namun tekstur masih agak keras.'
    elif predicted_class == 'matang':
        warna = 'Kuning cerah'
        status_konsumsi = 'Aman dan layak dikonsumsi.'
    elif predicted_class == 'bercak_matang':
        warna = 'Kuning dengan bercak coklat alami'
        status_konsumsi = ('Aman dikonsumsi. Bercak coklat adalah bagian dari proses pematangan normal, '
                           'bukan tanda kerusakan.')
    elif predicted_class == 'busuk':
        warna = 'Coklat tua / Hitam, tekstur lembek'
        status_konsumsi = 'TIDAK LAYAK KONSUMSI. Pisang sudah busuk atau terlalu matang.'
    else:
        warna = 'Tidak diketahui'
        status_konsumsi = 'Status tidak diketahui.'

    umur = SHELF_LIFE.get(predicted_class, '?')
    if predicted_class in ['mentah', 'bercak_mentah']:
        saran_simpan = ('Simpan di suhu ruang (25°C), jauh dari sinar matahari langsung. '
                        'Pisahkan dari buah lain untuk memperlambat pematangan.')
    elif predicted_class in ['setengah_matang']:
        saran_simpan = ('Simpan di suhu ruang hingga matang sempurna, atau masukkan ke kulkas '
                        'jika ingin memperlambat pematangan.')
    elif predicted_class in ['matang', 'bercak_matang']:
        saran_simpan = ('Simpan di lemari pendingin (14-16°C) untuk memperpanjang kesegaran. '
                        'Keluarkan 30 menit sebelum dikonsumsi agar rasa optimal.')
    elif predicted_class == 'busuk':
        saran_simpan = 'Segera buang. Jangan disimpan di dekat buah segar karena dapat mempercepat pembusukan.'
    else:
        saran_simpan = 'Tidak tersedia.'

    # Bangun string output
    alert = f"""
============================================================
🍌 HASIL DETEKSI KEMATANGAN & KUALITAS PISANG 🍌
============================================================
Kelas Utama         : {predicted_class.upper()}
Keyakinan           : {confidence:.2f}%
Warna Dominan       : {warna}
Status Konsumsi     : {status_konsumsi}

------------------------------------------------------------
📊 INDEKS KEMATANGAN (0-100%)
------------------------------------------------------------
Tingkat Kematangan  : {ripeness_percent:.1f}%
Estimasi Kadar Air  : {water:.1f}%

------------------------------------------------------------
⏳ UMUR SIMPAN & SARAN PENYIMPANAN
------------------------------------------------------------
Umur Simpan (ruang) : {umur}
Saran Penyimpanan   : {saran_simpan}

------------------------------------------------------------
📈 PROBABILITAS TIAP KELAS
------------------------------------------------------------"""
    for i, name in enumerate(class_names):
        bar = '█' * int(predictions[i] * 50)
        alert += f"\n  {name:20s}: {bar:20s} {predictions[i]*100:.2f}%"

    alert += "\n============================================================\n"
    return alert

# ==================== FUNGSI PREDIKSI ====================
def load_model(model_path='models/banana_ripeness_cnn.h5'):
    if not os.path.exists(model_path):
        print(f"❌ Model tidak ditemukan di: {model_path}")
        sys.exit(1)
    return keras.models.load_model(model_path)

def preprocess_image(image_path):
    img = Image.open(image_path).resize((224, 224))
    img_array = np.array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict(image_path):
    model = load_model()
    img_array = preprocess_image(image_path)
    preds = model.predict(img_array, verbose=0)[0]
    alert = get_detailed_alert(preds, CLASS_NAMES)
    print(alert)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cara pakai: python predict.py <path_gambar>")
        print("Contoh: python predict.py dataset/test/matang/pisang.jpg")
    else:
        predict(sys.argv[1])