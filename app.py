import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==================== KONFIGURASI HALAMAN ====================
st.set_page_config(page_title="Deteksi Pisang", page_icon="🍌", layout="wide")

# Tampilan mobile-friendly
st.markdown("""
<style>
    .block-container { padding-top: 1rem; padding-bottom: 1rem; }
    .stAlert { margin-top: -15px; }
</style>
""", unsafe_allow_html=True)

# ==================== KONFIGURASI KELAS ====================
CLASS_NAMES = ['bercak_matang', 'bercak_mentah', 'busuk', 'matang', 'mentah', 'setengah_matang']

RIPENESS_SCORES = {
    'mentah': 5,
    'bercak_mentah': 10,
    'setengah_matang': 40,
    'matang': 85,
    'bercak_matang': 90,
    'busuk': 100
}

WATER_CONTENT = {
    'mentah': 60,
    'bercak_mentah': 62,
    'setengah_matang': 65,
    'matang': 70,
    'bercak_matang': 72,
    'busuk': 85
}

SHELF_LIFE = {
    'mentah': '7-10 hari',
    'bercak_mentah': '7-10 hari (bercak alami tidak memengaruhi)',
    'setengah_matang': '5-7 hari',
    'matang': '2-3 hari',
    'bercak_matang': '2-3 hari',
    'busuk': '0 hari (buang)'
}

# ==================== FUNGSI ALERT ====================
@st.cache_resource(validate=False)
def load_model():
    return tf.keras.models.load_model('models/banana_ripeness_cnn.h5')

def get_alert(predictions):
    idx = np.argmax(predictions)
    predicted_class = CLASS_NAMES[idx]
    confidence = predictions[idx] * 100

    ripeness = sum(predictions[i] * RIPENESS_SCORES[CLASS_NAMES[i]] for i in range(len(CLASS_NAMES)))
    ripeness = min(ripeness, 100)
    water = sum(predictions[i] * WATER_CONTENT[CLASS_NAMES[i]] for i in range(len(CLASS_NAMES)))

    if predicted_class == 'mentah':
        warna = 'Hijau'
        status = 'Belum layak konsumsi, masih sangat mentah.'
    elif predicted_class == 'bercak_mentah':
        warna = 'Hijau dengan bercak hitam alami (bukan busuk)'
        status = 'Pisang masih mentah dan aman. Bercak adalah hal alami.'
    elif predicted_class == 'setengah_matang':
        warna = 'Kuning kehijauan'
        status = 'Dapat dikonsumsi, namun tekstur masih agak keras.'
    elif predicted_class == 'matang':
        warna = 'Kuning cerah'
        status = 'Aman dan layak dikonsumsi.'
    elif predicted_class == 'bercak_matang':
        warna = 'Kuning dengan bercak coklat alami'
        status = 'Aman dikonsumsi. Bercak coklat adalah bagian dari proses pematangan normal.'
    elif predicted_class == 'busuk':
        warna = 'Coklat tua / Hitam, tekstur lembek'
        status = 'TIDAK LAYAK KONSUMSI. Pisang sudah busuk atau terlalu matang.'
    else:
        warna = 'Tidak diketahui'
        status = 'Status tidak diketahui.'

    umur = SHELF_LIFE.get(predicted_class, '?')
    if predicted_class in ['mentah', 'bercak_mentah']:
        saran = 'Simpan di suhu ruang (25°C), jauh dari sinar matahari langsung. Pisahkan dari buah lain.'
    elif predicted_class == 'setengah_matang':
        saran = 'Simpan di suhu ruang hingga matang, atau masukkan kulkas untuk memperlambat.'
    elif predicted_class in ['matang', 'bercak_matang']:
        saran = 'Simpan di lemari pendingin (14-16°C). Keluarkan 30 menit sebelum dimakan.'
    elif predicted_class == 'busuk':
        saran = 'Segera buang. Jangan campur dengan buah segar.'
    else:
        saran = 'Tidak tersedia.'

    return predicted_class, confidence, ripeness, water, warna, status, umur, saran

# ==================== TAMPILAN APLIKASI ====================
st.title("🍌 Deteksi Kematangan & Kualitas Pisang")
st.write("Unggah gambar pisang untuk melihat tingkat kematangan, kadar air, umur simpan, dan saran penyimpanan.")

uploaded_file = st.file_uploader("Pilih gambar pisang...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Baca dan tampilkan gambar
    img = Image.open(uploaded_file).convert('RGB')
    st.image(img, caption='Gambar yang diunggah', use_column_width=True)

    # Preprocessing yang sudah dioptimasi
    img_resized = img.resize((224, 224))
    img_array = np.array(img_resized, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    with st.spinner("🔍 Menganalisis gambar..."):
        model = load_model()
        preds = model.predict(img_array, verbose=0)[0]
        kelas, conf, ripe, water, warna, status, umur, saran = get_alert(preds)

    # Tampilkan hasil
    st.markdown("---")
    st.subheader("📊 Hasil Deteksi")
    st.success(f"**Prediksi Utama:** {kelas.upper()} (Keyakinan {conf:.2f}%)")
    st.info(f"**Warna Dominan:** {warna}")
    if kelas == 'busuk':
        st.error(f"**Status:** {status}")
    else:
        st.write(f"**Status:** {status}")

    col1, col2 = st.columns(2)
    col1.metric("Tingkat Kematangan", f"{ripe:.1f}%")
    col2.metric("Estimasi Kadar Air", f"{water:.1f}%")
    
    st.caption(f"**Umur Simpan (suhu ruang):** {umur}")
    st.caption(f"**Saran Penyimpanan:** {saran}")

    # Grafik probabilitas
    st.subheader("📈 Probabilitas Tiap Kelas")
    prob_dict = {CLASS_NAMES[i]: preds[i]*100 for i in range(len(CLASS_NAMES))}
    st.bar_chart(prob_dict)