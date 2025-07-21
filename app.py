import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="ProSkin | Analisis Warna Kulit Profesional",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- DATABASE KONTEN (DISESUAIKAN UNTUK INDONESIA) ---
# Deskripsi dan rekomendasi diubah agar lebih relevan dengan audiens,
# produk, dan iklim di Indonesia.
SKIN_TONE_INFO = {
    "light": {
        "display_name": "light",
        "description": "Warna kulit cerah cenderung memiliki produksi melanin yang lebih sedikit. Di iklim tropis Indonesia, tipe kulit ini sangat rentan terhadap kerusakan akibat sinar UV seperti kemerahan dan flek hitam.",
        "recommendation": "Wajib gunakan tabir surya (sunscreen) minimal SPF 30 PA++ setiap hari, bahkan saat di dalam ruangan. Ulangi pemakaian setiap 2-3 jam. Cari produk dengan kandungan antioksidan seperti Vitamin C untuk melawan radikal bebas dan mencegah kulit kusam."
    },
    "mid-light": {
        "display_name": "mid-light",
        "description": "Ini adalah warna kulit paling umum di Indonesia, sering disebut kuning langsat. Memiliki keseimbangan melanin yang baik, namun tetap berisiko mengalami kulit kusam dan noda bekas jerawat (PIH) jika tidak dirawat dengan tepat.",
        "recommendation": "Gunakan tabir surya minimal SPF 30 untuk perlindungan harian. Untuk menjaga kecerahan, gunakan serum dengan Niacinamide atau Vitamin C. Lakukan eksfoliasi ringan 1-2 kali seminggu dengan produk AHA/BHA untuk mengangkat sel kulit mati."
    },
    "dark": {
        "display_name": "dark",
        "description": "Warna kulit sawo matang atau gelap memiliki pesona eksotis dan kaya akan melanin yang memberikan perlindungan alami lebih baik dari matahari. Namun, sangat rentan membentuk noda hitam atau bekas luka yang menggelap (hiperpigmentasi).",
        "recommendation": "Fokus utama adalah hidrasi dan meratakan warna kulit. Gunakan pelembap dengan Hyaluronic Acid atau Ceramide. Untuk mengatasi noda hitam, cari bahan aktif seperti Niacinamide, Alpha Arbutin, atau Azelaic Acid. Tetap gunakan tabir surya untuk mencegah noda semakin gelap."
    }
}


# --- FUNGSI HELPER ---
@st.cache_resource
def load_keras_model(model_path):
    """Memuat model Keras dengan aman."""
    try:
        return tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        st.error(f"Error: Gagal memuat model dari '{model_path}'. Pastikan file ada dan tidak rusak.", icon="🚨")
        return None

def process_and_predict(image, model):
    """Memproses gambar dan melakukan prediksi."""
    image_rgb = image.convert('RGB')
    img_resized = image_rgb.resize((128, 128))
    img_array = tf.keras.preprocessing.image.img_to_array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

# --- Inisialisasi Session State ---
if 'input_method' not in st.session_state:
    st.session_state.input_method = None
if 'image_input' not in st.session_state:
    st.session_state.image_input = None
if 'model_choice' not in st.session_state:
    st.session_state.model_choice = "MobileNet"


# --- SIDEBAR ---
with st.sidebar:
    st.title("✨ ProSkin Analyzer")
    st.info(
        "Aplikasi ini menggunakan AI untuk menganalisis warna kulit "
        "dan memberikan rekomendasi perawatan yang dipersonalisasi."
    )
    
    # PERBAIKAN: Menambahkan path folder 'model/' untuk mencegah error.
    MODEL_LIST = {
        "MobileNet": "MobileNet_best.h5",
        "MobileNetV2": "MobileNetV2_best.h5",
        "NASNetMobile": "NASNetMobile_best.h5"
    }

    st.session_state.model_choice = st.selectbox(
        "Pilih Model Analisis:",
        list(MODEL_LIST.keys()),
        key="model_selector"
    )
    
    st.success(f"Model **{st.session_state.model_choice}** siap digunakan.")
    st.markdown("---")
    st.caption("© 2025 ProSkin Analyzer. All Rights Reserved.")


# Memuat model yang dipilih dari sidebar
model_path = MODEL_LIST[st.session_state.model_choice]
model = load_keras_model(model_path)


# --- HALAMAN UTAMA ---
st.title("Analisis Warna Kulit Profesional")
st.write("Selamat datang! Unggah foto kulit Anda untuk mendapatkan analisis mendalam dan rekomendasi perawatan dari AI kami.")

if model:
    if st.session_state.image_input is None:
        with st.container(border=True):
            st.subheader("Pilih Sumber Gambar")
            col1, col2 = st.columns(2)
            if col1.button("📁 Unggah Gambar", use_container_width=True, type="primary"):
                st.session_state.input_method = 'upload'
            if col2.button("📸 Buka Kamera", use_container_width=True):
                st.session_state.input_method = 'camera'

        if st.session_state.input_method:
            with st.container(border=True):
                st.subheader("Sediakan Gambar")
                if st.session_state.input_method == 'upload':
                    uploaded_file = st.file_uploader(
                        "Pilih file gambar...", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
                    )
                    if uploaded_file:
                        st.session_state.image_input = Image.open(uploaded_file)
                        st.rerun()
                        
                elif st.session_state.input_method == 'camera':
                    camera_file = st.camera_input(
                        "Arahkan kamera ke kulit Anda", label_visibility="collapsed"
                    )
                    if camera_file:
                        st.session_state.image_input = Image.open(camera_file)
                        st.rerun()
else:
    st.error("Model tidak dapat dimuat. Aplikasi tidak dapat melanjutkan.")

# --- TAMPILKAN HASIL ---
if st.session_state.image_input is not None and model is not None:
    image_to_process = st.session_state.image_input

    st.header("Laporan Analisis Kulit Anda", divider='rainbow')

    with st.spinner(f"Model '{st.session_state.model_choice}' sedang menganalisis..."):
        prediction = process_and_predict(image_to_process, model)
    
    CLASS_NAMES = ['dark', 'light', 'mid-light']
    predicted_class_name = CLASS_NAMES[np.argmax(prediction[0])]
    
    result_info = SKIN_TONE_INFO[predicted_class_name]
    confidence_score = np.max(prediction[0]) * 100

    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        # PERBAIKAN: Mengganti parameter yang sudah usang
        st.image(image_to_process, caption="Gambar yang Dianalisis", use_container_width=True)
    with res_col2:
        st.subheader(f"Hasil: **{result_info['display_name']}**")
        st.write("Tingkat Keyakinan Model:")
        st.progress(int(confidence_score), text=f"{confidence_score:.2f}%")
        st.divider()
        st.write("#### 📝 Deskripsi")
        st.info(result_info['description'])
        st.write("#### ✨ Rekomendasi Perawatan")
        st.success(result_info['recommendation'])

    with st.expander("Lihat Detail Probabilitas Model"):
        for i, class_name in enumerate(CLASS_NAMES):
            prob = prediction[0][i] * 100
            st.write(f"{SKIN_TONE_INFO[class_name]['display_name']}: `{prob:.2f}%`")
            
    st.divider()

    # PENAMBAHAN: Disclaimer ditambahkan di sini.
    st.warning(
        "**Disclaimer:** Hasil analisis ini adalah prediksi berdasarkan model AI dan tidak menggantikan konsultasi dengan dokter kulit profesional.",
        icon="⚠️"
    )

    if st.button("Analisis Gambar Lain", use_container_width=True, type="primary"):
        st.session_state.input_method = None
        st.session_state.image_input = None
        st.rerun()