import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="ProSkin | Analisis Warna Kulit",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# --- FUNGSI HELPER ---
@st.cache_resource
def load_keras_model(model_path):
    """Memuat model Keras dengan aman dari file .h5."""
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

# --- SIDEBAR ---
with st.sidebar:
    st.title("✨ ProSkin Analyzer")
    st.info(
        "Aplikasi ini menggunakan model Deep Learning untuk menganalisis warna kulit "
        "dari gambar yang Anda berikan. Pilih model dan metode input untuk memulai."
    )
    
    MODEL_LIST = {
        "MobileNet": "MobileNet_best.h5",
        "MobileNetV2": "MobileNetV2_best.h5",
        "NASNetMobile": "NASNetMobile_best.h5"
    }

    model_choice = st.selectbox("Pilih Model:", list(MODEL_LIST.keys()))
    
    st.success(f"Model **{model_choice}** siap digunakan.")

# Memuat model yang dipilih
model_path = MODEL_LIST[model_choice]
model = load_keras_model(model_path)


# --- HALAMAN UTAMA ---
st.header("Analisis Warna Kulit Anda", divider='rainbow')

# Pilihan metode input dengan tombol
col1, col2 = st.columns(2)
with col1:
    if st.button("📁 Unggah Gambar", use_container_width=True, type="primary"):
        st.session_state.input_method = 'upload'
        st.session_state.image_input = None
with col2:
    if st.button("📸 Buka Kamera", use_container_width=True):
        st.session_state.input_method = 'camera'
        st.session_state.image_input = None

# --- Widget Input Dinamis ---
# Widget hanya akan muncul setelah tombol ditekan
if model:
    if st.session_state.input_method == 'upload':
        uploaded_file = st.file_uploader(
            "Pilih file gambar...",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed"
        )
        if uploaded_file:
            st.session_state.image_input = Image.open(uploaded_file)
            
    elif st.session_state.input_method == 'camera':
        camera_file = st.camera_input(
            "Arahkan kamera ke kulit Anda",
            label_visibility="collapsed"
        )
        if camera_file:
            st.session_state.image_input = Image.open(camera_file)
else:
    st.error("Model tidak dapat dimuat. Aplikasi tidak dapat melanjutkan.")

# --- TAMPILKAN HASIL ---
if st.session_state.image_input is not None:
    image_to_process = st.session_state.image_input

    st.write("---")
    st.header("Hasil Analisis")

    with st.spinner(f"Model '{model_choice}' sedang menganalisis..."):
        prediction = process_and_predict(image_to_process, model)
    
    CLASS_NAMES = ['dark', 'light', 'mid-light']
    predicted_class_index = np.argmax(prediction[0])
    predicted_class_name = CLASS_NAMES[predicted_class_index]
    confidence_score = np.max(prediction[0]) * 100

    # Tampilan hasil yang lebih profesional
    result_container = st.container(border=True)
    res_col1, res_col2 = result_container.columns([1, 2])

    with res_col1:
        st.image(image_to_process, caption="Gambar Anda", use_column_width=True)

    with res_col2:
        st.subheader("Prediksi Terdeteksi:")
        st.success(f"**{predicted_class_name.replace('-', ' ').title()}**")
        st.progress(int(confidence_score), text=f"Keyakinan: {confidence_score:.2f}%")
        
        with st.expander("Lihat Detail Probabilitas"):
            for i, class_name in enumerate(CLASS_NAMES):
                prob = prediction[0][i] * 100
                st.write(f"{class_name.replace('-', ' ').title()}: `{prob:.2f}%`")
                
    if st.button("Ulangi Analisis", use_container_width=True, type="primary"):
        st.session_state.input_method = None
        st.session_state.image_input = None
        st.rerun()