
# ✨ ProSkin Analyzer: Aplikasi Deteksi Warna Kulit

Aplikasi web cerdas yang menggunakan Deep Learning untuk menganalisis dan mengklasifikasikan warna kulit manusia secara real-time dari gambar.

---

## 🚀 Fitur Utama

- **Analisis Real-time:** Dapatkan hasil prediksi warna kulit secara instan.
- **Pilihan Model Fleksibel:** Pilih salah satu dari tiga model CNN yang telah dilatih:
  - MobileNet
  - MobileNetV2
  - NASNetMobile
- **Dua Metode Input:**
  - 📁 **Unggah Gambar:** Analisis gambar dari perangkat Anda (JPG, JPEG, PNG).
  - 📸 **Kamera Langsung:** Gunakan kamera perangkat untuk mengambil gambar secara langsung.
- **Visualisasi Hasil:** Tampilkan kelas terdeteksi, tingkat keyakinan (confidence), dan distribusi probabilitas.
- **Antarmuka Responsif:** Desain modern dan bersih yang dapat diakses dari desktop maupun perangkat mobile.

---

## 🛠️ Teknologi yang Digunakan

- **Bahasa Pemrograman:** Python  
- **Framework Web:** Streamlit  
- **Deep Learning:** TensorFlow (Keras), NumPy  
- **Pemrosesan Gambar:** Pillow  
- **Deployment:** Streamlit Community Cloud  

---

## 💻 Cara Menjalankan Secara Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi ini di komputer Anda:

### 1. Clone Repository

```bash
git clone https://github.com/makiatulmsyr17/apk-skin-detection.git
cd apk-skin-detection
````

### 2. Buat dan Aktifkan Virtual Environment

```bash
# Buat virtual environment
python -m venv .venv

# Aktifkan (Windows)
.venv\Scripts\activate

# Aktifkan (macOS/Linux)
source .venv/bin/activate
```

### 3. Install Dependensi

```bash
pip install -r requirements.txt
```

### 4. Jalankan Aplikasi

```bash
streamlit run app.py
```

Aplikasi akan terbuka otomatis di browser Anda.

---

## 📁 Struktur Proyek

```
.
├── app.py                  # Kode utama aplikasi Streamlit
├── requirements.txt        # Daftar library yang dibutuhkan
├── MobileNet_best.h5       # Model CNN 1
├── MobileNetV2_best.h5     # Model CNN 2
├── NASNetMobile_best.h5    # Model CNN 3
└── README.md               # Dokumentasi proyek
```

---

## 🤝 Kontribusi

Kontribusi sangat terbuka! Jangan ragu untuk membuat *issue*, *pull request*, atau memberikan saran fitur baru.

---


