# 📸 Aplikasi Face Dataset & Image Processing

Aplikasi Streamlit untuk pengumpulan dataset wajah dan pengolahan citra digital.

## ✨ Fitur

### 1. Tambah Dataset Wajah
- Capture wajah menggunakan webcam secara real-time
- Deteksi wajah otomatis menggunakan Haar Cascade
- Menyimpan 20 gambar wajah per person
- Progress tracking dengan visual feedback

### 2. Pengolahan Citra Dataset
Melakukan tiga tahap pengolahan citra:
- **a. Menambahkan Noise Salt and Pepper**: Mensimulasikan noise pada gambar
- **b. Menghilangkan Noise**: Menggunakan Median Filter untuk menghilangkan noise
- **c. Penajaman Citra**: Menggunakan kernel sharpening untuk meningkatkan detail gambar

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi Streamlit (Face Dataset & Processing)
```bash
streamlit run app.py
```

### 3. Jalankan Aplikasi FastAPI (Image Operations)
```bash
uvicorn main:app --reload
```
Kemudian buka: http://localhost:8000

## 📋 Persyaratan

- Python 3.8+
- Webcam (untuk fitur capture wajah)
- macOS: Pastikan memberikan izin kamera di System Preferences > Security & Privacy > Camera

## 📁 Struktur Folder

```
W4_TugasPCD/
├── app.py                 # Aplikasi Streamlit (Face Dataset & Processing)
├── main.py                # Aplikasi FastAPI (Image Operations)
├── requirements.txt       # Dependencies
├── dataset/              # Folder untuk menyimpan dataset wajah
│   └── [nama_person]/    # Folder per person
│       ├── img_0.jpg
│       ├── img_1.jpg
│       └── processed/    # Hasil pengolahan citra
├── static/
│   ├── uploads/          # Upload gambar sementara
│   └── histograms/       # Histogram yang dihasilkan
└── templates/            # Template HTML untuk FastAPI
```

## 🎯 Cara Menggunakan

### Menu 1: Tambah Dataset Wajah
1. Pilih menu "Tambah Dataset Wajah" di sidebar
2. Masukkan nama person
3. Klik "Tambahkan Wajah Baru"
4. Posisikan wajah di depan webcam
5. Aplikasi akan otomatis menangkap 20 gambar wajah

### Menu 2: Pengolahan Citra Dataset
1. Pilih menu "Pengolahan Citra Dataset" di sidebar
2. Pilih dataset person yang ingin diproses
3. Pilih gambar yang akan diproses
4. Atur parameter:
   - Probabilitas Salt Noise (0.0 - 0.1)
   - Probabilitas Pepper Noise (0.0 - 0.1)
   - Kernel Size Median Filter (3, 5, 7, 9)
5. Klik "🚀 Proses Gambar"
6. Lihat hasil: Original → Noisy → Denoised → Sharpened
7. Centang "Simpan hasil pengolahan" untuk menyimpan hasil

## 📊 Teknik Pengolahan Citra

### 1. Salt and Pepper Noise
Menambahkan noise acak berupa pixel putih (salt) dan hitam (pepper) pada gambar.

### 2. Median Filter
Menggunakan filter median untuk menghilangkan noise salt and pepper dengan mengambil nilai median dari pixel-pixel di sekitarnya.

### 3. Sharpening
Menggunakan kernel sharpening untuk meningkatkan ketajaman gambar dengan menonjolkan tepi dan detail.

## 👨‍💻 Author

Muhammad Samudera Bagja - 231524058
Jurusan Teknik Komputer - Politeknik Negeri Bandung

## 📝 Catatan

- Pastikan pencahayaan cukup saat mengambil gambar wajah
- Jaga jarak yang konsisten dari webcam
- Hindari background yang terlalu ramai untuk hasil deteksi lebih baik
