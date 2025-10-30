# 📸 Aplikasi Face Dataset & Image Processing

Aplikasi Streamlit untuk pengumpulan dataset wajah dan pengolahan citra digital, serta aplikasi FastAPI untuk berbagai operasi pengolahan citra termasuk deteksi tepi.

## ✨ Fitur

### 1. Tambah Dataset Wajah (Streamlit)
- Capture wajah menggunakan webcam secara real-time
- Deteksi wajah otomatis menggunakan Haar Cascade
- Menyimpan 20 gambar wajah per person
- Progress tracking dengan visual feedback

### 2. Pengolahan Citra Dataset (Streamlit)
Melakukan tiga tahap pengolahan citra:
- **a. Menambahkan Noise Salt and Pepper**: Mensimulasikan noise pada gambar
- **b. Menghilangkan Noise**: Menggunakan Median Filter untuk menghilangkan noise
- **c. Penajaman Citra**: Menggunakan kernel sharpening untuk meningkatkan detail gambar

### 3. Operasi Citra Web (FastAPI)
- **Home**: Upload dan operasi dasar pada citra
  - Operasi Aritmatika (add, subtract, max, min, inverse)
  - Operasi Logika (AND, XOR, NOT)
  - Statistik Citra (mean intensity, standard deviation)
  
- **Grayscale**: Konversi citra berwarna ke grayscale
  
- **Histogram**: Generate histogram grayscale dan berwarna
  
- **Equalization**: Equalisasi histogram untuk meningkatkan kontras
  
- **Specification**: Spesifikasi histogram dengan matching histogram
  
- **Filtering**: Berbagai operasi filtering
  - Convolution Filter (average, sharpen, edge)
  - Zero Padding
  - Frequency Filter (low pass, high pass, band pass)
  - Fourier Transform
  - Reduce Periodic Noise

### 4. Edge Detection (FastAPI) - **BARU!** ✨
- **Freeman Chain Code**: 
  - Deteksi kontur menggunakan OpenCV
  - Generate kode rantai Freeman 8-arah
  - Visualisasi kontur terbesar
  - Tampilkan area kontur dan jumlah titik
  - Parameter threshold yang bisa disesuaikan
  
- **Canny Edge Detection**:
  - Deteksi tepi menggunakan algoritma Canny
  - Gaussian blur preprocessing
  - Kontrol low dan high threshold
  - Visualisasi step-by-step (original → blurred → edges)
  
- **Integral Projection**:
  - Proyeksi horizontal dan vertikal
  - Visualisasi menggunakan GridSpec
  - Konversi otomatis ke citra biner
  - Cocok untuk analisis dokumen dan teks

## 🚀 Cara Menjalankan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Jalankan Aplikasi Streamlit (Face Dataset & Processing)
```bash
streamlit run app.py
```

### 3. Jalankan Aplikasi FastAPI (Image Operations & Edge Detection)
```bash
uvicorn main:app --reload --port 8000
```
Kemudian buka: http://localhost:8000

**Atau jika port 8000 sudah digunakan:**
```bash
uvicorn main:app --reload --port 8001
```
Kemudian buka: http://localhost:8001

## 📋 Persyaratan

- Python 3.8+
- Webcam (untuk fitur capture wajah)
- macOS: Pastikan memberikan izin kamera di System Preferences > Security & Privacy > Camera

## 📁 Struktur Folder

```
W5_TugasPCD_Kontur/
├── app.py                 # Aplikasi Streamlit (Face Dataset & Processing)
├── main.py                # Aplikasi FastAPI (Image Operations & Edge Detection)
├── 1.py                   # Script Freeman Chain Code (standalone)
├── 2.py                   # Script Canny Edge Detection (standalone)
├── 3.py                   # Script Integral Projection (standalone)
├── 4.py                   # (Reserved for future use)
├── requirements.txt       # Dependencies
├── dataset/              # Folder untuk menyimpan dataset wajah
│   └── [nama_person]/    # Folder per person
│       ├── img_0.jpg
│       ├── img_1.jpg
│       └── processed/    # Hasil pengolahan citra
├── static/
│   ├── uploads/          # Upload gambar sementara & hasil edge detection
│   └── histograms/       # Histogram yang dihasilkan
└── templates/            # Template HTML untuk FastAPI
    ├── base.html         # Template dasar dengan sidebar
    ├── home.html         # Halaman utama
    ├── grayscale.html    # Konversi grayscale
    ├── histogram.html    # Generate histogram
    ├── equalize.html     # Histogram equalization
    ├── specify.html      # Histogram specification
    ├── filtering.html    # Berbagai filter
    ├── edge_detection.html  # Edge detection (BARU!)
    ├── result.html       # Tampilan hasil
    └── statistics.html   # Statistik citra
```

## 🎯 Cara Menggunakan Edge Detection

### 1. Freeman Chain Code
1. Buka menu "Edge Detection" di sidebar
2. Pilih bagian "Freeman Chain Code"
3. Upload gambar (preferably dengan objek yang jelas)
4. Atur nilai threshold (default: 127)
5. Klik "Generate Freeman Chain Code"
6. Lihat hasil:
   - Citra asli (grayscale)
   - Citra biner hasil threshold
   - Kontur terbesar yang terdeteksi
   - Kode rantai Freeman (8-direction)

### 2. Canny Edge Detection
1. Buka menu "Edge Detection" di sidebar
2. Pilih bagian "Canny Edge Detection"
3. Upload gambar
4. Atur Low Threshold (default: 50)
5. Atur High Threshold (default: 150, biasanya 2-3x low threshold)
6. Klik "Detect Edges (Canny)"
7. Lihat hasil visualisasi 3 tahap:
   - Citra asli
   - Hasil Gaussian Blur
   - Hasil deteksi tepi Canny

### 3. Integral Projection
1. Buka menu "Edge Detection" di sidebar
2. Pilih bagian "Integral Projection"
3. Upload gambar
4. Atur nilai threshold untuk binarisasi (default: 127)
5. Klik "Generate Projection"
6. Lihat hasil:
   - Citra biner
   - Proyeksi horizontal (profil vertikal)
   - Proyeksi vertikal
   - Berguna untuk analisis dokumen dan segmentasi teks

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
