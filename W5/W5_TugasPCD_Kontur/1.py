import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_freeman_chain_code(contour):
    """
    Menghasilkan Kode Rantai Freeman 8-arah dari kontur OpenCV.
    ASUMSI: kontur didapat dari findContours dengan CHAIN_APPROX_NONE.
    """
    chain_code = []
    if len(contour) < 2:
        return chain_code # Kontur harus punya minimal 2 titik

    # Pemetaan (dx, dy) ke kode arah Freeman (sumbu Y positif ke bawah)
    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    for i in range(len(contour)):
        p1 = contour[i][0] # Titik saat ini (format: [[x, y]])
        # Dapatkan titik berikutnya, gunakan modulo % untuk kembali ke titik awal
        # pada iterasi terakhir (menangani kontur tertutup).
        p2 = contour[(i + 1) % len(contour)][0]

        dx = p2[0] - p1[0] # Perbedaan X
        dy = p2[1] - p1[1] # Perbedaan Y (Ingat: Y positif ke bawah)

        norm_dx = np.sign(dx)
        norm_dy = np.sign(dy)

        code = directions.get((norm_dx, norm_dy))
        if code is not None:
            chain_code.append(code)

    return chain_code

# --- Alur Proses Utama ---
# 1. Pemuatan Citra (langsung grayscale)
# Ganti dengan path citra Anda yang valid
img_path = 'hurufA.png' # <--- GANTI PATH INI
try:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        # Coba path relatif jika absolut gagal (atau sebaliknya)
        # Jika tetap gagal, lempar error
        raise FileNotFoundError(f"Citra tidak ditemukan atau tidak dapat dibaca di: {img_path}")
except Exception as e:
    print(f"Error saat memuat citra: {e}")
    print("Pastikan path citra sudah benar dan file citra tidak rusak.")
    exit()


# 2. Binarisasi (Sesuaikan threshold & type berdasarkan citra Anda)
threshold_value = 127
# Asumsi objek gelap di latar terang (misal huruf hitam di kertas putih)
_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)
# Jika objek terang di latar gelap, gunakan:
# _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

# 3. Deteksi Kontur (Wajib CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# --- Persiapan Visualisasi dengan Matplotlib ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8)) # Buat grid 2x2

# Plot Citra Asli (Grayscale)
axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Citra Asli (Grayscale)')
axs[0, 0].axis('off') # Sembunyikan sumbu

# Plot Citra Biner
axs[0, 1].imshow(binary_img, cmap='gray')
axs[0, 1].set_title('Citra Biner (Hasil Threshold)')
axs[0, 1].axis('off')

# Variabel untuk menyimpan hasil jika kontur ditemukan
chain_code_str = "Tidak ada kontur ditemukan."
img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Default display (jika tak ada kontur)

# 4. Proses Kontur dan Generasi Kode Rantai (jika kontur ada)
if contours:
    # Pilih kontur terbesar
    largest_contour = max(contours, key=cv2.contourArea)

    # Gambar kontur pada citra BGR untuk visualisasi warna
    # Buat salinan BGR dari citra asli
    img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 1) # Gambar kontur hijau

    # Generasi Kode Rantai
    chain_code_result = generate_freeman_chain_code(largest_contour)
    chain_code_list_str = ', '.join(map(str, chain_code_result))

    # Format teks kode rantai untuk ditampilkan (dengan wrapping sederhana)
    max_line_len = 70 # Maks karakter per baris
    wrapped_code = ""
    current_line_len = 0
    for i, code_str in enumerate(map(str, chain_code_result)):
        item = code_str + (", " if i < len(chain_code_result) - 1 else "")
        if current_line_len + len(item) > max_line_len:
            wrapped_code += "\n"
            current_line_len = 0
        wrapped_code += item
        current_line_len += len(item)


    chain_code_str = (
        f"Jumlah Kontur Total: {len(contours)}\n"
        f"Kode Rantai Kontur Terbesar (Panjang {len(chain_code_result)}):\n"
        f"{wrapped_code}"
    )

    # Print ke konsol juga (opsional)
    # print(f"Jumlah Kontur Ditemukan: {len(contours)}")
    # print(f"Kode Rantai Kontur Terbesar (Panjang {len(chain_code_result)}):")
    # print(chain_code_result) # Print list mentah

# Plot Citra dengan Kontur Terdeteksi
# Konversi BGR (OpenCV) ke RGB (Matplotlib) sebelum display
img_rgb_display = cv2.cvtColor(img_contour_display, cv2.COLOR_BGR2RGB)
axs[1, 0].imshow(img_rgb_display)
axs[1, 0].set_title('Kontur Terbesar Terdeteksi')
axs[1, 0].axis('off')

# Area untuk menampilkan Teks Kode Rantai
axs[1, 1].axis('off') # Sembunyikan sumbu plot teks
# Tampilkan teks di area plot keempat
axs[1, 1].text(0.05, 0.95, # Posisi x, y (0-1 relatif thd axes)
               chain_code_str, # Teks yang ditampilkan
               ha='left', # Horizontal alignment
               va='top', # Vertical alignment
               fontsize=9, # Ukuran font
               wrap=True) # Coba wrap otomatis (tergantung backend matplotlib)
axs[1, 1].set_title('Hasil Kode Rantai')


# --- Tampilkan Figure Matplotlib ---
plt.tight_layout(pad=1.5) # Atur layout agar tidak tumpang tindih
plt.suptitle("Analisis Kode Rantai", fontsize=16) # Judul keseluruhan
plt.subplots_adjust(top=0.92) # Beri ruang untuk suptitle
plt.show()