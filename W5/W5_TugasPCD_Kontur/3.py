import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# --- 1. Buat Contoh Citra Biner dengan Teks ---

# Ukuran citra
height, width = 150, 400
# Buat canvas hitam
binary_img = np.zeros((height, width), dtype=np.uint8)

# Teks yang akan ditulis
text_lines = ["Baris Teks Satu", "Ini Baris Dua", "Testing 123"]
start_y = 40  # Posisi Y awal untuk baris pertama
line_height = 40 # Jarak vertikal antar baris
font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.8
font_color = 255  # Putih
thickness = 2

# Tulis teks ke citra
y = start_y
for line in text_lines:
    # Dapatkan ukuran teks untuk centering (opsional)
    # (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, thickness)
    # x = (width - text_width) // 2 # Center X
    x = 20 # Mulai dari kiri
    cv2.putText(binary_img, line, (x, y), font, font_scale, font_color, thickness)
    y += line_height

# --- 2. Hitung Proyeksi Integral ---

# Normalisasi citra ke 0 (latar) dan 1 (objek/teks)
# Ini penting agar hasil sum langsung merepresentasikan jumlah piksel
binary_norm = binary_img / 255.0

# Proyeksi Horizontal (jumlah per kolom -> Profil Vertikal)
# axis=0: menjumlahkan sepanjang dimensi baris (secara vertikal)
horizontal_projection = np.sum(binary_norm, axis=0)

# Proyeksi Vertikal (jumlah per baris -> Profil Horizontal)
# axis=1: menjumlahkan sepanjang dimensi kolom (secara horizontal)
vertical_projection = np.sum(binary_norm, axis=1)

# --- 3. Buat Plot Menggunakan Matplotlib dan GridSpec ---

# Buat figure dan axes dengan GridSpec untuk kontrol layout
fig = plt.figure(figsize=(10, 7)) # Sesuaikan ukuran figure jika perlu
# Grid 2x2, atur rasio tinggi & lebar, dan spasi
# Rasio tinggi: baris atas (proyeksi H) lebih pendek dari baris bawah (gambar)
# Rasio lebar: kolom kiri (gambar) lebih lebar dari kolom kanan (proyeksi V)
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

# Axes untuk citra biner (pojok kiri bawah - indeks [1, 0])
ax_img = fig.add_subplot(gs[1, 0])
ax_img.imshow(binary_img, cmap='gray', aspect='auto') # aspect='auto' agar tidak terdistorsi
ax_img.set_title('Contoh Citra Biner')
ax_img.set_xlabel('Indeks Kolom')
ax_img.set_ylabel('Indeks Baris (0 di atas)')

# Axes untuk Proyeksi Horizontal (di atas citra biner - indeks [0, 0])
# Bagikan sumbu X dengan plot citra agar sejajar
ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
ax_hproj.plot(np.arange(width), horizontal_projection, color='blue')
ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
ax_hproj.set_ylabel('Jumlah Piksel Putih')
# Sembunyikan label tick X karena sudah ada di plot bawah (citra)
plt.setp(ax_hproj.get_xticklabels(), visible=False)
ax_hproj.grid(axis='y', linestyle='--', alpha=0.6) # Grid bantu

# Axes untuk Proyeksi Vertikal (di kanan citra biner - indeks [1, 1])
# Bagikan sumbu Y dengan plot citra agar sejajar
ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
# Perhatikan: plot(nilai_proyeksi, indeks_baris)
ax_vproj.plot(vertical_projection, np.arange(height), color='red')
ax_vproj.set_title('Proyeksi Vertikal')
ax_vproj.set_xlabel('Jumlah Piksel Putih')
# Invert sumbu Y agar 0 ada di atas, cocok dengan citra
ax_vproj.invert_yaxis()
# Sembunyikan label tick Y karena sudah ada di plot kiri (citra)
plt.setp(ax_vproj.get_yticklabels(), visible=False)
ax_vproj.grid(axis='x', linestyle='--', alpha=0.6) # Grid bantu

# Judul keseluruhan
plt.suptitle("Visualisasi Proyeksi Integral pada Citra Teks", fontsize=14)

# Tampilkan plot
plt.show()

# (Opsional) Simpan citra biner jika diperlukan
# cv2.imwrite("contoh_teks_biner.png", binary_img)