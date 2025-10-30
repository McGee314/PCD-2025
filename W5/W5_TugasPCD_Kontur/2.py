import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Pemuatan Citra
img_path = 'cameraman.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Citra tidak ditemukan di: {img_path}")

# 2. Konversi ke Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Reduksi Noise (Langkah pra-pemrosesan penting!)
# Eksperimen dengan ukuran kernel (ksize) dan sigma
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Kernel 5x5 umum digunakan

# 4. Deteksi Tepi Canny
# Nilai threshold SANGAT bergantung pada citra. Perlu eksperimen!
low_threshold = 50
high_threshold = 150 # Aturan umum: high sekitar 2x-3x low
edges = cv2.Canny(blurred, low_threshold, high_threshold)

# 5. Visualisasi Hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Tampilkan citra asli (konversi warna BGR->RGB)
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray') # Tampilkan hasil blur
plt.title('Grayscale + Gaussian Blur')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray') # Tampilkan peta tepi hasil Canny
plt.title(f'Tepi Canny (Th={low_threshold},{high_threshold})')
plt.axis('off')

plt.tight_layout()
plt.show()