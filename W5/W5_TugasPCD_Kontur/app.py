import streamlit as st
import cv2
import os
import numpy as np
import time
from PIL import Image

def detect_faces(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1, 
        minNeighbors=5, 
        minSize=(30, 30)
    )
    return faces

def add_salt_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    """Menambahkan noise salt and pepper ke gambar"""
    noisy_image = image.copy()
    
    # Salt noise (white pixels)
    salt_mask = np.random.random(image.shape[:2]) < salt_prob
    noisy_image[salt_mask] = 255
    
    # Pepper noise (black pixels)
    pepper_mask = np.random.random(image.shape[:2]) < pepper_prob
    noisy_image[pepper_mask] = 0
    
    return noisy_image

def remove_noise_median(image, kernel_size=5):
    """Menghilangkan noise menggunakan Median Filter"""
    return cv2.medianBlur(image, kernel_size)

def sharpen_image(image):
    """Melakukan penajaman citra menggunakan kernel sharpening"""
    # Kernel untuk sharpening
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened

# Judul Aplikasi
st.set_page_config(page_title="Face Dataset & Image Processing", layout="wide")

# Sidebar untuk navigasi
menu = st.sidebar.selectbox(
    "Pilih Menu",
    ["Tambah Dataset Wajah", "Pengolahan Citra Dataset"]
)

# ==================== MENU 1: TAMBAH DATASET WAJAH ====================
if menu == "Tambah Dataset Wajah":
    st.title("📸 Tambah Wajah Baru ke Dataset")
    st.markdown("---")

    # Input nama orang baru
    new_person = st.text_input("Masukkan nama orang baru:")

    # Tombol untuk memulai proses penambahan wajah
    capture = st.button("Tambahkan Wajah Baru")

    if capture:
        if not new_person:
            st.warning("Silakan masukkan nama orang baru.")
        else:
            save_path = os.path.join('dataset', new_person)
            
            if not os.path.exists('dataset'):
                os.makedirs('dataset')
                st.info("Folder 'dataset' telah dibuat.")
            
            if not os.path.exists(save_path):
                os.makedirs(save_path)
                st.success(f"Folder untuk {new_person} telah dibuat.")
                
                # Mulai menangkap gambar dari webcam
                cap = cv2.VideoCapture(0)
                
                # Set properties untuk webcam
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                cap.set(cv2.CAP_PROP_FPS, 30)
                
                if not cap.isOpened():
                    st.error("❌ Tidak dapat membuka webcam. Pastikan:")
                    st.error("1. Webcam terhubung dengan baik")
                    st.error("2. Tidak ada aplikasi lain yang menggunakan webcam")
                    st.error("3. Di macOS: System Preferences > Security & Privacy > Camera > izinkan Terminal/Python")
                else:
                    st.info("📹 Mencoba mengakses webcam...")
                    
                    # Baca beberapa frame pertama untuk warming up
                    for i in range(5):
                        ret, frame = cap.read()
                        time.sleep(0.1)
                    
                    # Test apakah berhasil membaca frame
                    ret, test_frame = cap.read()
                    
                    if not ret or test_frame is None:
                        st.error("❌ Webcam terbuka tetapi tidak dapat membaca frame.")
                        st.error("Periksa izin kamera di: System Preferences > Security & Privacy > Camera")
                        cap.release()
                    else:
                        st.success("✅ Webcam berhasil diinisialisasi!")
                        st.info(f"📸 Mulai menangkap {20} gambar wajah. Pastikan wajah Anda terlihat jelas di kamera.")
                        
                        num_images = 0
                        max_images = 20
                        frames_without_face = 0
                        max_frames_without_face = 100

                        frame_placeholder = st.empty()
                        progress_bar = st.progress(0)
                        status_text = st.empty()

                        try:
                            while num_images < max_images and frames_without_face < max_frames_without_face:
                                ret, frame = cap.read()
                                
                                if not ret or frame is None:
                                    continue

                                # Deteksi wajah dalam frame
                                faces = detect_faces(frame)

                                if len(faces) > 0:
                                    frames_without_face = 0
                                    
                                    for (x, y, w, h) in faces:
                                        face = frame[y:y+h, x:x+w]
                                        img_name = os.path.join(save_path, f"img_{num_images}.jpg")
                                        cv2.imwrite(img_name, face)
                                        num_images += 1

                                        # Menggambar kotak di sekitar wajah yang terdeteksi
                                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

                                        # Tampilkan hasil deteksi
                                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                        frame_placeholder.image(frame_rgb, channels="RGB", caption=f"Gambar {num_images}/{max_images}")

                                        # Update progress bar
                                        progress = num_images / max_images
                                        progress_bar.progress(progress)
                                        status_text.text(f"Menyimpan gambar {num_images} dari {max_images}...")

                                        # Hentikan setelah menyimpan satu wajah per frame
                                        break
                                else:
                                    frames_without_face += 1
                                    
                                    # Tampilkan frame dengan pesan
                                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                    cv2.putText(frame_rgb, "Tidak ada wajah terdeteksi", (10, 30), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                                    frame_placeholder.image(frame_rgb, channels="RGB", 
                                                          caption=f"Tidak ada wajah terdeteksi. Gambar: {num_images}/{max_images}")

                                # Small delay untuk mengurangi CPU usage
                                time.sleep(0.05)

                            if num_images >= max_images:
                                st.success(f"🎉 Berhasil! {num_images} gambar telah ditambahkan ke dataset {new_person}.")
                            else:
                                st.warning(f"⚠️ Proses dihentikan. {num_images} gambar tersimpan. Tidak ada wajah terdeteksi dalam waktu lama.")
                                
                        except Exception as e:
                            st.error(f"❌ Terjadi error: {str(e)}")
                        finally:
                            cap.release()
                            frame_placeholder.empty()
                            progress_bar.empty()
                            status_text.empty()
            else:
                st.warning("Nama sudah ada di dataset. Silakan pilih nama lain atau tambahkan lebih banyak gambar.")

# ==================== MENU 2: PENGOLAHAN CITRA DATASET ====================
elif menu == "Pengolahan Citra Dataset":
    st.title("🖼️ Pengolahan Citra Dataset Wajah")
    st.markdown("---")
    
    st.info("📌 **Proses Pengolahan:**\n1. Menambahkan Noise Salt and Pepper\n2. Menghilangkan Noise (Median Filter)\n3. Melakukan Penajaman Citra (Sharpening)")
    
    # Pilih folder dataset
    if os.path.exists('dataset'):
        folders = [f for f in os.listdir('dataset') if os.path.isdir(os.path.join('dataset', f))]
        
        if len(folders) > 0:
            selected_person = st.selectbox("Pilih Dataset Orang:", folders)
            
            # Tampilkan jumlah gambar
            person_path = os.path.join('dataset', selected_person)
            images = [f for f in os.listdir(person_path) if f.endswith(('.jpg', '.jpeg', '.png'))]
            
            st.success(f"📊 Ditemukan {len(images)} gambar dalam dataset **{selected_person}**")
            
            # Pilih gambar untuk diproses
            if len(images) > 0:
                selected_image = st.selectbox("Pilih Gambar untuk Diproses:", images)
                
                # Parameter noise
                st.markdown("### ⚙️ Pengaturan Parameter")
                col1, col2 = st.columns(2)
                
                with col1:
                    salt_prob = st.slider("Probabilitas Salt Noise:", 0.0, 0.1, 0.02, 0.01)
                    pepper_prob = st.slider("Probabilitas Pepper Noise:", 0.0, 0.1, 0.02, 0.01)
                
                with col2:
                    kernel_size = st.select_slider("Kernel Size Median Filter:", options=[3, 5, 7, 9], value=5)
                
                # Tombol proses
                if st.button("🚀 Proses Gambar", type="primary"):
                    image_path = os.path.join(person_path, selected_image)
                    
                    # Baca gambar
                    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        with st.spinner("Memproses gambar..."):
                            # Tahap 1: Tambahkan noise
                            img_noisy = add_salt_pepper_noise(img, salt_prob, pepper_prob)
                            
                            # Tahap 2: Hilangkan noise
                            img_denoised = remove_noise_median(img_noisy, kernel_size)
                            
                            # Tahap 3: Penajaman
                            img_sharpened = sharpen_image(img_denoised)
                            
                            st.success("✅ Proses selesai!")
                            
                            # Tampilkan hasil
                            st.markdown("### 📊 Hasil Pengolahan Citra")
                            
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.markdown("**1️⃣ Original**")
                                st.image(img, use_container_width=True, clamp=True)
                            
                            with col2:
                                st.markdown("**2️⃣ + Salt & Pepper Noise**")
                                st.image(img_noisy, use_container_width=True, clamp=True)
                            
                            with col3:
                                st.markdown("**3️⃣ Noise Removal**")
                                st.image(img_denoised, use_container_width=True, clamp=True)
                            
                            with col4:
                                st.markdown("**4️⃣ Sharpened**")
                                st.image(img_sharpened, use_container_width=True, clamp=True)
                            
                            # Opsi untuk menyimpan hasil
                            st.markdown("---")
                            save_results = st.checkbox("💾 Simpan hasil pengolahan?")
                            
                            if save_results:
                                output_folder = os.path.join(person_path, 'processed')
                                os.makedirs(output_folder, exist_ok=True)
                                
                                base_name = os.path.splitext(selected_image)[0]
                                
                                cv2.imwrite(os.path.join(output_folder, f"{base_name}_noisy.jpg"), img_noisy)
                                cv2.imwrite(os.path.join(output_folder, f"{base_name}_denoised.jpg"), img_denoised)
                                cv2.imwrite(os.path.join(output_folder, f"{base_name}_sharpened.jpg"), img_sharpened)
                                
                                st.success(f"✅ Hasil disimpan di: {output_folder}")
                    else:
                        st.error("❌ Gagal membaca gambar!")
            else:
                st.warning("⚠️ Tidak ada gambar dalam dataset ini.")
        else:
            st.warning("⚠️ Belum ada dataset. Silakan tambah dataset wajah terlebih dahulu.")
    else:
        st.warning("⚠️ Folder dataset belum dibuat. Silakan tambah dataset wajah terlebih dahulu.")