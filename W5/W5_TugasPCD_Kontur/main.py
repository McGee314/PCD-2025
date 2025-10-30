import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from skimage.exposure import match_histograms  # pastikan paket scikit-image sudah terinstal

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Set backend to Agg for server-side rendering
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

if not os.path.exists("static/histograms"):
    os.makedirs("static/histograms")

# Set matplotlib backend
matplotlib.use('Agg')


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})


@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    file_path = save_image(img, "uploaded")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": file_path,
        "modified_image_path": file_path,
        "back_url": "/"
    })


@app.post("/operation/", response_class=HTMLResponse)
async def perform_operation(
    request: Request,
    file: UploadFile = File(...),
    operation: str = Form(...),
    value: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    original_path = save_image(img, "original")

    if operation == "add":
        result_img = cv2.add(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "subtract":
        result_img = cv2.subtract(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "max":
        result_img = np.maximum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "min":
        result_img = np.minimum(img, np.full(img.shape, value, dtype=np.uint8))
    elif operation == "inverse":
        result_img = cv2.bitwise_not(img)
    else:
        return HTMLResponse("Operasi tidak dikenali.", status_code=400)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/"
    })


@app.post("/logic_operation/", response_class=HTMLResponse)
async def perform_logic_operation(
    request: Request,
    file1: UploadFile = File(...),
    file2: UploadFile = File(None),
    operation: str = Form(...)
):
    image_data1 = await file1.read()
    np_array1 = np.frombuffer(image_data1, np.uint8)
    img1 = cv2.imdecode(np_array1, cv2.IMREAD_COLOR)

    original_path = save_image(img1, "original")

    if operation == "not":
        result_img = cv2.bitwise_not(img1)
    else:
        if file2 is None:
            return HTMLResponse("Operasi AND dan XOR memerlukan dua gambar.", status_code=400)
        image_data2 = await file2.read()
        np_array2 = np.frombuffer(image_data2, np.uint8)
        img2 = cv2.imdecode(np_array2, cv2.IMREAD_COLOR)

        # Resize img2 agar memiliki dimensi yang sama dengan img1
        if img2.shape != img1.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        if operation == "and":
            result_img = cv2.bitwise_and(img1, img2)
        elif operation == "xor":
            result_img = cv2.bitwise_xor(img1, img2)
        else:
            return HTMLResponse("Operasi logika tidak dikenali.", status_code=400)

    modified_path = save_image(result_img, "modified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/"
    })


@app.get("/grayscale/", response_class=HTMLResponse)
async def grayscale_form(request: Request):
    # Menampilkan form untuk upload gambar ke grayscale
    return templates.TemplateResponse("grayscale.html", {"request": request})


@app.post("/grayscale/", response_class=HTMLResponse)
async def convert_grayscale(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    original_path = save_image(img, "original")
    modified_path = save_image(gray_img, "grayscale")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/grayscale/"
    })


@app.get("/histogram/", response_class=HTMLResponse)
async def histogram_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk histogram
    return templates.TemplateResponse("histogram.html", {"request": request})


@app.post("/histogram/", response_class=HTMLResponse)
async def generate_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Pastikan gambar berhasil diimpor
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar yang diunggah", status_code=400)

    # Buat histogram grayscale dan berwarna
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    grayscale_histogram_path = save_histogram(gray_img, "grayscale")

    color_histogram_path = save_color_histogram(img)

    return templates.TemplateResponse("histogram.html", {
        "request": request,
        "grayscale_histogram_path": grayscale_histogram_path,
        "color_histogram_path": color_histogram_path
    })



@app.get("/equalize/", response_class=HTMLResponse)
async def equalize_form(request: Request):
    # Menampilkan halaman untuk upload gambar untuk equalisasi histogram
    return templates.TemplateResponse("equalize.html", {"request": request})


@app.post("/equalize/", response_class=HTMLResponse)
async def equalize_histogram(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    equalized_img = cv2.equalizeHist(img)

    original_path = save_image(img, "original")
    modified_path = save_image(equalized_img, "equalized")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/equalize/"
    })


@app.get("/specify/", response_class=HTMLResponse)
async def specify_form(request: Request):
    # Menampilkan halaman untuk upload gambar dan referensi untuk spesifikasi histogram
    return templates.TemplateResponse("specify.html", {"request": request})


@app.post("/specify/", response_class=HTMLResponse)
async def specify_histogram(request: Request, file: UploadFile = File(...), ref_file: UploadFile = File(...)):
    # Baca gambar yang diunggah dan gambar referensi
    image_data = await file.read()
    ref_image_data = await ref_file.read()

    np_array = np.frombuffer(image_data, np.uint8)
    ref_np_array = np.frombuffer(ref_image_data, np.uint8)

    # jika ingin menggunakan grayscale, aktifkan baris di bawah ini
    # img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    # ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_GRAYSCALE)

    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)  # Membaca gambar dalam format BGR
    ref_img = cv2.imdecode(ref_np_array, cv2.IMREAD_COLOR)  # Membaca gambar referensi dalam format BGR

    if img is None or ref_img is None:
        return HTMLResponse("Gambar utama atau gambar referensi tidak dapat dibaca.", status_code=400)

    # Spesifikasi histogram menggunakan match_histograms dari skimage untuk gambar berwarna
    specified_img = match_histograms(img, ref_img, channel_axis=-1)
    # Konversi kembali ke format uint8 jika diperlukan
    specified_img = np.clip(specified_img, 0, 255).astype('uint8')

    original_path = save_image(img, "original")
    modified_path = save_image(specified_img, "specified")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/specify/"
    })


@app.post("/statistics/", response_class=HTMLResponse)
async def calculate_statistics(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)

    mean_intensity = np.mean(img)
    std_deviation = np.std(img)

    image_path = save_image(img, "statistics")

    return templates.TemplateResponse("statistics.html", {
        "request": request,
        "mean_intensity": mean_intensity,
        "std_deviation": std_deviation,
        "image_path": image_path
    })


# Filtering Routes
@app.get("/filtering/", response_class=HTMLResponse)
async def filtering_form(request: Request):
    return templates.TemplateResponse("filtering.html", {"request": request})


@app.post("/filtering/convolution/", response_class=HTMLResponse)
async def apply_convolution_filter(
    request: Request,
    file: UploadFile = File(...),
    kernel_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if kernel_type == "average":
        kernel = np.ones((3, 3), np.float32) / 9
    elif kernel_type == "sharpen":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    elif kernel_type == "edge":
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    else:
        return HTMLResponse("Kernel type tidak dikenali.", status_code=400)

    result_img = cv2.filter2D(img, -1, kernel)

    original_path = save_image(img, "original")
    modified_path = save_image(result_img, f"convolution_{kernel_type}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/filtering/"
    })


@app.post("/filtering/padding/", response_class=HTMLResponse)
async def apply_padding(
    request: Request,
    file: UploadFile = File(...),
    padding_size: int = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    padded_img = cv2.copyMakeBorder(
        img, padding_size, padding_size, padding_size, padding_size,
        cv2.BORDER_CONSTANT, value=[0, 0, 0]
    )

    original_path = save_image(img, "original")
    modified_path = save_image(padded_img, f"padded_{padding_size}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/filtering/"
    })


@app.post("/filtering/filter/", response_class=HTMLResponse)
async def apply_frequency_filter(
    request: Request,
    file: UploadFile = File(...),
    filter_type: str = Form(...)
):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    if filter_type == "low":
        filtered_img = cv2.GaussianBlur(img, (5, 5), 0)
    elif filter_type == "high":
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        filtered_img = cv2.filter2D(img, -1, kernel)
    elif filter_type == "band":
        low_pass = cv2.GaussianBlur(img, (9, 9), 0)
        high_pass = img - low_pass
        filtered_img = low_pass + high_pass
    else:
        return HTMLResponse("Filter type tidak dikenali.", status_code=400)

    original_path = save_image(img, "original")
    modified_path = save_image(filtered_img, f"filter_{filter_type}")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/filtering/"
    })


@app.post("/filtering/fourier/", response_class=HTMLResponse)
async def apply_fourier(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # Normalize to 0-255 for display
    magnitude_spectrum = np.uint8(
        255 * (magnitude_spectrum - magnitude_spectrum.min()) /
        (magnitude_spectrum.max() - magnitude_spectrum.min())
    )

    original_path = save_image(img, "original")
    modified_path = save_image(magnitude_spectrum, "fourier_transform")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/filtering/"
    })


@app.post("/filtering/denoise/", response_class=HTMLResponse)
async def reduce_noise(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)

    # Create a mask to remove periodic noise
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 30  # Radius of the mask
    mask[crow-r:crow+r, ccol-r:ccol+r] = 0

    fshift = fshift * mask
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # Normalize to 0-255
    img_back = np.uint8(255 * (img_back - img_back.min()) / (img_back.max() - img_back.min()))

    original_path = save_image(img, "original")
    modified_path = save_image(img_back, "denoised")

    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": original_path,
        "modified_image_path": modified_path,
        "back_url": "/filtering/"
    })


def save_image(image, prefix):
    filename = f"{prefix}_{uuid4()}.png"
    path = os.path.join("static/uploads", filename)
    cv2.imwrite(path, image)
    return f"/static/uploads/{filename}"


def save_histogram(image, prefix):
    histogram_path = f"static/histograms/{prefix}_{uuid4()}.png"
    plt.figure()
    plt.hist(image.ravel(), 256, [0, 256])
    plt.savefig(histogram_path)
    plt.close()
    return f"/{histogram_path}"


def save_color_histogram(image):
    color_histogram_path = f"static/histograms/color_{uuid4()}.png"
    plt.figure()
    for i, color in enumerate(['b', 'g', 'r']):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=color)
    plt.savefig(color_histogram_path)
    plt.close()
    return f"/{color_histogram_path}"


# ==================== EDGE DETECTION ROUTES ====================

def generate_freeman_chain_code(contour):
    """
    Menghasilkan Kode Rantai Freeman 8-arah dari kontur OpenCV.
    """
    chain_code = []
    if len(contour) < 2:
        return chain_code

    # Pemetaan (dx, dy) ke kode arah Freeman
    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    for i in range(len(contour)):
        current_point = contour[i][0]
        next_point = contour[(i + 1) % len(contour)][0]
        
        dx = next_point[0] - current_point[0]
        dy = next_point[1] - current_point[1]
        
        if (dx, dy) in directions:
            chain_code.append(directions[(dx, dy)])
    
    return chain_code


@app.get("/edge_detection/", response_class=HTMLResponse)
async def edge_detection_form(request: Request):
    return templates.TemplateResponse("edge_detection.html", {"request": request})


@app.post("/edge_detection/freeman/", response_class=HTMLResponse)
async def freeman_chain_code(
    request: Request, 
    file: UploadFile = File(...),
    threshold: int = Form(127)
):
    # Load image
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar", status_code=400)
    
    # Binarisasi
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY_INV)
    
    # Deteksi kontur
    contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Create visualization
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    
    # Original image
    axs[0, 0].imshow(img, cmap='gray')
    axs[0, 0].set_title('Citra Asli (Grayscale)')
    axs[0, 0].axis('off')
    
    # Binary image
    axs[0, 1].imshow(binary_img, cmap='gray')
    axs[0, 1].set_title('Citra Biner (Threshold)')
    axs[0, 1].axis('off')
    
    chain_code_str = "Tidak ada kontur ditemukan."
    img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    
    if contours:
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Draw contour
        cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 2)
        
        # Generate chain code
        chain_code = generate_freeman_chain_code(largest_contour)
        
        # Format chain code string
        chain_code_str = f"Kode Rantai Freeman:\n"
        chain_code_str += f"Jumlah titik: {len(largest_contour)}\n"
        chain_code_str += f"Area kontur: {cv2.contourArea(largest_contour):.2f}\n\n"
        chain_code_str += "Chain Code: "
        
        # Display first 100 codes
        if len(chain_code) > 100:
            chain_code_str += ''.join(map(str, chain_code[:100])) + f"...\n(Total: {len(chain_code)} kode)"
        else:
            chain_code_str += ''.join(map(str, chain_code))
    
    # Contour image
    img_rgb_display = cv2.cvtColor(img_contour_display, cv2.COLOR_BGR2RGB)
    axs[1, 0].imshow(img_rgb_display)
    axs[1, 0].set_title('Kontur Terbesar Terdeteksi')
    axs[1, 0].axis('off')
    
    # Chain code text
    axs[1, 1].axis('off')
    axs[1, 1].text(0.05, 0.95, chain_code_str,
                   ha='left', va='top', fontsize=9, 
                   family='monospace', wrap=True)
    axs[1, 1].set_title('Hasil Kode Rantai')
    
    plt.tight_layout(pad=1.5)
    plt.suptitle("Analisis Freeman Chain Code", fontsize=16)
    plt.subplots_adjust(top=0.92)
    
    # Save figure
    result_path = f"static/uploads/freeman_{uuid4()}.png"
    plt.savefig(result_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": save_image(img, "freeman_original"),
        "modified_image_path": f"/{result_path}",
        "back_url": "/edge_detection/"
    })


@app.post("/edge_detection/canny/", response_class=HTMLResponse)
async def canny_edge_detection(
    request: Request,
    file: UploadFile = File(...),
    low_threshold: int = Form(50),
    high_threshold: int = Form(150)
):
    # Load image
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar", status_code=400)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, low_threshold, high_threshold)
    
    # Create visualization
    fig = plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Citra Asli')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(blurred, cmap='gray')
    plt.title('Grayscale + Gaussian Blur')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(edges, cmap='gray')
    plt.title(f'Tepi Canny (Th={low_threshold},{high_threshold})')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Save figure
    result_path = f"static/uploads/canny_{uuid4()}.png"
    plt.savefig(result_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": save_image(img, "canny_original"),
        "modified_image_path": f"/{result_path}",
        "back_url": "/edge_detection/"
    })


@app.post("/edge_detection/projection/", response_class=HTMLResponse)
async def integral_projection(
    request: Request,
    file: UploadFile = File(...),
    threshold: int = Form(127)
):
    # Load image
    image_data = await file.read()
    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        return HTMLResponse("Tidak dapat membaca gambar", status_code=400)
    
    # Binarize
    _, binary_img = cv2.threshold(img, threshold, 255, cv2.THRESH_BINARY)
    
    # Normalize to 0-1
    binary_norm = binary_img / 255.0
    
    # Calculate projections
    horizontal_projection = np.sum(binary_norm, axis=0)
    vertical_projection = np.sum(binary_norm, axis=1)
    
    # Create visualization
    height, width = binary_img.shape
    fig = plt.figure(figsize=(12, 8))
    gs = gridspec.GridSpec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)
    
    # Binary image
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(binary_img, cmap='gray', aspect='auto')
    ax_img.set_title('Citra Biner')
    ax_img.set_xlabel('Indeks Kolom')
    ax_img.set_ylabel('Indeks Baris')
    
    # Horizontal projection
    ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img)
    ax_hproj.plot(np.arange(width), horizontal_projection, color='blue')
    ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
    ax_hproj.set_ylabel('Jumlah Piksel Putih')
    plt.setp(ax_hproj.get_xticklabels(), visible=False)
    ax_hproj.grid(axis='y', linestyle='--', alpha=0.6)
    
    # Vertical projection
    ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img)
    ax_vproj.plot(vertical_projection, np.arange(height), color='red')
    ax_vproj.set_title('Proyeksi Vertikal')
    ax_vproj.set_xlabel('Jumlah Piksel Putih')
    ax_vproj.invert_yaxis()
    plt.setp(ax_vproj.get_yticklabels(), visible=False)
    ax_vproj.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.suptitle("Visualisasi Proyeksi Integral", fontsize=14)
    
    # Save figure
    result_path = f"static/uploads/projection_{uuid4()}.png"
    plt.savefig(result_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    return templates.TemplateResponse("result.html", {
        "request": request,
        "original_image_path": save_image(img, "projection_original"),
        "modified_image_path": f"/{result_path}",
        "back_url": "/edge_detection/"
    })
