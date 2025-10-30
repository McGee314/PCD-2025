import os
from uuid import uuid4
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import numpy as np
import cv2

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

if not os.path.exists("static/uploads"):
    os.makedirs("static/uploads")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.post("/upload/", response_class=HTMLResponse)
async def upload_image(request: Request, file: UploadFile = File(...)):
    image_data = await file.read()
    file_extension = file.filename.split(".")[-1]
    filename = f"{uuid4()}.{file_extension}"
    file_path = os.path.join("static", "uploads", filename)

    with open(file_path, "wb") as f:
        f.write(image_data)

    np_array = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
    b, g, r = cv2.split(img)

    rgb_array = {"R": r.tolist(), "G": g.tolist(), "B": b.tolist()}
    
    # Resize image jika terlalu besar untuk tampilan matriks
    max_display_size = 50  # Maximum 50x50 pixels untuk matriks
    height, width = img.shape[:2]
    
    if width > max_display_size or height > max_display_size:
        # Resize untuk display matriks
        scale = min(max_display_size / width, max_display_size / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        img_small = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        img_small = img.copy()
    
    # Convert BGR to RGB untuk display
    img_rgb = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)
    
    # Create pixel matrix data
    pixel_matrix = []
    matrix_height, matrix_width = img_rgb.shape[:2]
    
    for y in range(matrix_height):
        row = []
        for x in range(matrix_width):
            r_val, g_val, b_val = img_rgb[y, x]
            row.append({
                'r': int(r_val),
                'g': int(g_val),
                'b': int(b_val),
                'hex': '#{:02x}{:02x}{:02x}'.format(int(r_val), int(g_val), int(b_val))
            })
        pixel_matrix.append(row)
    
    # Calculate histogram data untuk chart
    # Hitung histogram untuk setiap channel
    hist_r = cv2.calcHist([r], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([g], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([b], [0], None, [256], [0, 256])
    
    # Convert ke list untuk JSON
    histogram_data = {
        'r': hist_r.flatten().tolist(),
        'g': hist_g.flatten().tolist(),
        'b': hist_b.flatten().tolist()
    }
    
    # Image info
    image_info = {
        'width': width,
        'height': height,
        'channels': img.shape[2] if len(img.shape) > 2 else 1,
        'matrix_width': matrix_width,
        'matrix_height': matrix_height
    }

    return templates.TemplateResponse("display.html", {
        "request": request,
        "image_path": f"/static/uploads/{filename}",
        "rgb_array": rgb_array,
        "pixel_matrix": pixel_matrix,
        "histogram_data": histogram_data,
        "image_info": image_info
    })
