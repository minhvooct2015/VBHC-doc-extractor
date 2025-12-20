from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder

import tempfile
import os

from ocr_pipeline import (
    load_yolo_model_once,
    load_vietocr_detector_once,
    file_to_json_ocr,
    image_to_json_ocr,
)

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "YL02.pt"

app = FastAPI()

# Khởi tạo model & detector 1 lần
model = load_yolo_model_once(str(MODEL_PATH))
detector = load_vietocr_detector_once()

ALLOWED_PDF_TYPE = "application/pdf"
ALLOWED_IMAGE_TYPES = ["image/png", "image/jpeg", "image/tiff", "image/webp"]

@app.post("/extract1")
async def extract_json(pdf: UploadFile = File(...)):
    if pdf.content_type != ALLOWED_PDF_TYPE:
        return JSONResponse(content={"error": "File must be PDF"}, status_code=400)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
        temp_pdf.write(await pdf.read())
        temp_pdf_path = temp_pdf.name

    try:
        json_data = file_to_json_ocr(temp_pdf_path, model, detector)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        os.remove(temp_pdf_path)

    return JSONResponse(content=json_data)

@app.post("/extract")
async def extract_mixed_file(file: UploadFile = File(...)):
    content_type = file.content_type
    temp_file_path = None

    if content_type == ALLOWED_PDF_TYPE:
        processor_func = file_to_json_ocr
        suffix = ".pdf"
    elif content_type in ALLOWED_IMAGE_TYPES:
        processor_func = file_to_json_ocr  # file_to_json_ocr đã tự detect image/pdf
        suffix = os.path.splitext(file.filename)[-1].lower() or ".jpg"
    else:
        supported = ", ".join(ALLOWED_IMAGE_TYPES + [ALLOWED_PDF_TYPE])
        return JSONResponse(
            content={"error": f"Unsupported file type: {content_type}. Supported types: {supported}"},
            status_code=400,
        )

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_file.write(await file.read())
            temp_file_path = temp_file.name

        json_data = processor_func(temp_file_path, model, detector)
    except Exception as e:
        return JSONResponse(content={"error": f"Processing failed: {str(e)}"}, status_code=500)
    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)

    return JSONResponse(content=jsonable_encoder(json_data))

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # có thể giới hạn theo domain frontend
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)