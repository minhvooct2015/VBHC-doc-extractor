#!/usr/bin/env python3
"""
OCR pipeline with added debug logging to help trace execution, timings, and data shapes.
Save this file and run your existing script logic with this module (or replace your current script).
"""

import os
import shutil
import logging
import time
import traceback
from pathlib import Path
from typing import List, Dict, Union, IO, Tuple

from ultralytics import YOLO
from pdf2image import convert_from_path, convert_from_bytes
from PIL import Image
import cv2
import numpy as np
import json
import re
import torch
from vietocr.tool.config import Cfg
from vietocr.tool.predictor import Predictor

# ===== Logging setup =====
LOG_LEVEL = os.environ.get("OCR_LOG_LEVEL", "DEBUG").upper()
LOG_TO_FILE = os.environ.get("OCR_LOG_TO_FILE", "")  # path to log file, if set logs to file
LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
if LOG_TO_FILE:
    logging.basicConfig(filename=LOG_TO_FILE, level=LOG_LEVEL, format=LOG_FORMAT)
else:
    logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)

logger = logging.getLogger("ocr_pipeline")

# ===== Constants =====
CLASS_NAMES = [
    "code",
    "documentTitle",
    "documentType",
    "issuanceDate",
    "issuingAgency",
    "signatory",
    "signatoryPosition"
]
DETECT2CLASS = {
    "code": 0,
    "signatory": 1,
    "signatoryPosition": 1,
}
CLASS_FOLDER = {
    0: "Front",
    1: "Last",
    2: "Others"
}

# ===== Helpers =====
def remove_folder_if_exists(root_folder: str):
    if os.path.exists(root_folder) and os.path.isdir(root_folder):
        logger.debug("Removing existing folder: %s", root_folder)
        shutil.rmtree(root_folder)

def load_yolo_model_once(model_path: str):
    logger.info("Loading YOLO model from: %s", model_path)
    t0 = time.time()
    model = YOLO(model_path)
    logger.info("YOLO model loaded in %.3fs", time.time() - t0)
    return model

# ======== VIETOCR LOAD & PREDICT =========
def predict_vietocr(detector: Predictor, image: Image.Image) -> Tuple[str, float]:
    """Single-image predict wrapper with logging and exception handling."""
    try:
        t0 = time.time()
        pred, prob = detector.predict(image, return_prob=True)
        logger.debug("VietOCR predict: text='%s' prob=%s time=%.3fs", pred, prob, time.time() - t0)
        return pred, prob
    except Exception as e:
        logger.exception("VietOCR prediction failed: %s", e)
        return "", 0.0

def load_model_vietocr():
    logger.info("Loading VietOCR predictor...")
    config = Cfg.load_config_from_name('vgg_transformer')
    config['device'] = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.debug("VietOCR config device: %s", config['device'])
    detector = Predictor(config)
    logger.info("VietOCR predictor ready")
    return detector

def load_vietocr_detector_once():
    return load_model_vietocr()

def batch_ocr_predict(detector: Predictor, pil_rois: List[Image.Image]):
    results = []
    logger.debug("Batch OCR predict called with %d ROIs", len(pil_rois))
    t_all = time.time()
    for idx, roi in enumerate(pil_rois):
        t0 = time.time()
        pred, prob = predict_vietocr(detector, roi)
        elapsed = time.time() - t0
        logger.debug("OCR roi[%d] time=%.3fs text_len=%d", idx, elapsed, len(pred))
        results.append((pred, prob))
    logger.info("Batch OCR completed %d ROIs in %.3fs", len(pil_rois), time.time() - t_all)
    return results

# ======== END VIETOCR LOAD & PREDICT =========

def classify_by_detection(detected_classes: List[str]) -> int:
    if "code" in detected_classes:
        return 0
    if any(c in detected_classes for c in ["signatory", "signatoryPosition"]):
        return 1
    return 2
# Đề xuất: dùng DPI thấp hơn (150–200) mặc định, đủ tốt cho OCR nhưng nhẹ RAM hơn rất nhiều:
def pdf_to_images(pdf_path_or_file: Union[str, Path, IO[bytes]], dpi=200):
    logger.debug("pdf_to_images called with %s (type=%s) dpi=%d", getattr(pdf_path_or_file, "name", str(pdf_path_or_file)), type(pdf_path_or_file), dpi)
    # Accept path or file-like object
    t0 = time.time()
    if hasattr(pdf_path_or_file, "read"):
        pdf_path_or_file.seek(0)
        pdf_bytes = pdf_path_or_file.read()
        images = convert_from_bytes(pdf_bytes, dpi=dpi)
        basename = "document"
        logger.info("Converted PDF bytes to %d images in %.3fs", len(images), time.time() - t0)
        return images, basename
    else:
        images = convert_from_path(str(pdf_path_or_file), dpi=dpi)
        basename = Path(pdf_path_or_file).stem
        logger.info("Converted PDF path '%s' to %d images in %.3fs", pdf_path_or_file, len(images), time.time() - t0)
        return images, basename

def is_image_file(filepath_or_file):
    logger.debug("is_image_file check for: %s", getattr(filepath_or_file, "name", str(filepath_or_file)))
    if hasattr(filepath_or_file, "read"):
        try:
            pos = filepath_or_file.tell()
            img = Image.open(filepath_or_file)
            img.verify()
            filepath_or_file.seek(pos)
            logger.debug("File-like object is a valid image")
            return True
        except Exception:
            filepath_or_file.seek(0)
            logger.debug("File-like object is not an image")
            return False
    else:
        ext = Path(filepath_or_file).suffix.lower()
        is_img = ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        logger.debug("Path extension '%s' -> is_image=%s", ext, is_img)
        return is_img

# def batch_detect_images(model, pil_images: List[Image.Image]):
#     logger.debug("batch_detect_images called with %d images", len(pil_images))
#     np_images = [np.array(img.convert('RGB')) for img in pil_images]
#     for i, np_img in enumerate(np_images):
#         logger.debug("Image[%d] shape: %s dtype=%s", i, np_img.shape, np_img.dtype)
#     t0 = time.time()
#     results = model(np_images)
#     logger.info("YOLO model inference on %d images done in %.3fs", len(np_images), time.time() - t0)
#     all_detected_classes = []
#     all_boxes = []
#     for img_idx, result in enumerate(results):
#         detected_classes = set()
#         boxes = []
#         try:
#             box_count = len(result.boxes)
#         except Exception:
#             # For compatibility: check attributes
#             box_count = getattr(result.boxes, "xyxy", []).shape[0] if getattr(result.boxes, "xyxy", None) is not None else 0
#         logger.debug("Image[%d] detected %d boxes", img_idx, box_count)
#         for i, cls_idx in enumerate(result.boxes.cls):
#             cls_idx = int(cls_idx)
#             cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"unknown_{cls_idx}"
#             detected_classes.add(cls_name)
#             xyxy = result.boxes.xyxy[i].cpu().numpy()
#             boxes.append((cls_idx, xyxy))
#             logger.debug("Image[%d] box[%d] cls=%s xyxy=%s", img_idx, i, cls_name, xyxy.tolist())
#         all_detected_classes.append(list(detected_classes))
#         all_boxes.append(boxes)
#     return all_detected_classes, all_boxes

def batch_detect_images(model, pil_images: List[Image.Image], batch_size: int = 4):
    logger.debug("batch_detect_images called with %d images (batch_size=%d)", len(pil_images), batch_size)
    all_detected_classes = []
    all_boxes = []

    t_total = time.time()
    for start in range(0, len(pil_images), batch_size):
        end = min(start + batch_size, len(pil_images))
        batch = pil_images[start:end]

        np_images = [np.array(img.convert("RGB")) for img in batch]
        for i, np_img in enumerate(np_images):
            logger.debug(
                "Image[%d] shape: %s dtype=%s",
                start + i, np_img.shape, np_img.dtype
            )

        t0 = time.time()
        results = model(np_images)
        logger.info(
            "YOLO inference on images[%d:%d] done in %.3fs",
            start, end, time.time() - t0
        )

        # Gộp kết quả batch vào list chung
        for img_idx, result in enumerate(results):
            global_idx = start + img_idx
            detected_classes = set()
            boxes = []

            try:
                box_count = len(result.boxes)
            except Exception:
                box_count = getattr(result.boxes, "xyxy", []).shape[0] if getattr(result.boxes, "xyxy", None) is not None else 0

            logger.debug("Image[%d] detected %d boxes", global_idx, box_count)

            for i, cls_idx in enumerate(result.boxes.cls):
                cls_idx = int(cls_idx)
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"unknown_{cls_idx}"
                detected_classes.add(cls_name)
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                boxes.append((cls_idx, xyxy))
                logger.debug(
                    "Image[%d] box[%d] cls=%s xyxy=%s",
                    global_idx, i, cls_name, xyxy.tolist()
                )

            all_detected_classes.append(list(detected_classes))
            all_boxes.append(boxes)

        # Giải phóng batch khỏi RAM
        del np_images
        del results

    logger.info(
        "YOLO model inference on %d images (batch_size=%d) done in %.3fs",
        len(pil_images), batch_size, time.time() - t_total
    )
    return all_detected_classes, all_boxes
    
def save_yolo_structure(images: List[Image.Image], basename: str, model, out_root: str):
    logger.info("Saving YOLO structure to '%s' (basename=%s)", out_root, basename)
    remove_folder_if_exists(out_root)
    os.makedirs(out_root, exist_ok=True)
    detected_classes_list, boxes_list = batch_detect_images(model, images)
    summary = []
    for idx, (img, detected_classes, boxes) in enumerate(zip(images, detected_classes_list, boxes_list)):
        class_idx = classify_by_detection(detected_classes)
        folder_name = CLASS_FOLDER[class_idx]
        out_dir = os.path.join(out_root, folder_name)
        os.makedirs(out_dir, exist_ok=True)
        img_name = f"{basename}_{idx}.jpg"
        label_name = f"{basename}_{idx}.txt"
        img_path = os.path.join(out_dir, img_name)
        label_path = os.path.join(out_dir, label_name)
        img.save(img_path)
        w, h = img.size
        logger.debug("Saved image %s size=(%d,%d) to %s", img_name, w, h, out_dir)
        lines = []
        for box in boxes:
            cls_idx = box[0]
            xyxy = box[1]
            x1, y1, x2, y2 = xyxy[:4]
            x_center = ((x1 + x2) / 2) / w
            y_center = ((y1 + y2) / 2) / h
            bbox_w = (x2 - x1) / w
            bbox_h = (y2 - y1) / h
            line = f"{cls_idx} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}"
            lines.append(line)
        with open(label_path, "w") as f:
            f.write("\n".join(lines))
        logger.debug("Wrote label file %s with %d entries", label_path, len(lines))
        summary.append({
            "img_path": img_path,
            "label_path": label_path,
            "class_folder": folder_name,
            "boxes": lines,
            "class_type": folder_name
        })
    logger.info("YOLO structure saved with %d images", len(summary))
    return summary

def read_yolo_label(label_path, img_shape):
    h, w = img_shape[:2]
    logger.debug("Reading yolo label %s for image shape (h=%d,w=%d)", label_path, h, w)
    boxes = []
    if not os.path.exists(label_path):
        logger.warning("Label file does not exist: %s", label_path)
        return boxes
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                logger.debug("Skipping malformed line in %s: %s", label_path, line.strip())
                continue
            try:
                class_id = int(parts[0])
                x_center = float(parts[1]) * w
                y_center = float(parts[2]) * h
                box_w = float(parts[3]) * w
                box_h = float(parts[4]) * h
                x1 = int(x_center - box_w / 2)
                y1 = int(y_center - box_h / 2)
                x2 = int(x_center + box_w / 2)
                y2 = int(y_center + box_h / 2)
                boxes.append({
                    "class_id": class_id,
                    "class_name": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"unknown_{class_id}",
                    "xyxy": [x1, y1, x2, y2],
                    "x_center": x_center,
                    "y_center": y_center
                })
            except Exception:
                logger.exception("Error parsing label line: %s", line.strip())
    logger.debug("Parsed %d boxes from %s", len(boxes), label_path)
    return boxes

def group_and_sort_bboxes_rowwise(boxes, row_threshold=15):
    boxes = sorted(boxes, key=lambda b: b['y_center'])
    rows = []
    for box in boxes:
        placed = False
        for row in rows:
            if abs(row[0]['y_center'] - box['y_center']) < row_threshold:
                row.append(box)
                placed = True
                break
        if not placed:
            rows.append([box])
    sorted_rows = [sorted(row, key=lambda b: b['x_center']) for row in rows]
    logger.debug("Grouped into %d rows (row_threshold=%d)", len(sorted_rows), row_threshold)
    return sorted_rows

def get_all_images_and_labels(root_folder):
    pairs = []
    for sub in ["Front", "Last"]:
        folder = Path(root_folder) / sub
        if not folder.exists():
            logger.debug("Skipping missing folder: %s", folder)
            continue
        for img_file in folder.glob("*.jpg"):
            label_file = img_file.with_suffix(".txt")
            if label_file.exists():
                pairs.append((str(img_file), str(label_file)))
            else:
                logger.debug("Missing label for image: %s", img_file)
    logger.info("Found %d image-label pairs under %s", len(pairs), root_folder)
    return pairs

def extract_date_values_String(text):
    if text == "":
        return ""
    ngay_match = re.search(r'(?:ngày\s*)?(\d{1,2})', text)
    thang_match = re.search(r'tháng\s+(\d{1,2})', text)
    nam_match = re.search(r'năm\s+(\d{4})', text)
    ngay = ngay_match.group(1) if ngay_match else None
    thang = thang_match.group(1) if thang_match else None
    nam = nam_match.group(1) if nam_match else None
    if None in (ngay, thang, nam):
        logger.debug("Date parsing incomplete for text: %s", text)
        return text
    return "ngày " + ngay + " tháng " + thang + " năm " + nam

def remove_before_first_space(s):
    return s.split(' ', 1)[-1]

def ocr_and_group_text(image_path, label_path, detector):
    logger.info("Running ocr_and_group_text on %s with labels %s", image_path, label_path)
    img = cv2.imread(image_path)
    if img is None:
        logger.error("Failed to read image: %s", image_path)
        return []
    boxes = read_yolo_label(label_path, img.shape)
    if not boxes:
        logger.debug("No boxes to OCR for %s", image_path)
        return []
    row_groups = group_and_sort_bboxes_rowwise(boxes)
    pil_rois = []
    roi_info = []
    for r_idx, row in enumerate(row_groups):
        for b_idx, box in enumerate(row):
            x1, y1, x2, y2 = box["xyxy"]
            class_name = box["class_name"]
            # clamp coordinates
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
            roi = img[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                logger.debug("Skipping empty ROI row=%d box=%d coords=%s", r_idx, b_idx, box["xyxy"])
                continue
            roi_pil = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            pil_rois.append(roi_pil)
            roi_info.append(class_name)
            logger.debug("Prepared ROI row=%d box=%d class=%s shape=%s", r_idx, b_idx, class_name, roi.shape)
    if pil_rois:
        pred_results = batch_ocr_predict(detector, pil_rois)
    else:
        pred_results = []
    class_text_dict = dict()
    for class_name, (text, prob) in zip(roi_info, pred_results):
        text = text.strip()
        logger.debug("OCR result class=%s text='%s' prob=%s", class_name, text, prob)
        if text:
            class_text_dict.setdefault(class_name, []).append({"text": text, "prob": prob})
    final_result = []
    for class_name, text_chunks in class_text_dict.items():
        full_text = " ".join([chunk["text"] for chunk in text_chunks])
        probs = [chunk["prob"] for chunk in text_chunks]
        if class_name == "issuanceDate":
            full_text = extract_date_values_String(full_text)
        if class_name == "code":
            full_text = remove_before_first_space(full_text)
        final_result.append({
            "class": class_name,
            "text": full_text,
            "probs": probs
        })
        logger.debug("Finalized OCR for class=%s text_len=%d probs=%s", class_name, len(full_text), probs)
    return final_result

def pdf_to_json_ocr(pdf_path_or_file, model, detector, temp_yolo_folder='./yolo_data', remove_temp=True):
    logger.info("pdf_to_json_ocr start for %s", getattr(pdf_path_or_file, "name", str(pdf_path_or_file)))
    t_all = time.time()
    images, basename = pdf_to_images(pdf_path_or_file)
    save_yolo_structure(images, basename, model, temp_yolo_folder)
    pairs = get_all_images_and_labels(temp_yolo_folder)
    def extract_page_idx(img_path):
        m = re.search(r'_(\d+)\.jpg', img_path)
        return int(m.group(1)) if m else 0
    pairs = sorted(pairs, key=lambda x: extract_page_idx(x[0]))
    result_dict = {}
    for img_path, label_path in pairs:
        try:
            ocr_results = ocr_and_group_text(img_path, label_path, detector)
            for item in ocr_results:
                class_name = item["class"]
                if class_name not in result_dict:
                    result_dict[class_name] = {
                        "class": class_name,
                        "text": item["text"],
                        "probs": item["probs"]
                    }
        except Exception:
            logger.exception("Error processing page %s, label %s", img_path, label_path)
    if remove_temp:
        remove_folder_if_exists(temp_yolo_folder)
    result_list = [result_dict[c] for c in CLASS_NAMES if c in result_dict]
    logger.info("pdf_to_json_ocr completed in %.3fs, found %d classes", time.time() - t_all, len(result_list))
    return {basename: result_list}

from pathlib import Path
from typing import Any, Dict, List, Union
from PIL import Image, UnidentifiedImageError
from io import BytesIO
import tempfile
import shutil
import logging

logger = logging.getLogger(__name__)
def image_to_json_ocr(
    image_file_or_path: Union[str, Path, Any],
    model: Any,
    detector: Any,
    temp_yolo_folder_prefix: str = "yolo_tmp_",
    remove_temp: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Convert an image (path or file-like) to OCR JSON results using a YOLO temp structure.

    Params:
    - image_file_or_path: path string/Path or file-like object with .read()
    - model: model parameter required by save_yolo_structure (kept for compatibility)
    - detector: OCR/detector used by ocr_and_group_text
    - temp_yolo_folder_prefix: prefix for temporary yolo folder (unique folder created each call)
    - remove_temp: whether to delete the temporary folder at the end

    Returns:
    - dict mapping basename -> list of ocr result dicts
    """
    name = getattr(image_file_or_path, "name", str(image_file_or_path))
    logger.info("image_to_json_ocr called for %s", name)

    # Prepare image object and basename
    try:
        if hasattr(image_file_or_path, "read"):
            # file-like: read bytes and open from BytesIO (safe and doesn't rely on seek)
            image_bytes = image_file_or_path.read()
            image = Image.open(BytesIO(image_bytes))
            basename = Path(getattr(image_file_or_path, "name", "document")).stem
        else:
            # path-like
            image = Image.open(str(image_file_or_path))
            basename = Path(image_file_or_path).stem
    except UnidentifiedImageError as e:
        logger.exception("Cannot identify image file: %s", e)
        raise
    except Exception as e:
        logger.exception("Error opening image: %s", e)
        raise

    # normalize image mode to avoid format issues (e.g., RGBA -> JPEG)
    try:
        if image.mode not in ("RGB", "L"):
            # convert preserves appearance; choose RGB for color images
            image = image.convert("RGB")
    except Exception:
        # fallback: ensure we have a usable image
        image = image.convert("RGB")

    # create a unique temporary folder for this call to avoid race conditions
    temp_dir = Path(tempfile.mkdtemp(prefix=temp_yolo_folder_prefix))
    logger.debug("Using temp yolo folder: %s", temp_dir)

    result_list = []
    try:
        # save_yolo_structure should write images/labels under temp_dir
        # Make sure save_yolo_structure accepts a Path or str
        save_yolo_structure([image], basename, model, str(temp_dir))

        # get all pairs from the temporary folder
        pairs = get_all_images_and_labels(str(temp_dir))

        for img_path, label_path in pairs:
            try:
                ocr_results = ocr_and_group_text(img_path, label_path, detector)
                if ocr_results:
                    result_list.extend(ocr_results)
            except Exception as e:
                # don't fail entire run for a single image/label pair
                logger.exception("OCR processing failed for %s / %s : %s", img_path, label_path, e)

    finally:
        if remove_temp:
            try:
                shutil.rmtree(temp_dir)
                logger.debug("Removed temp folder %s", temp_dir)
            except Exception as e:
                logger.warning("Failed to remove temp folder %s: %s", temp_dir, e)

    logger.info("image_to_json_ocr completed, found %d results for %s", len(result_list), basename)
    return {basename: result_list}

def file_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder='./yolo_data', remove_temp=True):
    logger.info("xxxxxx")
    logger.info("file_to_json_ocr called for %s", getattr(file_obj_or_path, "name", str(file_obj_or_path)))
    try:
        if hasattr(file_obj_or_path, "read"):
            file_obj_or_path.seek(0)
            header = file_obj_or_path.read(4)
            file_obj_or_path.seek(0)
            if header == b'%PDF':
                logger.debug("Detected PDF from file-like object")
                return pdf_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder, remove_temp)
            else:
                if is_image_file(file_obj_or_path):
                    logger.debug("Detected image from file-like object")
                    return image_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder, remove_temp)
                else:
                    raise ValueError("File-like object is not a supported image/PDF.")
        else:
            ext = Path(file_obj_or_path).suffix.lower()
            if ext == '.pdf':
                logger.debug("Detected PDF from path")
                return pdf_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder, remove_temp)
            elif is_image_file(file_obj_or_path):
                logger.debug("Detected image from path")
                return image_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder, remove_temp)
            else:
                raise ValueError(f"File type {ext} not supported for OCR.")
    except Exception:
        logger.exception("file_to_json_ocr failed for %s", getattr(file_obj_or_path, "name", str(file_obj_or_path)))
        raise
