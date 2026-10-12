#!/usr/bin/env python3
"""
OCR pipeline for Vietnamese administrative documents: YOLO detects the fields,
VietOCR reads the text inside each detected box.

Entry point: file_to_json_ocr(file_obj_or_path, model, detector, ...).

PDF pages are rendered in small chunks (the next chunk renders in the background while
the current one is detected/OCR'd) and processed in memory. Nothing is written to disk
unless remove_temp=False, in which case the YOLO structure (Front/Last/Others images +
label files) is dumped exactly like the previous version did.

Load the YOLO model and the VietOCR predictor once and pass them to every call:
loading takes seconds and each extra copy holds its own (GPU) memory.
"""

import os
import shutil
import logging
import time
import re
import tempfile
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, IO, Iterator, List, Optional, Tuple, Union

from ultralytics import YOLO
from pdf2image import convert_from_path, convert_from_bytes, pdfinfo_from_path
from PIL import Image, UnidentifiedImageError
import cv2
import numpy as np
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

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        logger.warning("Invalid %s=%r, using default %d", name, os.environ.get(name), default)
        return default


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
# Only pages classified into these folders are OCR'd ("Others" pages are skipped).
OCR_FOLDERS = ("Front", "Last")

# VietOCR batch size. 1 = one ROI at a time (previous behavior).
# >1 uses Predictor.predict_batch, falling back to single predict on any error (e.g. CUDA OOM).
OCR_BATCH_SIZE = _env_int("OCR_BATCH_SIZE", 1)

# 1 = add "det_confs" (YOLO box confidences, parallel to "probs") to each result item.
# Off by default so the output schema stays exactly as before.
OCR_INCLUDE_DET_CONF = _env_int("OCR_INCLUDE_DET_CONF", 0) == 1

# 1 = when no documentType was detected, infer "Công văn" from a documentTitle starting
# with "V/v". Off by default so the output stays exactly as before.
OCR_INFER_DOC_TYPE = _env_int("OCR_INFER_DOC_TYPE", 0) == 1

# PDF pages rendered per poppler call. Each call spawns pdfinfo + pdftoppm and re-parses
# the PDF, so 1 is slow; larger values hold more rendered pages in RAM
# (~12 MB per A4 page at 200 DPI, x2 because the next chunk is prefetched).
OCR_PDF_PAGE_CHUNK = max(1, _env_int("OCR_PDF_PAGE_CHUNK", 4))


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
    check_class_names(model)
    return model

_checked_model_ids = set()

def check_class_names(model) -> bool:
    """
    Warn if the YOLO model's class names differ from CLASS_NAMES. Field labels come from
    CLASS_NAMES[cls_idx], so a different training order silently mislabels fields.
    Only logs; never changes results.
    """
    _checked_model_ids.add(id(model))
    try:
        names = getattr(model, "names", None)
        if names is None:
            logger.warning("Cannot verify YOLO class names: model has no 'names'")
            return False
        model_names = [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    except Exception as e:
        logger.warning("Cannot verify YOLO class names: %s", e)
        return False
    if model_names != CLASS_NAMES:
        logger.warning(
            "YOLO class names do not match CLASS_NAMES, field labels may be wrong.\n"
            "  model.names : %s\n  CLASS_NAMES : %s",
            model_names, CLASS_NAMES
        )
        return False
    logger.debug("YOLO class names match CLASS_NAMES")
    return True

def _imread_unicode(path: str):
    """cv2.imread cannot open non-ASCII paths on Windows (e.g. Vietnamese file names)."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)

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

def _predict_chunk(detector: Predictor, rois: List[Image.Image]) -> List[Tuple[str, float]]:
    try:
        with torch.inference_mode():
            texts, probs = detector.predict_batch(rois, return_prob=True)
        if len(texts) == len(rois) and len(probs) == len(rois):
            return [(t, float(p)) for t, p in zip(texts, probs)]
        logger.warning("predict_batch returned %d results for %d ROIs, falling back", len(texts), len(rois))
    except Exception as e:  # CUDA OOM, vietocr version without return_prob, ...
        logger.warning("predict_batch failed (%s), falling back to single predict", e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [predict_vietocr(detector, roi) for roi in rois]

def batch_ocr_predict(detector: Predictor, pil_rois: List[Image.Image], batch_size: Optional[int] = None):
    if batch_size is None:
        batch_size = OCR_BATCH_SIZE
    results = []
    logger.debug("Batch OCR predict called with %d ROIs (batch_size=%d)", len(pil_rois), batch_size)
    t_all = time.time()
    if batch_size <= 1:
        for idx, roi in enumerate(pil_rois):
            t0 = time.time()
            pred, prob = predict_vietocr(detector, roi)
            elapsed = time.time() - t0
            logger.debug("OCR roi[%d] time=%.3fs text_len=%d", idx, elapsed, len(pred))
            results.append((pred, prob))
    else:
        for start in range(0, len(pil_rois), batch_size):
            results.extend(_predict_chunk(detector, pil_rois[start:start + batch_size]))
    logger.info("Batch OCR completed %d ROIs in %.3fs", len(pil_rois), time.time() - t_all)
    return results

# ======== END VIETOCR LOAD & PREDICT =========

def classify_by_detection(detected_classes: List[str]) -> int:
    if "code" in detected_classes:
        return 0
    if any(c in detected_classes for c in ["signatory", "signatoryPosition"]):
        return 1
    return 2

def pdf_to_images(pdf_path_or_file: Union[str, Path, IO[bytes]], dpi=200):
    """Convert every page at once. Kept for callers; the pipeline itself uses _open_pdf_pages."""
    logger.debug("pdf_to_images called with %s (type=%s) dpi=%d", getattr(pdf_path_or_file, "name", str(pdf_path_or_file)), type(pdf_path_or_file), dpi)
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

def _render_pages(pdf_path: str, dpi: int, first: int, last: int) -> List[Image.Image]:
    t0 = time.time()
    pages = convert_from_path(pdf_path, dpi=dpi, first_page=first, last_page=last)
    logger.debug("Converted pages %d-%d in %.3fs", first, last, time.time() - t0)
    return pages

def _iter_pdf_chunks(pdf_path: str, dpi: int, chunk: Optional[int] = None) -> Iterator[List[Image.Image]]:
    """
    Yield the PDF's pages in chunks of `chunk` pages, in order. The next chunk is rendered in a
    background thread while the caller works on the current one (poppler runs as a subprocess,
    so rendering overlaps with YOLO/OCR). Closing the generator waits for an in-flight render.
    """
    chunk = OCR_PDF_PAGE_CHUNK if chunk is None else max(1, chunk)
    n_pages = pdfinfo_from_path(pdf_path)["Pages"]
    logger.debug("PDF %s has %d pages (chunk=%d)", pdf_path, n_pages, chunk)
    starts = list(range(1, n_pages + 1, chunk))
    if not starts:
        return
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="pdf_render") as pool:
        future = pool.submit(_render_pages, pdf_path, dpi, starts[0], min(starts[0] + chunk - 1, n_pages))
        for i in range(len(starts)):
            pages = future.result()
            if i + 1 < len(starts):
                nxt = starts[i + 1]
                future = pool.submit(_render_pages, pdf_path, dpi, nxt, min(nxt + chunk - 1, n_pages))
            yield pages

@contextmanager
def _open_pdf_pages(pdf_path_or_file: Union[str, Path, IO[bytes]], dpi=200):
    """Yield (basename, iterator over page chunks). At most two chunks are in memory at a time."""
    if hasattr(pdf_path_or_file, "read"):
        pdf_path_or_file.seek(0)
        # Poppler needs a file; convert_from_bytes writes a temp file internally too.
        with tempfile.TemporaryDirectory(prefix="ocr_pdf_") as tmp:
            pdf_path = os.path.join(tmp, "input.pdf")
            with open(pdf_path, "wb") as f:
                shutil.copyfileobj(pdf_path_or_file, f)
            chunks = _iter_pdf_chunks(pdf_path, dpi)
            try:
                yield "document", chunks
            finally:
                # Stop the background render before the temp file is deleted (Windows file lock).
                chunks.close()
    else:
        chunks = _iter_pdf_chunks(str(pdf_path_or_file), dpi)
        try:
            yield Path(pdf_path_or_file).stem, chunks
        finally:
            chunks.close()

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

def batch_detect_images(model, pil_images: List[Image.Image], batch_size: int = 4):
    """Returns (detected class names per image, [(cls_idx, xyxy), ...] per image)."""
    all_detected_classes, all_boxes, _ = _detect_images(model, pil_images, batch_size)
    return all_detected_classes, all_boxes

def _detect_images(model, pil_images: List[Image.Image], batch_size: int = 4):
    """Like batch_detect_images, plus the YOLO confidence of every box (parallel to boxes)."""
    if id(model) not in _checked_model_ids:
        check_class_names(model)
    logger.debug("batch_detect_images called with %d images (batch_size=%d)", len(pil_images), batch_size)
    all_detected_classes = []
    all_boxes = []
    all_confs = []

    t_total = time.time()
    for start in range(0, len(pil_images), batch_size):
        end = min(start + batch_size, len(pil_images))
        # Pass PIL images: Ultralytics treats numpy arrays as BGR, PIL as RGB.
        batch = [img if img.mode == "RGB" else img.convert("RGB") for img in pil_images[start:end]]
        for i, img in enumerate(batch):
            logger.debug("Image[%d] size: %s", start + i, img.size)

        t0 = time.time()
        results = model(batch)
        logger.info(
            "YOLO inference on images[%d:%d] done in %.3fs",
            start, end, time.time() - t0
        )

        for img_idx, result in enumerate(results):
            global_idx = start + img_idx
            detected_classes = set()
            boxes = []
            confs = []
            logger.debug("Image[%d] detected %d boxes", global_idx, len(result.boxes))

            for i, cls_idx in enumerate(result.boxes.cls):
                cls_idx = int(cls_idx)
                cls_name = CLASS_NAMES[cls_idx] if cls_idx < len(CLASS_NAMES) else f"unknown_{cls_idx}"
                detected_classes.add(cls_name)
                xyxy = result.boxes.xyxy[i].cpu().numpy()
                conf = float(result.boxes.conf[i])
                boxes.append((cls_idx, xyxy))
                confs.append(conf)
                logger.debug(
                    "Image[%d] box[%d] cls=%s conf=%.3f xyxy=%s",
                    global_idx, i, cls_name, conf, xyxy.tolist()
                )

            all_detected_classes.append(list(detected_classes))
            all_boxes.append(boxes)
            all_confs.append(confs)

        del batch
        del results

    logger.info(
        "YOLO model inference on %d images (batch_size=%d) done in %.3fs",
        len(pil_images), batch_size, time.time() - t_total
    )
    return all_detected_classes, all_boxes, all_confs

def _detect_page(model, page: Image.Image):
    detected_classes_list, boxes_list, confs_list = _detect_images(model, [page])
    return detected_classes_list[0], boxes_list[0], confs_list[0]

def _save_page_yolo(img: Image.Image, boxes, basename: str, idx: int, out_root: str, folder_name: str) -> Dict:
    """Write one page image + YOLO label file under out_root/folder_name."""
    out_dir = os.path.join(out_root, folder_name)
    os.makedirs(out_dir, exist_ok=True)
    img_name = f"{basename}_{idx}.jpg"
    label_name = f"{basename}_{idx}.txt"
    img_path = os.path.join(out_dir, img_name)
    label_path = os.path.join(out_dir, label_name)
    img.save(img_path)
    w, h = img.size
    logger.debug("Saved image %s size=(%d,%d) to %s", img_name, w, h, out_dir)
    lines = _yolo_label_lines(boxes, w, h)
    with open(label_path, "w") as f:
        f.write("\n".join(lines))
    logger.debug("Wrote label file %s with %d entries", label_path, len(lines))
    return {
        "img_path": img_path,
        "label_path": label_path,
        "class_folder": folder_name,
        "boxes": lines,
        "class_type": folder_name
    }

def save_yolo_structure(images: List[Image.Image], basename: str, model, out_root: str):
    logger.info("Saving YOLO structure to '%s' (basename=%s)", out_root, basename)
    remove_folder_if_exists(out_root)
    os.makedirs(out_root, exist_ok=True)
    detected_classes_list, boxes_list = batch_detect_images(model, images)
    summary = []
    for idx, (img, detected_classes, boxes) in enumerate(zip(images, detected_classes_list, boxes_list)):
        folder_name = CLASS_FOLDER[classify_by_detection(detected_classes)]
        summary.append(_save_page_yolo(img, boxes, basename, idx, out_root, folder_name))
    logger.info("YOLO structure saved with %d images", len(summary))
    return summary

def _yolo_label_lines(boxes, w: int, h: int) -> List[str]:
    """(cls_idx, xyxy) detections -> YOLO label lines (normalized, 6 decimals)."""
    lines = []
    for cls_idx, xyxy in boxes:
        x1, y1, x2, y2 = xyxy[:4]
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        bbox_w = (x2 - x1) / w
        bbox_h = (y2 - y1) / h
        lines.append(f"{cls_idx} {x_center:.6f} {y_center:.6f} {bbox_w:.6f} {bbox_h:.6f}")
    return lines

def _parse_yolo_lines(lines, w: int, h: int, source: str = "") -> List[Dict]:
    """YOLO label lines -> pixel boxes. The in-memory pipeline goes through the same
    label round-trip so crops stay pixel-identical to the previous on-disk version."""
    boxes = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            logger.debug("Skipping malformed line in %s: %s", source, line.strip())
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
    return boxes

def read_yolo_label(label_path, img_shape):
    h, w = img_shape[:2]
    logger.debug("Reading yolo label %s for image shape (h=%d,w=%d)", label_path, h, w)
    if not os.path.exists(label_path):
        logger.warning("Label file does not exist: %s", label_path)
        return []
    with open(label_path, "r") as f:
        boxes = _parse_yolo_lines(f, w, h, label_path)
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
    for sub in OCR_FOLDERS:
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

# "V/v", "V/V", "v/v", "V/ v", "V /v" ... at the start of the title. Only the abbreviation
# counts: "Về việc" spelled out also opens Quyết định / Thông báo titles.
_CONG_VAN_TITLE_RE = re.compile(r'^\s*[Vv]\s*/\s*[Vv]')

def infer_document_type(result_list: List[Dict]) -> List[Dict]:
    """
    Công văn prints no type name (Nghị định 30/2020), so YOLO has nothing to detect as
    documentType. If documentType is missing and documentTitle starts with "V/v", add
    {"class": "documentType", "text": "Công văn", "probs": []} right after documentTitle.
    Empty probs marks the value as inferred, not read from the image.
    Returns a new list; the input is not modified.
    """
    if any(item["class"] == "documentType" for item in result_list):
        return result_list
    title_idx = next((i for i, item in enumerate(result_list) if item["class"] == "documentTitle"), None)
    if title_idx is None or not _CONG_VAN_TITLE_RE.match(result_list[title_idx]["text"]):
        return result_list
    inferred = {"class": "documentType", "text": "Công văn", "probs": []}
    if "det_confs" in result_list[title_idx]:
        inferred["det_confs"] = []
    logger.debug("Inferred documentType='Công văn' from title: %s", result_list[title_idx]["text"])
    return result_list[:title_idx + 1] + [inferred] + result_list[title_idx + 1:]

def _group_text_by_class(
    roi_info: List[str],
    pred_results: List[Tuple[str, float]],
    det_confs: Optional[List[float]] = None,
) -> List[Dict]:
    """
    Join OCR chunks per class (in reading order) and apply per-class post-processing.
    If det_confs (YOLO confidence per ROI) is given, each item also gets "det_confs",
    parallel to "probs".
    """
    if det_confs is None:
        confs_iter = [None] * len(roi_info)
    else:
        confs_iter = det_confs
    class_text_dict = dict()
    for class_name, (text, prob), det_conf in zip(roi_info, pred_results, confs_iter):
        text = text.strip()
        logger.debug("OCR result class=%s text='%s' prob=%s det_conf=%s", class_name, text, prob, det_conf)
        if text:
            class_text_dict.setdefault(class_name, []).append({"text": text, "prob": prob, "det_conf": det_conf})
    final_result = []
    for class_name, text_chunks in class_text_dict.items():
        full_text = " ".join([chunk["text"] for chunk in text_chunks])
        probs = [chunk["prob"] for chunk in text_chunks]
        if class_name == "issuanceDate":
            full_text = extract_date_values_String(full_text)
        if class_name == "code":
            full_text = remove_before_first_space(full_text)
        item = {
            "class": class_name,
            "text": full_text,
            "probs": probs
        }
        if det_confs is not None:
            item["det_confs"] = [chunk["det_conf"] for chunk in text_chunks]
        final_result.append(item)
        logger.debug("Finalized OCR for class=%s text_len=%d probs=%s", class_name, len(full_text), probs)
    return final_result

def _ocr_page(
    page: Image.Image,
    boxes,
    detector,
    confs: Optional[List[float]] = None,
    skip_classes: Optional[set] = None,
) -> List[Dict]:
    """
    OCR the detected boxes of one in-memory page. confs (YOLO, parallel to boxes) -> "det_confs".
    Boxes whose class is in skip_classes are not OCR'd (rows are still grouped over all boxes,
    so the reading order of the remaining boxes is unchanged).
    """
    w, h = page.size
    items = _parse_yolo_lines(_yolo_label_lines(boxes, w, h), w, h)
    if not items:
        logger.debug("No boxes to OCR on this page")
        return []
    if confs is not None:
        if len(confs) == len(items):
            for item, conf in zip(items, confs):
                item["det_conf"] = conf
        else:
            logger.warning("Box/confidence count mismatch (%d vs %d), dropping det_confs", len(items), len(confs))
            confs = None
    if page.mode != "RGB":
        page = page.convert("RGB")
    pil_rois = []
    roi_info = []
    roi_confs = []
    for r_idx, row in enumerate(group_and_sort_bboxes_rowwise(items)):
        for b_idx, box in enumerate(row):
            if skip_classes and box["class_name"] in skip_classes:
                continue
            x1, y1, x2, y2 = box["xyxy"]
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2c <= x1c or y2c <= y1c:
                logger.debug("Skipping empty ROI row=%d box=%d coords=%s", r_idx, b_idx, box["xyxy"])
                continue
            pil_rois.append(page.crop((x1c, y1c, x2c, y2c)))
            roi_info.append(box["class_name"])
            roi_confs.append(box.get("det_conf"))
            logger.debug("Prepared ROI row=%d box=%d class=%s size=%s", r_idx, b_idx, box["class_name"], (x2c - x1c, y2c - y1c))
    pred_results = batch_ocr_predict(detector, pil_rois) if pil_rois else []
    return _group_text_by_class(roi_info, pred_results, roi_confs if confs is not None else None)

def ocr_and_group_text(image_path, label_path, detector):
    """OCR an image + YOLO label pair from disk (e.g. a dumped YOLO structure)."""
    logger.info("Running ocr_and_group_text on %s with labels %s", image_path, label_path)
    img = _imread_unicode(image_path)
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
            x1c, y1c, x2c, y2c = max(0, x1), max(0, y1), min(img.shape[1], x2), min(img.shape[0], y2)
            roi = img[y1c:y2c, x1c:x2c]
            if roi.size == 0:
                logger.debug("Skipping empty ROI row=%d box=%d coords=%s", r_idx, b_idx, box["xyxy"])
                continue
            pil_rois.append(Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)))
            roi_info.append(class_name)
            logger.debug("Prepared ROI row=%d box=%d class=%s shape=%s", r_idx, b_idx, class_name, roi.shape)
    pred_results = batch_ocr_predict(detector, pil_rois) if pil_rois else []
    return _group_text_by_class(roi_info, pred_results)

def pdf_to_json_ocr(pdf_path_or_file, model, detector, temp_yolo_folder='./yolo_data', remove_temp=True):
    """
    OCR a PDF page by page. The first occurrence of each class (in page order, Front/Last
    pages only) wins, so classes already found are not OCR'd again on later pages, and
    processing stops once every class is found. temp_yolo_folder is only touched when
    remove_temp=False, in which case the YOLO structure for every page is written there
    (the folder is cleared first).
    """
    logger.info("pdf_to_json_ocr start for %s", getattr(pdf_path_or_file, "name", str(pdf_path_or_file)))
    t_all = time.time()
    dump_root = None if remove_temp else temp_yolo_folder
    if dump_root:
        remove_folder_if_exists(dump_root)
        os.makedirs(dump_root, exist_ok=True)

    result_dict = {}
    idx = -1
    all_found = False
    with _open_pdf_pages(pdf_path_or_file) as (basename, chunks):
        for chunk_pages in chunks:
            detected_list, boxes_list, confs_list = _detect_images(model, chunk_pages)
            for page, detected_classes, boxes, confs in zip(chunk_pages, detected_list, boxes_list, confs_list):
                idx += 1
                folder_name = CLASS_FOLDER[classify_by_detection(detected_classes)]
                if dump_root:
                    _save_page_yolo(page, boxes, basename, idx, dump_root, folder_name)
                if folder_name not in OCR_FOLDERS:
                    logger.debug("Page %d classified as %s, skipping OCR", idx, folder_name)
                    continue
                try:
                    page_items = _ocr_page(
                        page, boxes, detector,
                        confs if OCR_INCLUDE_DET_CONF else None,
                        skip_classes=set(result_dict),
                    )
                    for item in page_items:
                        result_dict.setdefault(item["class"], dict(item))
                except Exception:
                    logger.exception("Error processing page %d of %s", idx, basename)
                # Later pages cannot change the result once every class is found.
                if not dump_root and all(c in result_dict for c in CLASS_NAMES):
                    logger.debug("All classes found at page %d, stopping early", idx)
                    all_found = True
                    break
            if all_found:
                break

    result_list = [result_dict[c] for c in CLASS_NAMES if c in result_dict]
    if OCR_INFER_DOC_TYPE:
        result_list = infer_document_type(result_list)
    logger.info("pdf_to_json_ocr completed in %.3fs, found %d classes", time.time() - t_all, len(result_list))
    return {basename: result_list}

def image_to_json_ocr(
    image_file_or_path: Union[str, Path, Any],
    model: Any,
    detector: Any,
    temp_yolo_folder_prefix: str = "yolo_tmp_",
    remove_temp: bool = True,
) -> Dict[str, List[Dict]]:
    """
    Convert an image (path or file-like) to OCR JSON results.

    Params:
    - image_file_or_path: path string/Path or file-like object with .read()
    - model: YOLO model
    - detector: VietOCR predictor
    - temp_yolo_folder_prefix: prefix of the temp folder the YOLO structure is dumped to
      when remove_temp=False
    - remove_temp: False keeps a YOLO structure dump (image + label) in a new temp folder

    Returns:
    - dict mapping basename -> list of ocr result dicts
    """
    name = getattr(image_file_or_path, "name", str(image_file_or_path))
    logger.info("image_to_json_ocr called for %s", name)

    try:
        if hasattr(image_file_or_path, "read"):
            image_bytes = image_file_or_path.read()
            image = Image.open(BytesIO(image_bytes))
            basename = Path(getattr(image_file_or_path, "name", "document")).stem
        else:
            image = Image.open(str(image_file_or_path))
            basename = Path(image_file_or_path).stem
    except UnidentifiedImageError as e:
        logger.exception("Cannot identify image file: %s", e)
        raise
    except Exception as e:
        logger.exception("Error opening image: %s", e)
        raise

    # normalize image mode to avoid format issues (e.g., RGBA -> JPEG)
    if image.mode not in ("RGB", "L"):
        image = image.convert("RGB")

    detected_classes, boxes, confs = _detect_page(model, image)
    folder_name = CLASS_FOLDER[classify_by_detection(detected_classes)]

    if not remove_temp:
        temp_dir = tempfile.mkdtemp(prefix=temp_yolo_folder_prefix)
        _save_page_yolo(image, boxes, basename, 0, temp_dir, folder_name)
        logger.info("Kept YOLO structure in %s", temp_dir)

    result_list = []
    if folder_name in OCR_FOLDERS:
        try:
            result_list.extend(_ocr_page(image, boxes, detector, confs if OCR_INCLUDE_DET_CONF else None))
        except Exception as e:
            logger.exception("OCR processing failed for %s : %s", name, e)
    else:
        logger.debug("Image classified as %s, skipping OCR", folder_name)
    if OCR_INFER_DOC_TYPE:
        result_list = infer_document_type(result_list)

    logger.info("image_to_json_ocr completed, found %d results for %s", len(result_list), basename)
    return {basename: result_list}

def file_to_json_ocr(file_obj_or_path, model, detector, temp_yolo_folder='./yolo_data', remove_temp=True):
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
