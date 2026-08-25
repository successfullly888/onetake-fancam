#!/usr/bin/env python3
"""Local web, person-detection, and identity-tracking service for 一键直拍.

The service binds to 127.0.0.1 only. Uploaded video is written to a temporary
file for analysis and deleted immediately after the response is built.
"""

import argparse
import base64
import cgi
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

try:
    from sam2_backend import status as sam2_runtime_status, track_video_object
except Exception as exc:  # optional high-precision runtime
    sam2_runtime_status = lambda: {"available": False, "engine": "SAM 2.1 Hiera Small", "error": str(exc)}
    track_video_object = None


ROOT = Path(__file__).resolve().parent
VENDOR = ROOT / "vendor"
MODELS = ROOT / "models"
OUTPUTS = ROOT / "outputs"
MAX_UPLOAD_BYTES = 1_500 * 1024 * 1024
BUILD_ID = "2026.08.25.5"

if VENDOR.exists():
    sys.path.insert(0, str(VENDOR))

try:
    import cv2  # type: ignore
    import numpy as np  # type: ignore

    CV_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - surfaced by health endpoint
    cv2 = None
    np = None
    CV_IMPORT_ERROR = str(exc)


class AnalysisError(RuntimeError):
    def __init__(self, code: str, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.hint = hint


def model_files_ready() -> bool:
    return all(
        path.exists() and path.stat().st_size >= minimum
        for path, minimum in (
            (MODELS / "yolov3-tiny.cfg", 1_000),
            (MODELS / "yolov3-tiny.weights", 30_000_000),
            (MODELS / "coco.names", 100),
        )
    )


def find_media_tool(name: str):
    executable = f"{name}.exe" if os.name == "nt" else name
    candidates = [
        ROOT / "tools" / "ffmpeg" / "bin" / executable,
        ROOT / "tools" / executable,
    ]
    configured = os.environ.get(f"ONETAKE_{name.upper()}")
    if configured:
        candidates.insert(0, Path(configured))
    discovered = shutil.which(name)
    if discovered:
        candidates.append(Path(discovered))
    if os.name == "nt":
        candidates.append(Path("E:/ffmpeg/bin") / executable)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate.resolve()
    return None


def render_status() -> dict:
    ffmpeg = find_media_tool("ffmpeg")
    ffprobe = find_media_tool("ffprobe")
    return {
        "ready": bool(ffmpeg and ffprobe),
        "renderer": "FFmpeg · H.264/AAC · local CPU",
        "error": None if ffmpeg and ffprobe else "ffmpeg or ffprobe is missing",
    }


def ai_status() -> dict:
    ready = CV_IMPORT_ERROR is None and model_files_ready()
    return {
        "build": BUILD_ID,
        "ready": ready,
        "detector": "YOLOv3-tiny + OpenCV DNN",
        "runtime": "local-cpu",
        "privacy": "video stays on this computer",
        "error": None if ready else CV_IMPORT_ERROR or "model files are missing",
        "render": render_status(),
        "high_precision": sam2_runtime_status(),
    }


class PersonDetector:
    def __init__(self) -> None:
        self._net = None
        self._lock = threading.Lock()

    def _load(self):
        if cv2 is None or np is None:
            raise AnalysisError(
                "dependency_missing",
                "本地人物检测依赖尚未安装。",
                "请双击“安装AI能力.cmd”，完成后重新启动一键直拍。",
            )
        if not model_files_ready():
            raise AnalysisError(
                "model_missing",
                "人物检测模型文件不完整。",
                "请双击“安装AI能力.cmd”，下载完成后重试。",
            )
        if self._net is None:
            # Passing byte buffers avoids OpenCV's Windows Unicode-path limitation.
            cfg = (MODELS / "yolov3-tiny.cfg").read_bytes()
            weights = (MODELS / "yolov3-tiny.weights").read_bytes()
            self._net = cv2.dnn.readNetFromDarknet(cfg, weights)
            self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return self._net

    def detect(self, frame, threshold: float = 0.12) -> list:
        net = self._load()
        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1 / 255.0, size=(416, 416), swapRB=True, crop=False
        )
        with self._lock:
            net.setInput(blob)
            outputs = net.forward(net.getUnconnectedOutLayersNames())

        raw_boxes = []
        confidences = []
        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = int(np.argmax(scores))
                if class_id != 0:
                    continue
                confidence = float(detection[4] * scores[class_id])
                if confidence < threshold:
                    continue
                center_x = float(detection[0] * width)
                center_y = float(detection[1] * height)
                box_width = float(detection[2] * width)
                box_height = float(detection[3] * height)
                x = center_x - box_width / 2
                y = center_y - box_height / 2
                raw_boxes.append([int(x), int(y), int(box_width), int(box_height)])
                confidences.append(confidence)

        if not raw_boxes:
            return []
        indices = cv2.dnn.NMSBoxes(raw_boxes, confidences, threshold, 0.38)
        if indices is None or len(indices) == 0:
            return []

        boxes = []
        for index in np.array(indices).reshape(-1).tolist():
            x, y, box_width, box_height = raw_boxes[index]
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(width - 1, x + box_width)
            y2 = min(height - 1, y + box_height)
            actual_width = x2 - x1
            actual_height = y2 - y1
            if actual_height < max(32, height * 0.075) or actual_width < 16:
                continue
            aspect = actual_height / max(actual_width, 1)
            if aspect < 0.72 or aspect > 6.0:
                continue
            margin_x = width * 0.012
            margin_y = height * 0.012
            cutoff = x1 <= margin_x or x2 >= width - margin_x or y1 <= margin_y or y2 >= height - margin_y
            boxes.append(
                {
                    "x1": int(x1),
                    "y1": int(y1),
                    "x2": int(x2),
                    "y2": int(y2),
                    "cx": round((x1 + x2) / 2, 2),
                    "cy": round((y1 + y2) / 2, 2),
                    "confidence": round(float(confidences[index]), 4),
                    "cutoff": bool(cutoff),
                }
            )

        boxes.sort(key=lambda item: item["cx"])
        for order, box in enumerate(boxes, 1):
            box["id"] = f"person-{order}"
            box["label"] = f"人物 {order}"
        return boxes[:20]


DETECTOR = PersonDetector()
TRACKING_CACHE = {}
TRACKING_CACHE_LOCK = threading.Lock()
TRACKING_CACHE_TTL_SECONDS = 45 * 60
TRACKING_CACHE_MAX_ENTRIES = 4


def get_tracking_cache(tracking_id: str):
    if not tracking_id:
        return None
    now = time.time()
    with TRACKING_CACHE_LOCK:
        expired = [
            key for key, value in TRACKING_CACHE.items()
            if now - float(value.get("last_access", 0.0)) > TRACKING_CACHE_TTL_SECONDS
        ]
        for key in expired:
            TRACKING_CACHE.pop(key, None)
        cached = TRACKING_CACHE.get(tracking_id)
        if cached is not None:
            cached["last_access"] = now
        return cached


def store_tracking_cache(tracking_id: str, payload: dict) -> None:
    now = time.time()
    with TRACKING_CACHE_LOCK:
        payload["last_access"] = now
        TRACKING_CACHE[tracking_id] = payload
        while len(TRACKING_CACHE) > TRACKING_CACHE_MAX_ENTRIES:
            oldest = min(TRACKING_CACHE, key=lambda key: TRACKING_CACHE[key].get("last_access", 0.0))
            TRACKING_CACHE.pop(oldest, None)


def box_iou(first: dict, second: dict) -> float:
    left = max(first["x1"], second["x1"])
    top = max(first["y1"], second["y1"])
    right = min(first["x2"], second["x2"])
    bottom = min(first["y2"], second["y2"])
    intersection = max(0, right - left) * max(0, bottom - top)
    if intersection <= 0:
        return 0.0
    first_area = (first["x2"] - first["x1"]) * (first["y2"] - first["y1"])
    second_area = (second["x2"] - second["x1"]) * (second["y2"] - second["y1"])
    return intersection / max(1, first_area + second_area - intersection)


def frame_measurements(frame, boxes: list) -> dict:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    brightness_score = max(0.0, 1.0 - abs(132.0 - brightness) / 132.0)
    overlaps = [
        box_iou(boxes[first], boxes[second])
        for first in range(len(boxes))
        for second in range(first + 1, len(boxes))
    ]
    overlap_penalty = sum(overlaps) / len(overlaps) if overlaps else 0.0
    average_confidence = sum(box["confidence"] for box in boxes) / len(boxes) if boxes else 0.0
    cutoff_ratio = sum(1 for box in boxes if box["cutoff"]) / len(boxes) if boxes else 1.0
    return {
        "sharpness_raw": sharpness,
        "brightness": brightness,
        "brightness_score": brightness_score,
        "overlap_penalty": overlap_penalty,
        "average_confidence": average_confidence,
        "cutoff_ratio": cutoff_ratio,
    }


def encode_jpeg(frame) -> str:
    parameters = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
    ok, buffer = cv2.imencode(".jpg", frame, parameters)
    if not ok:
        raise AnalysisError("encode_failed", "代表帧编码失败。", "请更换视频格式后重试。")
    encoded = base64.b64encode(buffer.tobytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def sample_times(start: float, end: float, count: int = 9) -> list:
    span = max(0.0, end - start)
    if span <= 0.1:
        return [start]
    inset = min(0.25, span * 0.06)
    return [float(value) for value in np.linspace(start + inset, end - inset, count)]


def clamp_box(box: dict, width: int, height: int) -> dict:
    x1 = max(0.0, min(float(box["x1"]), width - 2.0))
    y1 = max(0.0, min(float(box["y1"]), height - 2.0))
    x2 = max(x1 + 1.0, min(float(box["x2"]), width - 1.0))
    y2 = max(y1 + 1.0, min(float(box["y2"]), height - 1.0))
    return {
        "x1": round(x1, 2),
        "y1": round(y1, 2),
        "x2": round(x2, 2),
        "y2": round(y2, 2),
        "cx": round((x1 + x2) / 2.0, 2),
        "cy": round((y1 + y2) / 2.0, 2),
    }


def performer_role_score(box: dict, width: int, height: int) -> float:
    """Estimate whether a detection belongs to the active stage instead of the sidelines.

    This is deliberately a scene prior, not an identity signal. Fancam source videos
    usually keep the performers in the middle stage band while spectators, teachers,
    and foreground heads collect near the edges or become much larger than the cast.
    Every box remains selectable; the score only ranks and styles likely performers.
    """
    safe = clamp_box(box, width, height)
    normalized_x = safe["cx"] / max(1.0, width)
    normalized_foot = safe["y2"] / max(1.0, height)
    height_ratio = (safe["y2"] - safe["y1"]) / max(1.0, height)
    center = max(0.0, 1.0 - abs(normalized_x - 0.5) / 0.48)
    stage_depth = max(0.0, 1.0 - abs(normalized_foot - 0.68) / 0.34)
    if height_ratio <= 0.04:
        scale = 0.0
    elif height_ratio <= 0.22:
        scale = min(1.0, height_ratio / 0.10)
    else:
        scale = max(0.0, 1.0 - (height_ratio - 0.22) / 0.30)
    full_body = 0.35 if bool(box.get("cutoff", False)) else 1.0
    score = 0.42 * center + 0.24 * stage_depth + 0.20 * scale + 0.14 * full_body
    if normalized_x < 0.08 or normalized_x > 0.92:
        score *= 0.58
    if height_ratio > 0.42:
        score *= 0.46
    return round(max(0.0, min(1.0, score)), 4)


def decorate_performer_roles(boxes: list, width: int, height: int) -> list:
    decorated = []
    for box in boxes:
        item = dict(box)
        score = performer_role_score(item, width, height)
        item["performer_score"] = score
        item["role"] = "performer" if score >= 0.70 else "context"
        item["selection_box"] = identity_region(item, width, height)
        decorated.append(item)
    return decorated


def merge_person_fragments(boxes: list, width: int, height: int) -> list:
    """Merge detector boxes that split one distant person into upper/lower fragments.

    This is used for the clear selection frame only. Tracking keeps the raw detections
    so two genuinely crossing dancers are never irreversibly fused.
    """
    merged = [dict(box) for box in boxes]
    changed = True
    while changed:
        changed = False
        for first_index in range(len(merged)):
            if changed:
                break
            first = merged[first_index]
            first_width = max(1.0, first["x2"] - first["x1"])
            first_height = max(1.0, first["y2"] - first["y1"])
            for second_index in range(first_index + 1, len(merged)):
                second = merged[second_index]
                second_width = max(1.0, second["x2"] - second["x1"])
                second_height = max(1.0, second["y2"] - second["y1"])
                horizontal_overlap = max(0.0, min(first["x2"], second["x2"]) - max(first["x1"], second["x1"]))
                overlap_ratio = horizontal_overlap / max(1.0, min(first_width, second_width))
                horizontal_close = abs(first["cx"] - second["cx"]) <= max(18.0, max(first_width, second_width) * 0.46)
                vertical_gap = max(0.0, max(first["y1"], second["y1"]) - min(first["y2"], second["y2"]))
                compatible_scale = max(first_height, second_height) / min(first_height, second_height) <= 2.6
                same_fragment = (
                    overlap_ratio >= 0.56 and horizontal_close and compatible_scale
                    and vertical_gap <= max(12.0, height * 0.035)
                )
                if not same_fragment:
                    continue
                union = {
                    "x1": int(min(first["x1"], second["x1"])),
                    "y1": int(min(first["y1"], second["y1"])),
                    "x2": int(max(first["x2"], second["x2"])),
                    "y2": int(max(first["y2"], second["y2"])),
                    "confidence": round(max(float(first.get("confidence", 0.0)), float(second.get("confidence", 0.0))), 4),
                    "cutoff": bool(first.get("cutoff", False) or second.get("cutoff", False)),
                }
                union["cx"] = round((union["x1"] + union["x2"]) / 2.0, 2)
                union["cy"] = round((union["y1"] + union["y2"]) / 2.0, 2)
                merged[first_index] = union
                merged.pop(second_index)
                changed = True
                break
    merged.sort(key=lambda item: item["cx"])
    for order, box in enumerate(merged, 1):
        box["id"] = f"person-{order}"
        box["label"] = f"人物 {order}"
    return merged


def correction_candidate_boxes(frame, cached_detections: list) -> list:
    """Build a high-recall person list only for frames shown to the user.

    The tracking pass deliberately runs one inexpensive full-frame detection per
    sample.  Distant dancers can become only a few pixels wide at the detector's
    416px input size, especially with raised arms or partial overlap.  Correction
    is a much smaller workload (at most twelve still frames), so we can afford
    overlapping stage tiles at a lower threshold without slowing the whole clip.
    Cached full-frame boxes remain in the list, including people at the sidelines.
    """
    height, width = frame.shape[:2]
    proposals = []
    for detection in cached_detections:
        item = dict(detection.get("box") or {})
        if not item:
            continue
        item["confidence"] = round(float(detection.get("detector_confidence", 0.0)), 4)
        item["cutoff"] = bool(
            item["x1"] <= width * 0.012
            or item["x2"] >= width * 0.988
            or item["y1"] <= height * 0.012
            or item["y2"] >= height * 0.988
        )
        proposals.append(item)

    # Three overlapping crops cover the usual central dance floor while making
    # small performers roughly 2x larger to YOLO.  They are a correction aid,
    # never an automatic identity decision.
    y1 = int(round(height * 0.25))
    y2 = int(round(height * 0.90))
    tile_ranges = ((0.10, 0.52), (0.34, 0.76), (0.58, 0.98))
    for left_ratio, right_ratio in tile_ranges:
        x1 = int(round(width * left_ratio))
        x2 = int(round(width * right_ratio))
        tile = frame[y1:y2, x1:x2]
        if tile.size == 0:
            continue
        for detected in DETECTOR.detect(tile, threshold=0.045):
            mapped = {
                "x1": int(detected["x1"] + x1),
                "y1": int(detected["y1"] + y1),
                "x2": int(detected["x2"] + x1),
                "y2": int(detected["y2"] + y1),
                "confidence": round(float(detected.get("confidence", 0.0)), 4),
                "cutoff": False,
            }
            mapped["cutoff"] = bool(
                mapped["x1"] <= width * 0.012
                or mapped["x2"] >= width * 0.988
                or mapped["y1"] <= height * 0.012
                or mapped["y2"] >= height * 0.988
            )
            mapped["cx"] = round((mapped["x1"] + mapped["x2"]) / 2.0, 2)
            mapped["cy"] = round((mapped["y1"] + mapped["y2"]) / 2.0, 2)
            duplicate = next((box for box in proposals if box_iou(box, mapped) >= 0.46), None)
            if duplicate is None:
                proposals.append(mapped)
            elif mapped["confidence"] > float(duplicate.get("confidence", 0.0)):
                duplicate.update(mapped)

    proposals = merge_person_fragments(proposals, width, height)
    proposals = decorate_performer_roles(proposals, width, height)
    return proposals[:28]


def identity_region(box: dict, width: int, height: int) -> dict:
    """Expand detector fragments into a more useful clothing/whole-body identity crop."""
    safe = clamp_box(box, width, height)
    box_width = max(2.0, safe["x2"] - safe["x1"])
    box_height = max(2.0, safe["y2"] - safe["y1"])
    aspect = box_height / box_width
    if aspect < 1.6:
        top, bottom, side = 1.35, 0.24, 0.34
    elif aspect < 2.25:
        top, bottom, side = 0.72, 0.18, 0.18
    else:
        top, bottom, side = 0.28, 0.12, 0.10
    expanded = {
        "x1": safe["x1"] - box_height * side,
        "y1": safe["y1"] - box_height * top,
        "x2": safe["x2"] + box_height * side,
        "y2": safe["y2"] + box_height * bottom,
    }
    return clamp_box(expanded, width, height)


def shift_box(box: dict, center_x: float, center_y: float, width: int, height: int) -> dict:
    box_width = max(2.0, box["x2"] - box["x1"])
    box_height = max(2.0, box["y2"] - box["y1"])
    x1 = center_x - box_width / 2.0
    y1 = center_y - box_height / 2.0
    if x1 < 0:
        x1 = 0.0
    if y1 < 0:
        y1 = 0.0
    if x1 + box_width > width - 1:
        x1 = max(0.0, width - 1.0 - box_width)
    if y1 + box_height > height - 1:
        y1 = max(0.0, height - 1.0 - box_height)
    return clamp_box(
        {"x1": x1, "y1": y1, "x2": x1 + box_width, "y2": y1 + box_height},
        width,
        height,
    )


def appearance_histogram(frame, box: dict):
    """Describe mostly clothing pixels while reducing background and head influence."""
    safe = identity_region(box, frame.shape[1], frame.shape[0])
    box_width = safe["x2"] - safe["x1"]
    box_height = safe["y2"] - safe["y1"]
    x1 = int(safe["x1"] + box_width * 0.14)
    x2 = int(safe["x2"] - box_width * 0.14)
    y1 = int(safe["y1"] + box_height * 0.12)
    y2 = int(safe["y1"] + box_height * 0.78)
    roi = frame[max(0, y1):max(y1 + 1, y2), max(0, x1):max(x1 + 1, x2)]
    if roi.size == 0:
        return np.zeros((24 * 16,), dtype=np.float32)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
    return cv2.normalize(histogram, histogram).flatten().astype(np.float32)


def appearance_signature(frame, box: dict):
    """Keep upper/lower clothing separate so one similar colour cannot steal the track."""
    safe = identity_region(box, frame.shape[1], frame.shape[0])
    box_width = safe["x2"] - safe["x1"]
    x1 = max(0, int(safe["x1"] + box_width * 0.12))
    x2 = min(frame.shape[1], int(safe["x2"] - box_width * 0.12))
    y1 = max(0, int(safe["y1"]))
    y2 = min(frame.shape[0], int(safe["y2"]))
    roi = frame[y1:max(y1 + 1, y2), x1:max(x1 + 1, x2)]
    if roi.size == 0:
        return [np.zeros((18 * 12,), np.float32), np.zeros((24,), np.float32)] * 2 + [
            np.zeros((12 * 24 * 3,), np.float32)
        ]
    roi = cv2.resize(roi, (48, 96), interpolation=cv2.INTER_CUBIC)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    features = []
    for start, end in ((8, 48), (42, 94)):
        part = hsv[start:end]
        hue_saturation = cv2.calcHist([part], [0, 1], None, [18, 12], [0, 180, 0, 256])
        value = cv2.calcHist([part], [2], None, [24], [0, 256])
        features.extend([
            cv2.normalize(hue_saturation, hue_saturation).flatten().astype(np.float32),
            cv2.normalize(value, value).flatten().astype(np.float32),
        ])
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
    thumbnail = cv2.resize(lab, (12, 24), interpolation=cv2.INTER_AREA).astype(np.float32)
    thumbnail = (thumbnail - thumbnail.mean(axis=(0, 1), keepdims=True)) / (
        thumbnail.std(axis=(0, 1), keepdims=True) + 8.0
    )
    features.append((thumbnail.flatten() / np.sqrt(thumbnail.size)).astype(np.float32))
    return features


def appearance_signature_distance(left, right) -> float:
    histogram_distances = [
        float(cv2.compareHist(left[index], right[index], cv2.HISTCMP_BHATTACHARYYA))
        for index in range(4)
    ]
    denominator = max(float(np.linalg.norm(left[4]) * np.linalg.norm(right[4])), 1e-9)
    thumbnail_distance = min(1.5, 1.0 - float(np.dot(left[4], right[4]) / denominator))
    return float(
        0.24 * histogram_distances[0]
        + 0.18 * histogram_distances[1]
        + 0.27 * histogram_distances[2]
        + 0.21 * histogram_distances[3]
        + 0.10 * thumbnail_distance
    )


def blend_signatures(current, observed, observed_weight: float = 0.08):
    blended = []
    for left, right in zip(current, observed):
        value = left * (1.0 - observed_weight) + right * observed_weight
        norm = float(np.linalg.norm(value))
        blended.append((value / max(norm, 1e-9)).flatten().astype(np.float32))
    return blended


def tracking_sample_times(start: float, end: float, selection_time: float, sample_fps: float) -> list:
    span = max(0.0, end - start)
    target_count = max(2, int(math.ceil(span * sample_fps)) + 1)
    target_count = min(target_count, 360)
    values = [float(value) for value in np.linspace(start, end, target_count)]
    values.append(float(max(start, min(selection_time, end))))
    return sorted({round(value, 4) for value in values})


def enrich_detections(frame, boxes: list) -> list:
    enriched = []
    for box in boxes:
        performer_score = float(box.get(
            "performer_score",
            performer_role_score(box, frame.shape[1], frame.shape[0]),
        ))
        enriched.append({
            "box": clamp_box(box, frame.shape[1], frame.shape[0]),
            "histogram": appearance_histogram(frame, box),
            "signature": appearance_signature(frame, box),
            "detector_confidence": float(box.get("confidence", 0.0)),
            "performer_score": performer_score,
        })
    return enriched


def identity_metrics(candidate: dict, seed_histogram, seed_signature) -> tuple:
    histogram_distance = float(cv2.compareHist(
        seed_histogram,
        candidate["histogram"],
        cv2.HISTCMP_BHATTACHARYYA,
    ))
    signature_distance = appearance_signature_distance(seed_signature, candidate["signature"])
    return histogram_distance, signature_distance


def global_track_direction(records: list, indices, anchor_index: int, seed_box: dict,
                           seed_histogram, seed_signature, width: int, height: int) -> dict:
    """Keep multiple whole-video identity paths instead of committing frame by frame.

    A greedy tracker lets one bad association poison its appearance template and motion
    state. This small Viterbi/beam search retains competing paths, scores every person
    against the immutable user anchor, adds a main-stage prior, and only decides after
    seeing the full direction of the clip.
    """
    ordered_indices = list(indices)
    if not ordered_indices:
        return {}
    diagonal = math.hypot(width, height)
    anchor_time = records[anchor_index]["time"]
    anchor_performer_score = performer_role_score(seed_box, width, height)
    anchor_foot_ratio = float(seed_box["y2"]) / max(1.0, height)
    anchor_height_ratio = float(seed_box["y2"] - seed_box["y1"]) / max(1.0, height)
    hypotheses = [{
        "box": dict(seed_box),
        "time": anchor_time,
        "velocity_x": 0.0,
        "velocity_y": 0.0,
        "missing_count": 0,
        "cost": 0.0,
        "path": [],
    }]

    for index in ordered_indices:
        record = records[index]
        timestamp = record["time"]
        candidates = record["detections"]
        best_for_candidate = {}
        missing_expansions = []
        for hypothesis in hypotheses:
            delta = timestamp - hypothesis["time"]
            previous = hypothesis["box"]
            predicted = shift_box(
                previous,
                previous["cx"] + hypothesis["velocity_x"] * delta,
                previous["cy"] + hypothesis["velocity_y"] * delta,
                width,
                height,
            )
            previous_height = max(1.0, previous["y2"] - previous["y1"])
            for candidate_index, candidate in enumerate(candidates):
                box = candidate["box"]
                histogram_distance, signature_distance = identity_metrics(
                    candidate, seed_histogram, seed_signature
                )
                pixel_distance = math.hypot(box["cx"] - predicted["cx"], box["cy"] - predicted["cy"])
                motion_scale = max(
                    diagonal * (0.035 + min(0.80, abs(delta)) * 0.075),
                    previous_height * 1.55,
                )
                motion_distance = min(2.0, pixel_distance / max(1.0, motion_scale))
                candidate_height = max(1.0, box["y2"] - box["y1"])
                size_distance = min(1.5, abs(math.log(candidate_height / previous_height)))
                performer_penalty = 1.0 - float(candidate.get("performer_score", 0.5))
                detector_penalty = 1.0 - float(candidate.get("detector_confidence", 0.0))
                candidate_foot_ratio = float(box["y2"]) / max(1.0, height)
                candidate_height_ratio = candidate_height / max(1.0, height)
                stage_depth_distance = min(
                    1.5,
                    abs(candidate_foot_ratio - anchor_foot_ratio) / 0.18,
                )
                local_cost = (
                    0.46 * signature_distance
                    + 0.10 * histogram_distance
                    + 0.22 * motion_distance
                    + 0.05 * size_distance
                    + 0.09 * performer_penalty
                    + 0.03 * detector_penalty
                    + 0.05 * stage_depth_distance
                )
                # A user-selected stage performer should not be stolen by a
                # spectator/teacher standing much closer to the camera.  This is
                # scene-depth gating, not left/right rank identity.
                if anchor_performer_score >= 0.70:
                    candidate_performer_score = float(candidate.get("performer_score", 0.0))
                    if candidate_performer_score < 0.58:
                        local_cost += 0.16 + (0.58 - candidate_performer_score) * 0.70
                    if (
                        candidate_foot_ratio >= max(0.91, anchor_foot_ratio + 0.16)
                        and candidate_height_ratio >= max(0.27, anchor_height_ratio * 1.35)
                    ):
                        local_cost += 0.62
                allowed_jump = max(diagonal * 0.17, previous_height * 3.4)
                if pixel_distance > allowed_jump:
                    local_cost += 0.55 + min(0.45, (pixel_distance - allowed_jump) / diagonal)
                total_cost = hypothesis["cost"] + local_cost
                observed_velocity_x = (box["cx"] - previous["cx"]) / delta if abs(delta) > 1e-6 else 0.0
                observed_velocity_y = (box["cy"] - previous["cy"]) / delta if abs(delta) > 1e-6 else 0.0
                confidence = max(0.0, min(1.0, 1.0 - local_cost / 0.88))
                confidence = 0.70 * confidence + 0.18 * float(candidate.get("detector_confidence", 0.0)) + 0.12 * float(candidate.get("performer_score", 0.5))
                overlap = any(
                    other_index != candidate_index and box_iou(box, other["box"]) > 0.12
                    for other_index, other in enumerate(candidates)
                )
                path_item = {
                    "index": index,
                    "box": box,
                    "confidence": round(float(max(0.0, min(1.0, confidence))), 4),
                    "detector_confidence": round(float(candidate.get("detector_confidence", 0.0)), 4),
                    "status": "tracked" if confidence >= 0.50 else "low_confidence",
                    "ambiguous": False,
                    "overlap": bool(overlap),
                    "appearance_distance": round(histogram_distance, 4),
                    "identity_distance": round(signature_distance, 4),
                    "motion_distance": round(motion_distance, 4),
                    "performer_score": round(float(candidate.get("performer_score", 0.0)), 4),
                }
                expanded = {
                    "box": box,
                    "time": timestamp,
                    "velocity_x": hypothesis["velocity_x"] * 0.58 + observed_velocity_x * 0.42,
                    "velocity_y": hypothesis["velocity_y"] * 0.58 + observed_velocity_y * 0.42,
                    "missing_count": 0,
                    "cost": total_cost,
                    "path": hypothesis["path"] + [path_item],
                }
                previous_best = best_for_candidate.get(candidate_index)
                if previous_best is None or expanded["cost"] < previous_best["cost"]:
                    best_for_candidate[candidate_index] = expanded

            missing_count = hypothesis["missing_count"] + 1
            missing_cost = hypothesis["cost"] + 0.50 + 0.12 * min(4, missing_count)
            missing_item = {
                "index": index,
                "box": predicted,
                "confidence": 0.0,
                "detector_confidence": 0.0,
                "status": "missing",
                "ambiguous": False,
                "overlap": False,
                "appearance_distance": 1.0,
                "identity_distance": 1.0,
                "motion_distance": 1.0,
                "performer_score": 0.0,
            }
            missing_expansions.append({
                "box": predicted,
                "time": timestamp,
                "velocity_x": hypothesis["velocity_x"] * 0.90,
                "velocity_y": hypothesis["velocity_y"] * 0.90,
                "missing_count": missing_count,
                "cost": missing_cost,
                "path": hypothesis["path"] + [missing_item],
            })

        expanded_hypotheses = list(best_for_candidate.values())
        if missing_expansions:
            expanded_hypotheses.append(min(missing_expansions, key=lambda item: item["cost"]))
        expanded_hypotheses.sort(key=lambda item: item["cost"])
        hypotheses = expanded_hypotheses[:18]
        if not hypotheses:
            break

    if not hypotheses:
        return {}
    best_path = min(hypotheses, key=lambda item: item["cost"])["path"]
    return {item.pop("index"): item for item in best_path}


def anchor_path_item(anchor: dict, width: int, height: int) -> dict:
    box = anchor["box"]
    return {
        "box": box,
        "confidence": 1.0,
        "detector_confidence": round(float(anchor.get("detector_confidence", 1.0)), 4),
        "status": "anchor",
        "ambiguous": False,
        "overlap": False,
        "appearance_distance": 0.0,
        "identity_distance": 0.0,
        "motion_distance": 0.0,
        "performer_score": performer_role_score(box, width, height),
    }


def build_multi_anchor_path(records: list, anchors: list, width: int, height: int) -> dict:
    """Track between confirmed anchors, then join the locally constrained paths.

    A correction is not treated as a replacement for the original identity.  All
    user-confirmed points remain hard constraints.  Each interval is solved from
    both ends and the lower-cost temporal side is selected, so one bad area cannot
    overwrite the already-confirmed parts of the clip.
    """
    if not anchors:
        return {}
    by_index = {}
    for anchor in anchors:
        # Later entries are manual corrections and intentionally win if two
        # timestamps map to the same sampled frame.
        by_index[int(anchor["index"])] = anchor
    ordered = [by_index[index] for index in sorted(by_index)]
    selected = {
        int(anchor["index"]): anchor_path_item(anchor, width, height)
        for anchor in ordered
    }

    first = ordered[0]
    first_index = int(first["index"])
    selected.update(global_track_direction(
        records,
        range(first_index - 1, -1, -1),
        first_index,
        first["box"],
        first["histogram"],
        first["signature"],
        width,
        height,
    ))

    for left, right in zip(ordered, ordered[1:]):
        left_index = int(left["index"])
        right_index = int(right["index"])
        if right_index <= left_index + 1:
            continue
        forward = global_track_direction(
            records,
            range(left_index + 1, right_index),
            left_index,
            left["box"],
            left["histogram"],
            left["signature"],
            width,
            height,
        )
        backward = global_track_direction(
            records,
            range(right_index - 1, left_index, -1),
            right_index,
            right["box"],
            right["histogram"],
            right["signature"],
            width,
            height,
        )
        interval = max(1, right_index - left_index)
        for index in range(left_index + 1, right_index):
            left_item = forward.get(index)
            right_item = backward.get(index)
            if left_item is None:
                selected[index] = right_item
                continue
            if right_item is None:
                selected[index] = left_item
                continue
            ratio = (index - left_index) / interval
            left_penalty = ratio + (1.0 - float(left_item.get("confidence", 0.0))) * 0.30
            right_penalty = (1.0 - ratio) + (1.0 - float(right_item.get("confidence", 0.0))) * 0.30
            if left_item.get("status") == "missing":
                left_penalty += 0.50
            if right_item.get("status") == "missing":
                right_penalty += 0.50
            selected[index] = left_item if left_penalty <= right_penalty else right_item

    last = ordered[-1]
    last_index = int(last["index"])
    selected.update(global_track_direction(
        records,
        range(last_index + 1, len(records)),
        last_index,
        last["box"],
        last["histogram"],
        last["signature"],
        width,
        height,
    ))
    # Re-apply anchors after path joins so they can never be replaced by an
    # interval result.
    for anchor in ordered:
        selected[int(anchor["index"])] = anchor_path_item(anchor, width, height)
    return selected


def augment_risk_detections(video_path: Path, records: list, selected: dict,
                            width: int, height: int, anchor_indices: set,
                            max_frames: int = 32) -> dict:
    """Re-detect only identity-risk frames at a larger effective scale.

    Full-clip tiled detection would roughly quadruple CPU time.  The first pass
    already tells us where identity evidence is weak, so only those frames receive
    the more expensive stage-tile detector.  Added candidates are observations;
    the identity path still decides among them using appearance and continuity.
    """
    diagonal = math.hypot(width, height)
    risks = []
    for index, item in selected.items():
        if index in anchor_indices:
            continue
        confidence = float(item.get("confidence", 0.0))
        identity_distance = float(item.get("identity_distance", 1.0))
        risk = max(0.0, 0.58 - confidence) * 1.8 + max(0.0, identity_distance - 0.58)
        if item.get("status") == "missing":
            risk += 1.0
        if item.get("overlap"):
            risk += 0.28
        if item.get("ambiguous"):
            risk += 0.22
        if index > 0 and index - 1 in selected:
            previous = selected[index - 1]["box"]
            current = item["box"]
            jump = math.hypot(current["cx"] - previous["cx"], current["cy"] - previous["cy"])
            if jump > diagonal * 0.075:
                risk += min(0.8, jump / diagonal * 2.5)
        if risk > 0.08:
            risks.append((risk, index))
    risks.sort(reverse=True)
    chosen = sorted(index for _, index in risks[:max_frames])
    if not chosen:
        return {"frames": 0, "candidates": 0}

    capture = cv2.VideoCapture(str(video_path))
    augmented_frames = 0
    added_candidates = 0
    for index in chosen:
        record = records[index]
        capture.set(cv2.CAP_PROP_POS_MSEC, float(record["time"]) * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        enhanced_boxes = correction_candidate_boxes(frame, record.get("detections", []))
        additions = []
        for box in enhanced_boxes:
            box_width = float(box["x2"] - box["x1"])
            box_height = float(box["y2"] - box["y1"])
            if not (
                height * 0.075 <= box_height <= height * 0.44
                and width * 0.014 <= box_width <= width * 0.22
            ):
                continue
            if any(box_iou(box, existing["box"]) >= 0.52 for existing in record["detections"]):
                continue
            additions.append(box)
        if additions:
            record["detections"].extend(enrich_detections(frame, additions[:12]))
            augmented_frames += 1
            added_candidates += min(12, len(additions))
    capture.release()
    return {"frames": augmented_frames, "candidates": added_candidates}


def match_identity(candidates: list, previous: dict, predicted: dict, seed_histogram, adaptive_histogram,
                   seed_signature, adaptive_signature, frame_diagonal: float) -> tuple:
    scored = []
    previous_height = max(1.0, previous["y2"] - previous["y1"])
    for candidate in candidates:
        box = candidate["box"]
        seed_distance = float(cv2.compareHist(
            seed_histogram, candidate["histogram"], cv2.HISTCMP_BHATTACHARYYA
        ))
        adaptive_distance = float(cv2.compareHist(
            adaptive_histogram, candidate["histogram"], cv2.HISTCMP_BHATTACHARYYA
        ))
        signature_distance = appearance_signature_distance(seed_signature, candidate["signature"])
        adaptive_signature_distance = appearance_signature_distance(
            adaptive_signature, candidate["signature"]
        )
        pixel_distance = math.hypot(box["cx"] - predicted["cx"], box["cy"] - predicted["cy"])
        motion_scale = max(frame_diagonal * 0.16, previous_height * 2.6)
        motion_distance = min(1.5, pixel_distance / max(1.0, motion_scale))
        candidate_height = max(1.0, box["y2"] - box["y1"])
        size_distance = min(1.0, abs(math.log(candidate_height / previous_height)))
        overlap = box_iou(previous, box)
        cost = (
            0.54 * signature_distance
            + 0.10 * adaptive_signature_distance
            + 0.08 * seed_distance
            + 0.18 * motion_distance
            + 0.06 * size_distance
            + 0.04 * (1.0 - overlap)
        )
        allowed_jump = max(frame_diagonal * 0.34, previous_height * 4.2)
        if pixel_distance > allowed_jump and seed_distance > 0.24:
            cost += 0.45
        scored.append({
            "candidate": candidate,
            "cost": cost,
            "seed_distance": seed_distance,
            "signature_distance": signature_distance,
            "motion_distance": motion_distance,
        })

    scored.sort(key=lambda item: item["cost"])
    if not scored:
        return None, {"cost": 1.0, "ambiguous": False, "overlap": False}
    identity_ranked = sorted(scored, key=lambda item: item["signature_distance"])
    identity_best = identity_ranked[0]
    if (
        identity_best is not scored[0]
        and identity_best["signature_distance"] <= 0.64
        and scored[0]["signature_distance"] - identity_best["signature_distance"] >= 0.11
    ):
        identity_best["cost"] = max(0.0, identity_best["cost"] - 0.13)
        scored.sort(key=lambda item: item["cost"])
    if scored[0]["cost"] > 0.76 or scored[0]["signature_distance"] > 0.80:
        return None, {"cost": 1.0, "ambiguous": False, "overlap": False}

    best = scored[0]
    second_gap = scored[1]["cost"] - best["cost"] if len(scored) > 1 else 1.0
    identity_gap = (
        identity_ranked[1]["signature_distance"] - identity_ranked[0]["signature_distance"]
        if len(identity_ranked) > 1 else 1.0
    )
    selected_box = best["candidate"]["box"]
    overlap = any(
        box_iou(selected_box, item["candidate"]["box"]) > 0.12
        for item in scored[1:]
    )
    return best["candidate"], {
        "cost": float(best["cost"]),
        "seed_distance": float(best["seed_distance"]),
        "signature_distance": float(best["signature_distance"]),
        "motion_distance": float(best["motion_distance"]),
        "ambiguous": second_gap < 0.06 or identity_gap < 0.035,
        "overlap": overlap,
    }


def track_direction(records: list, indices, seed_box: dict, seed_histogram, seed_signature,
                    width: int, height: int) -> dict:
    selected = {}
    previous = dict(seed_box)
    previous_time = None
    velocity_x = 0.0
    velocity_y = 0.0
    adaptive_histogram = seed_histogram.copy()
    adaptive_signature = [value.copy() for value in seed_signature]
    diagonal = math.hypot(width, height)

    for index in indices:
        record = records[index]
        timestamp = record["time"]
        delta = 0.0 if previous_time is None else timestamp - previous_time
        predicted = shift_box(
            previous,
            previous["cx"] + velocity_x * delta,
            previous["cy"] + velocity_y * delta,
            width,
            height,
        )
        candidate, details = match_identity(
            record["detections"], previous, predicted, seed_histogram, adaptive_histogram,
            seed_signature, adaptive_signature, diagonal
        )
        if candidate is None:
            current = predicted
            status = "missing"
            confidence = 0.0
            detector_confidence = 0.0
        else:
            current = candidate["box"]
            raw_match = max(0.0, 1.0 - details["cost"] / 0.78)
            detector_confidence = candidate["detector_confidence"]
            signature_quality = max(0.0, 1.0 - details["signature_distance"] / 0.80)
            confidence = 0.58 * raw_match + 0.24 * detector_confidence + 0.18 * signature_quality
            if details["ambiguous"]:
                confidence *= 0.70
            if details["overlap"]:
                confidence *= 0.76
            trusted = (
                details["signature_distance"] <= 0.66
                and (confidence >= 0.42 or details["signature_distance"] <= 0.58)
            )
            if not trusted:
                current = predicted
            status = "tracked" if confidence >= 0.52 and trusted else "low_confidence"
            if (
                confidence >= 0.68 and details["signature_distance"] <= 0.62
                and not details["overlap"] and not details["ambiguous"]
            ):
                adaptive_histogram = cv2.normalize(
                    adaptive_histogram * 0.88 + candidate["histogram"] * 0.12,
                    None,
                ).flatten().astype(np.float32)
                adaptive_signature = blend_signatures(adaptive_signature, candidate["signature"])

        if previous_time is not None and abs(delta) > 1e-6 and status != "missing":
            observed_velocity_x = (current["cx"] - previous["cx"]) / delta
            observed_velocity_y = (current["cy"] - previous["cy"]) / delta
            velocity_x = velocity_x * 0.55 + observed_velocity_x * 0.45
            velocity_y = velocity_y * 0.55 + observed_velocity_y * 0.45
        selected[index] = {
            "box": current,
            "confidence": round(float(confidence), 4),
            "detector_confidence": round(float(detector_confidence), 4),
            "status": status,
            "ambiguous": bool(details.get("ambiguous", False)),
            "overlap": bool(details.get("overlap", False)),
            "appearance_distance": round(float(details.get("seed_distance", 1.0)), 4),
            "identity_distance": round(float(details.get("signature_distance", 1.0)), 4),
            "motion_distance": round(float(details.get("motion_distance", 1.0)), 4),
        }
        previous = current
        previous_time = timestamp
    return selected


def confidence_ranges(keyframes: list, sample_step: float) -> list:
    flagged = [
        frame for frame in keyframes
        if frame["status"] == "missing" or frame["confidence"] < 0.48
        or frame["overlap"] or frame["ambiguous"]
    ]
    if not flagged:
        return []
    ranges = []
    current = [flagged[0]]
    for frame in flagged[1:]:
        if frame["time"] - current[-1]["time"] <= sample_step * 1.65:
            current.append(frame)
        else:
            ranges.append(current)
            current = [frame]
    ranges.append(current)
    return [
        {
            "start": round(group[0]["time"], 3),
            "end": round(group[-1]["time"], 3),
            "min_confidence": round(min(item["confidence"] for item in group), 3),
            "reason": "遮挡/多人重叠" if any(item["overlap"] for item in group) else "身份匹配置信度较低",
        }
        for group in ranges
    ]


def tracking_qa_frames(video_path: Path, keyframes: list, records: list, selection_time: float,
                       ranges: list, limit: int = 12, anchor_times: list = None) -> list:
    # Always keep timeline coverage, then distribute risky frames across the
    # whole clip. Simply sorting every low-confidence range and truncating the
    # first N made late identity swaps invisible in the review UI.
    timeline_indices = {0, len(keyframes) - 1}
    for ratio in (0.25, 0.5, 0.75):
        timeline_indices.add(min(len(keyframes) - 1, int(round((len(keyframes) - 1) * ratio))))
    for anchor_time in (anchor_times or [selection_time]):
        timeline_indices.add(min(
            range(len(keyframes)),
            key=lambda i: abs(keyframes[i]["time"] - float(anchor_time)),
        ))
    risk_indices = []
    for item in ranges:
        candidates = [
            (index, frame) for index, frame in enumerate(keyframes)
            if item["start"] - 1e-6 <= frame["time"] <= item["end"] + 1e-6
        ]
        if candidates:
            risk_indices.append(min(candidates, key=lambda pair: pair[1]["confidence"])[0])

    indices = set(timeline_indices)
    available_slots = max(0, limit - len(indices))
    unique_risks = sorted(set(risk_indices))
    if len(unique_risks) <= available_slots:
        indices.update(unique_risks)
    elif available_slots:
        spread_positions = np.linspace(0, len(unique_risks) - 1, available_slots)
        indices.update(unique_risks[int(round(position))] for position in spread_positions)
        # If rounding collided with an existing timeline sample, fill the slot
        # with the globally least-confident remaining risk frame.
        for risk_index in sorted(unique_risks, key=lambda index: keyframes[index]["confidence"]):
            if len(indices) >= limit:
                break
            indices.add(risk_index)
    ordered = []
    seen_times = set()
    for index in sorted(indices):
        time_key = round(float(keyframes[index]["time"]), 3)
        if time_key in seen_times:
            continue
        seen_times.add(time_key)
        ordered.append(index)
    if len(ordered) < limit:
        fallback_indices = [int(round(position)) for position in np.linspace(0, len(keyframes) - 1, limit * 2)]
        for index in fallback_indices:
            time_key = round(float(keyframes[index]["time"]), 3)
            if time_key in seen_times:
                continue
            seen_times.add(time_key)
            ordered.append(index)
            if len(ordered) >= limit:
                break
        ordered.sort()
    capture = cv2.VideoCapture(str(video_path))
    output = []
    for index in ordered:
        frame_data = keyframes[index]
        capture.set(cv2.CAP_PROP_POS_MSEC, frame_data["time"] * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        clean_image = encode_jpeg(frame)
        correction_boxes = correction_candidate_boxes(
            frame,
            records[index].get("detections", []),
        )
        box = frame_data["box"]
        color = (42, 196, 111)
        if frame_data["status"] == "missing":
            color = (54, 54, 230)
        elif frame_data["confidence"] < 0.48 or frame_data["overlap"] or frame_data["ambiguous"]:
            color = (24, 166, 239)
        x1, y1, x2, y2 = [int(box[key]) for key in ("x1", "y1", "x2", "y2")]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, max(2, round(frame.shape[1] / 420)))
        label = f"{frame_data['time']:.1f}s  {frame_data['status']}  {frame_data['confidence']:.0%}"
        cv2.rectangle(frame, (x1, max(0, y1 - 30)), (min(frame.shape[1] - 1, x1 + 255), y1), color, -1)
        cv2.putText(frame, label, (x1 + 6, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
        output.append({
            "time": frame_data["time"],
            "relative_time": frame_data["relative_time"],
            "confidence": frame_data["confidence"],
            "status": frame_data["status"],
            "requires_review": bool(
                frame_data["status"] == "missing" or frame_data["confidence"] < 0.48
                or frame_data["overlap"] or frame_data["ambiguous"]
            ),
            "image": encode_jpeg(frame),
            "correction_image": clean_image,
            "width": frame.shape[1],
            "height": frame.shape[0],
            "candidates": correction_boxes,
        })
    capture.release()
    return output


def track_video(video_path: Path, start: float, end: float, selection_time: float,
                seed_box_input: dict, sample_fps: float = 4.0, reuse_tracking_id: str = "",
                corrections: list = None) -> dict:
    if not ai_status()["ready"]:
        DETECTOR._load()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AnalysisError("video_unreadable", "AI 服务无法读取这个视频。", "请优先使用 H.264 编码的 MP4 文件。")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    duration = frame_count / fps if fps > 0 else end
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise AnalysisError("video_metadata", "无法读取视频尺寸。", "请转换为 MP4 后重试。")
    safe_start = max(0.0, min(float(start), max(duration - 0.05, 0.0)))
    safe_end = max(safe_start + 0.05, min(float(end), duration))
    readable_end = min(safe_end, max(safe_start, duration - 1.0 / max(fps, 1.0)))
    safe_selection = max(safe_start, min(float(selection_time), readable_end))
    seed_box = clamp_box(seed_box_input, width, height)

    capture.set(cv2.CAP_PROP_POS_MSEC, safe_selection * 1000.0)
    ok, seed_frame = capture.read()
    if not ok:
        capture.release()
        raise AnalysisError("anchor_unreadable", "无法读取用户选人的锚点画面。", "请返回上一步换一张候选画面。")
    anchor_inputs = [{
        "time": safe_selection,
        "box": seed_box,
        "histogram": appearance_histogram(seed_frame, seed_box),
        "signature": appearance_signature(seed_frame, seed_box),
        "detector_confidence": float(seed_box_input.get("confidence", 1.0)),
        "is_correction": False,
    }]
    for correction in corrections or []:
        if not isinstance(correction, dict) or not isinstance(correction.get("box"), dict):
            continue
        correction_time = max(
            safe_start,
            min(float(correction.get("timestamp", safe_selection)), readable_end),
        )
        correction_box = clamp_box(correction["box"], width, height)
        capture.set(cv2.CAP_PROP_POS_MSEC, correction_time * 1000.0)
        correction_ok, correction_frame = capture.read()
        if not correction_ok:
            capture.release()
            raise AnalysisError(
                "correction_unreadable",
                "无法读取其中一个人工纠偏画面。",
                "请取消这个纠偏点，改选相邻的复核画面。",
            )
        anchor_inputs.append({
            "time": correction_time,
            "box": correction_box,
            "histogram": appearance_histogram(correction_frame, correction_box),
            "signature": appearance_signature(correction_frame, correction_box),
            "detector_confidence": float(correction.get(
                "detectionConfidence",
                correction["box"].get("confidence", 1.0),
            )),
            "is_correction": True,
        })
    cached = get_tracking_cache(reuse_tracking_id)
    cache_matches = bool(
        cached
        and int(cached.get("width", 0)) == width
        and int(cached.get("height", 0)) == height
        and abs(float(cached.get("start", -1.0)) - safe_start) <= 0.02
        and abs(float(cached.get("end", -1.0)) - safe_end) <= 0.02
        and cached.get("records")
    )
    reused_detections = bool(cache_matches)
    if cache_matches:
        records = cached["records"]
    else:
        times = tracking_sample_times(safe_start, readable_end, safe_selection, sample_fps)
        records = []
        for timestamp in times:
            capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000.0)
            ok, frame = capture.read()
            if not ok:
                records.append({"time": timestamp, "detections": []})
                continue
            records.append({"time": timestamp, "detections": enrich_detections(frame, DETECTOR.detect(frame))})
    capture.release()
    if not records:
        raise AnalysisError("tracking_no_frames", "没有成功读取跟踪画面。", "请缩短保留片段后重试。")

    anchors_by_index = {}
    for anchor in anchor_inputs:
        anchor["index"] = min(
            range(len(records)),
            key=lambda index: abs(records[index]["time"] - anchor["time"]),
        )
        anchors_by_index[int(anchor["index"])] = anchor
    anchors = [anchors_by_index[index] for index in sorted(anchors_by_index)]
    selected = build_multi_anchor_path(records, anchors, width, height)
    redetection = augment_risk_detections(
        video_path,
        records,
        selected,
        width,
        height,
        {int(anchor["index"]) for anchor in anchors},
    )
    if redetection["candidates"]:
        selected = build_multi_anchor_path(records, anchors, width, height)

    keyframes = []
    for index, record in enumerate(records):
        item = selected.get(index)
        if item is None:
            raise AnalysisError(
                "tracking_path_incomplete",
                "多锚点轨迹没有覆盖完整片段。",
                "请减少纠偏点后重试。",
            )
        keyframes.append({
            "time": round(record["time"], 3),
            "relative_time": round(record["time"] - safe_start, 3),
            **item,
        })
    sample_step = (safe_end - safe_start) / max(1, len(keyframes) - 1)
    ranges = confidence_ranges(keyframes, sample_step)
    detected = [frame for frame in keyframes if frame["status"] != "missing"]
    reliable = [
        frame for frame in keyframes
        if frame["status"] in {"anchor", "tracked"} and not frame["overlap"] and not frame["ambiguous"]
    ]
    average_confidence = sum(frame["confidence"] for frame in keyframes) / len(keyframes)
    coverage = len(detected) / len(keyframes)
    reliability = len(reliable) / len(keyframes)
    confirmed_anchor_times = [records[int(anchor["index"])]["time"] for anchor in anchors]
    qa_frames = tracking_qa_frames(
        video_path,
        keyframes,
        records,
        safe_selection,
        ranges,
        anchor_times=confirmed_anchor_times,
    )
    previous_keyframes = cached.get("keyframes", []) if cache_matches else []
    changed_samples = 0
    largest_center_shift = 0.0
    if len(previous_keyframes) == len(keyframes):
        diagonal = max(1.0, math.hypot(width, height))
        for previous, current in zip(previous_keyframes, keyframes):
            previous_box = previous["box"]
            current_box = current["box"]
            center_shift = math.hypot(
                float(previous_box["cx"]) - float(current_box["cx"]),
                float(previous_box["cy"]) - float(current_box["cy"]),
            )
            largest_center_shift = max(largest_center_shift, center_shift)
            if center_shift >= diagonal * 0.012 or box_iou(previous_box, current_box) < 0.82:
                changed_samples += 1
    correction_anchor_results = []
    for anchor in anchors:
        if not anchor.get("is_correction"):
            continue
        anchor_index = int(anchor["index"])
        current_box = keyframes[anchor_index]["box"]
        previous_box = (
            previous_keyframes[anchor_index]["box"]
            if len(previous_keyframes) == len(keyframes) else None
        )
        correction_anchor_results.append({
            "time": round(float(keyframes[anchor_index]["time"]), 3),
            "status": keyframes[anchor_index]["status"],
            "hard_anchor_applied": box_iou(current_box, anchor["box"]) >= 0.995,
            "previous_iou": round(box_iou(previous_box, current_box), 4) if previous_box else None,
        })
    tracking_id = uuid.uuid4().hex[:12]
    store_tracking_cache(tracking_id, {
        "width": width,
        "height": height,
        "start": safe_start,
        "end": safe_end,
        "records": records,
        "keyframes": keyframes,
    })
    return {
        "tracking_id": tracking_id,
        "engine": (
            f"复用检测缓存 + {len(anchors)} 个身份锚点分段修正 + 风险帧按需重检测"
            if reused_detections and len(anchors) > 1 else
            "主舞区理解 + 补全身份特征 + 双向路径 + 风险帧按需重检测"
        ),
        "source": {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "duration": round(duration, 3),
            "start": round(safe_start, 3),
            "end": round(safe_end, 3),
        },
        "anchor": {
            "time": round(safe_selection, 3),
            "box": seed_box,
        },
        "anchors": [
            {
                "time": round(float(records[int(anchor["index"])]["time"]), 3),
                "box": anchor["box"],
                "is_correction": bool(anchor.get("is_correction", False)),
            }
            for anchor in anchors
        ],
        "sample_fps": round(len(keyframes) / max(0.05, safe_end - safe_start), 3),
        "keyframes": keyframes,
        "low_confidence_ranges": ranges,
        "qa_frames": qa_frames,
        "correction_summary": {
            "applied_count": len(correction_anchor_results),
            "all_hard_anchors_applied": bool(correction_anchor_results) and all(
                item["hard_anchor_applied"] for item in correction_anchor_results
            ),
            "changed_samples": changed_samples,
            "changed_ratio": round(changed_samples / max(1, len(keyframes)), 4),
            "largest_center_shift_pixels": round(largest_center_shift, 2),
            "anchors": correction_anchor_results,
        },
        "metrics": {
            "samples": len(keyframes),
            "detected_samples": len(detected),
            "missing_samples": len(keyframes) - len(detected),
            "coverage": round(coverage, 4),
            "average_confidence": round(average_confidence, 4),
            "reliable_ratio": round(reliability, 4),
            "review_ranges": len(ranges),
            "requires_review": bool(ranges),
            "reused_detections": reused_detections,
            "reused_detection_samples": len(records) if reused_detections else 0,
            "anchor_count": len(anchors),
            "correction_anchor_count": sum(
                1 for anchor in anchors if anchor.get("is_correction", False)
            ),
            "redetected_frames": redetection["frames"],
            "redetection_candidates": redetection["candidates"],
        },
    }


def track_video_high_precision(video_path: Path, start: float, end: float,
                               selection_time: float, seed_box_input: dict,
                               corrections: list = None) -> dict:
    """Run SAM 2.1 and adapt its mask trajectory to the existing product contract."""
    runtime = sam2_runtime_status()
    if track_video_object is None or not runtime.get("available"):
        raise AnalysisError(
            "sam2_unavailable",
            "高精度人物跟踪服务尚未部署。",
            "当前本地版继续使用轻量模式；部署 cloud/modal_app.py 后，手机网站会默认使用 SAM 2.1。",
        )
    started = time.perf_counter()
    try:
        result = track_video_object(
            video_path, start, end, selection_time, seed_box_input,
            corrections=corrections or [],
        )
    except Exception as exc:
        raise AnalysisError(
            "sam2_tracking_failed", "SAM 2.1 高精度人物跟踪失败。", str(exc)[-1200:]
        ) from exc
    keyframes = result.get("keyframes") or []
    if not keyframes:
        raise AnalysisError("sam2_empty", "SAM 2.1 没有返回人物轨迹。", "请更换更清晰的初始选人画面。")
    sample_step = (float(end) - float(start)) / max(1, len(keyframes) - 1)
    ranges = confidence_ranges(keyframes, sample_step)
    records = [{"time": frame["time"], "detections": []} for frame in keyframes]
    anchor_times = [float(anchor["time"]) for anchor in result.get("anchors", [])]
    qa_frames = tracking_qa_frames(
        video_path, keyframes, records, float(selection_time), ranges,
        anchor_times=anchor_times,
    )
    detected = [frame for frame in keyframes if frame["status"] != "missing"]
    reliable = [frame for frame in keyframes if frame["status"] in {"anchor", "tracked"}]
    average_confidence = sum(float(frame["confidence"]) for frame in keyframes) / len(keyframes)
    correction_count = sum(bool(anchor.get("is_correction")) for anchor in result.get("anchors", []))
    result.update({
        "tracking_id": uuid.uuid4().hex[:12],
        "anchor": {"time": round(float(selection_time), 3), "box": clamp_box(
            seed_box_input, int(result["source"]["width"]), int(result["source"]["height"])
        )},
        "low_confidence_ranges": ranges,
        "qa_frames": qa_frames,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "metrics": {
            "samples": len(keyframes),
            "detected_samples": len(detected),
            "missing_samples": len(keyframes) - len(detected),
            "coverage": round(len(detected) / len(keyframes), 4),
            "average_confidence": round(average_confidence, 4),
            "reliable_ratio": round(len(reliable) / len(keyframes), 4),
            "review_ranges": len(ranges),
            "requires_review": bool(ranges),
            "reused_detections": bool(corrections),
            "reused_detection_samples": 0,
            "anchor_count": len(result.get("anchors", [])),
            "correction_anchor_count": correction_count,
            "redetected_frames": 0,
            "redetection_candidates": 0,
        },
    })
    return result


CAMERA_PROFILES = {
    "stable": {
        "label": "稳定跟随",
        "smoothing_seconds": 0.72,
        "dead_zone_x": 0.10,
        "dead_zone_y": 0.075,
        "max_speed_x": 180.0,
        "max_speed_y": 110.0,
        "max_acceleration_x": 430.0,
        "max_acceleration_y": 260.0,
    },
    "balanced": {
        "label": "自然跟随",
        "smoothing_seconds": 0.48,
        "dead_zone_x": 0.072,
        "dead_zone_y": 0.055,
        "max_speed_x": 230.0,
        "max_speed_y": 145.0,
        "max_acceleration_x": 560.0,
        "max_acceleration_y": 330.0,
    },
    "responsive": {
        "label": "灵敏跟随",
        "smoothing_seconds": 0.30,
        "dead_zone_x": 0.048,
        "dead_zone_y": 0.038,
        "max_speed_x": 260.0,
        "max_speed_y": 175.0,
        "max_acceleration_x": 680.0,
        "max_acceleration_y": 410.0,
    },
}

FRAMING_ZOOMS = {
    "wide": ("全身宽松", 2.20),
    "standard": ("标准直拍", 2.50),
    "close": ("近景突出", 2.95),
}

HORIZONTAL_ZOOMS = {
    "wide": ("原景宽松", 1.00),
    "standard": ("适度放大", 1.15),
    "close": ("重点突出", 1.30),
}


def clamp_number(value: float, lower: float, upper: float) -> float:
    return max(lower, min(float(value), upper))


def median_filter(values: list, radius: int = 2) -> list:
    if not values:
        return []
    return [
        float(np.median(values[max(0, index - radius):min(len(values), index + radius + 1)]))
        for index in range(len(values))
    ]


def exponential_smooth(values: list, times: list, smoothing_seconds: float) -> list:
    if not values:
        return []

    def one_direction(series, timestamps):
        output = [float(series[0])]
        for index in range(1, len(series)):
            delta = max(0.001, abs(float(timestamps[index]) - float(timestamps[index - 1])))
            alpha = 1.0 - math.exp(-delta / max(0.05, smoothing_seconds))
            output.append(output[-1] + alpha * (float(series[index]) - output[-1]))
        return output

    forward = one_direction(values, times)
    backward = list(reversed(one_direction(list(reversed(values)), list(reversed(times)))))
    return [(left + right) / 2.0 for left, right in zip(forward, backward)]


def constrained_camera_axis(targets: list, times: list, dead_zone: float, max_speed: float,
                            max_acceleration: float, lower: float, upper: float) -> list:
    if not targets:
        return []
    camera = clamp_number(targets[0], lower, upper)
    velocity = 0.0
    output = [camera]
    for index in range(1, len(targets)):
        delta_time = max(0.001, float(times[index]) - float(times[index - 1]))
        target = clamp_number(targets[index], lower, upper)
        difference = target - camera
        if abs(difference) <= dead_zone:
            goal = camera
        else:
            goal = target - math.copysign(dead_zone, difference)
        desired_velocity = clamp_number((goal - camera) / delta_time, -max_speed, max_speed)
        velocity_change = max_acceleration * delta_time
        velocity = clamp_number(desired_velocity, velocity - velocity_change, velocity + velocity_change)
        previous = camera
        camera = clamp_number(camera + velocity * delta_time, lower, upper)
        if (goal - previous) * (goal - camera) < 0:
            camera = goal
            velocity = 0.0
        output.append(camera)
    return output


def simplify_camera_points(points: list, tolerance: float = 1.4, max_gap: float = 1.0) -> list:
    if len(points) <= 2:
        return points
    keep = {0, len(points) - 1}

    def reduce_segment(first: int, last: int) -> None:
        if last <= first + 1:
            return
        start = points[first]
        finish = points[last]
        span = max(0.001, finish["t"] - start["t"])
        largest_error = -1.0
        largest_index = None
        for index in range(first + 1, last):
            ratio = (points[index]["t"] - start["t"]) / span
            expected_x = start["x"] + (finish["x"] - start["x"]) * ratio
            expected_y = start["y"] + (finish["y"] - start["y"]) * ratio
            error = math.hypot(points[index]["x"] - expected_x, points[index]["y"] - expected_y)
            if error > largest_error:
                largest_error = error
                largest_index = index
        if largest_index is not None and largest_error > tolerance:
            keep.add(largest_index)
            reduce_segment(first, largest_index)
            reduce_segment(largest_index, last)

    reduce_segment(0, len(points) - 1)
    selected = sorted(keep)
    for first, last in zip(selected, selected[1:]):
        if points[last]["t"] - points[first]["t"] > max_gap:
            for index in range(first + 1, last):
                if points[index]["t"] - points[first]["t"] < max_gap:
                    continue
                keep.add(index)
                first = index
    return [points[index] for index in sorted(keep)]


def estimate_fancam_body_envelope(box: dict, width: int, height: int) -> dict:
    """Turn a detector fragment into a conservative head-to-feet dance envelope."""
    safe = clamp_box(box, width, height)
    box_width = max(2.0, safe["x2"] - safe["x1"])
    box_height = max(2.0, safe["y2"] - safe["y1"])
    aspect = box_height / box_width
    height_ratio = box_height / max(1.0, height)
    if aspect < 1.65 and height_ratio < 0.30:
        top_pad, bottom_pad, side_pad = 0.70, 0.80, 0.42
    elif aspect < 2.35 and height_ratio < 0.34:
        top_pad, bottom_pad, side_pad = 0.48, 0.58, 0.28
    else:
        top_pad, bottom_pad, side_pad = 0.18, 0.20, 0.16
    return clamp_box({
        "x1": safe["x1"] - box_height * side_pad,
        "y1": safe["y1"] - box_height * top_pad,
        "x2": safe["x2"] + box_height * side_pad,
        "y2": safe["y2"] + box_height * bottom_pad,
    }, width, height)


def build_camera_plan(tracking: dict, framing: str = "standard", motion: str = "balanced") -> dict:
    source = tracking.get("source") or {}
    frames = tracking.get("keyframes") or []
    if not frames:
        raise AnalysisError("tracking_missing", "没有收到人物轨迹。", "请返回身份跟踪阶段重新分析。")
    width = int(source.get("width") or 0)
    height = int(source.get("height") or 0)
    start = float(source.get("start") or 0.0)
    end = float(source.get("end") or 0.0)
    if width <= 0 or height <= 0 or end <= start:
        raise AnalysisError("tracking_invalid", "人物轨迹缺少视频尺寸或时间范围。", "请重新跟踪后再生成运镜。")
    framing = framing if framing in FRAMING_ZOOMS else "standard"
    motion = motion if motion in CAMERA_PROFILES else "balanced"
    framing_label, requested_zoom = FRAMING_ZOOMS[framing]
    profile = CAMERA_PROFILES[motion]

    ordered = sorted(frames, key=lambda item: float(item.get("time", 0.0)))
    if float(ordered[0].get("time", start)) > start + 0.001:
        ordered.insert(0, {**ordered[0], "time": start})
    if float(ordered[-1].get("time", end)) < end - 0.001:
        ordered.append({**ordered[-1], "time": end})

    base_unit = min(width / 9.0, height / 16.0)
    # Ceil keeps the actual crop at or wider than the requested framing; floor
    # could silently zoom tighter and clip a dancer's extended hands.
    requested_unit = max(2, int(math.ceil(base_unit / requested_zoom)))
    body_envelopes = [estimate_fancam_body_envelope(item["box"], width, height) for item in ordered]
    body_heights = [max(2.0, body["y2"] - body["y1"]) for body in body_envelopes]
    body_widths = [max(2.0, body["x2"] - body["x1"]) for body in body_envelopes]
    # Use the inferred whole-body/action envelope for safety. The requested
    # framing remains the lower bound so standard direct-cam footage stays tight.
    safe_height = float(np.percentile(body_heights, 90)) * 1.06
    safe_width = float(np.percentile(body_widths, 85)) * 1.08
    safe_unit = int(math.ceil(max(safe_height / 16.0, safe_width / 9.0)))
    unit = max(requested_unit, safe_unit)
    unit = max(2, min(unit, int(math.floor(base_unit))))
    crop_width = int(9 * unit)
    crop_height = int(16 * unit)
    applied_zoom = height / max(1.0, crop_height)

    times = [float(item.get("time", start)) - start for item in ordered]
    raw_x = [float(body["cx"]) for body in body_envelopes]
    target_headroom = 0.055
    target_footroom = {"wide": 0.10, "standard": 0.08, "close": 0.065}[framing]
    raw_y = []
    for body in body_envelopes:
        estimated_head = float(body["y1"])
        estimated_feet = float(body["y2"])
        foot_biased_top = estimated_feet - crop_height * (1.0 - target_footroom)
        head_safe_top = estimated_head - crop_height * target_headroom
        raw_y.append(min(foot_biased_top, head_safe_top))
    filtered_x = median_filter(raw_x)
    filtered_y = median_filter(raw_y)
    smooth_x = exponential_smooth(filtered_x, times, profile["smoothing_seconds"])
    smooth_y = exponential_smooth(filtered_y, times, profile["smoothing_seconds"] * 1.25)

    camera_x = constrained_camera_axis(
        smooth_x,
        times,
        crop_width * profile["dead_zone_x"],
        profile["max_speed_x"],
        profile["max_acceleration_x"],
        crop_width / 2.0,
        width - crop_width / 2.0,
    )
    camera_y = constrained_camera_axis(
        smooth_y,
        times,
        crop_height * profile["dead_zone_y"],
        profile["max_speed_y"],
        profile["max_acceleration_y"],
        0.0,
        height - crop_height,
    )

    detailed = []
    for timestamp, center_x, center_y in zip(times, camera_x, camera_y):
        detailed.append({
            "t": round(timestamp, 4),
            "source_time": round(start + timestamp, 4),
            "x": round(clamp_number(center_x - crop_width / 2.0, 0.0, width - crop_width), 3),
            "y": round(clamp_number(center_y, 0.0, height - crop_height), 3),
        })
    simplified = simplify_camera_points(detailed)
    return {
        "aspect": "9:16",
        "width": width,
        "height": height,
        "start": round(start, 4),
        "end": round(end, 4),
        "duration": round(end - start, 4),
        "crop_width": crop_width,
        "crop_height": crop_height,
        "framing": framing,
        "framing_label": framing_label,
        "requested_zoom": requested_zoom,
        "applied_zoom": round(applied_zoom, 3),
        "motion": motion,
        "motion_label": profile["label"],
        "dead_zone_pixels": round(crop_width * profile["dead_zone_x"], 1),
        "max_pan_speed": profile["max_speed_x"],
        "target_headroom_percent": round(target_headroom * 100.0, 1),
        "target_footroom_percent": round(target_footroom * 100.0, 1),
        "raw_samples": len(ordered),
        "camera_keyframes": len(simplified),
        "keyframes": simplified,
    }


def simplify_focus_points(points: list, fields: tuple = ("x", "y", "width", "height"),
                          tolerance: float = 2.0, max_gap: float = 0.75) -> list:
    """Keep a compact but accurate moving soft-mask path."""
    if len(points) <= 2:
        return points
    keep = {0, len(points) - 1}
    def reduce_segment(first: int, last: int) -> None:
        if last <= first + 1:
            return
        start_point = points[first]
        end_point = points[last]
        span = max(0.001, end_point["t"] - start_point["t"])
        largest_error = -1.0
        largest_index = None
        for index in range(first + 1, last):
            ratio = (points[index]["t"] - start_point["t"]) / span
            error = max(
                abs(
                    points[index][field]
                    - (start_point[field] + (end_point[field] - start_point[field]) * ratio)
                )
                for field in fields
            )
            if error > largest_error:
                largest_error = error
                largest_index = index
        if largest_index is not None and largest_error > tolerance:
            keep.add(largest_index)
            reduce_segment(first, largest_index)
            reduce_segment(largest_index, last)

    reduce_segment(0, len(points) - 1)
    selected = sorted(keep)
    for original_first, last in zip(selected, selected[1:]):
        first = original_first
        for index in range(original_first + 1, last):
            if points[index]["t"] - points[first]["t"] >= max_gap:
                keep.add(index)
                first = index
    return [points[index] for index in sorted(keep)]


def build_horizontal_focus_plan(tracking: dict, framing: str = "standard",
                                motion: str = "balanced") -> dict:
    """Build a same-aspect crop plus a full-height, feathered subject focus band."""
    source = tracking.get("source") or {}
    frames = tracking.get("keyframes") or []
    if not frames:
        raise AnalysisError("tracking_missing", "没有收到人物轨迹。", "请先完成身份跟踪。")
    width = int(source.get("width") or 0)
    height = int(source.get("height") or 0)
    start = float(source.get("start") or 0.0)
    end = float(source.get("end") or 0.0)
    if width <= 0 or height <= 0 or end <= start:
        raise AnalysisError("tracking_invalid", "人物轨迹缺少视频尺寸或时间范围。", "请重新跟踪后再生成横屏直拍。")
    framing = framing if framing in HORIZONTAL_ZOOMS else "standard"
    motion = motion if motion in CAMERA_PROFILES else "balanced"
    framing_label, requested_zoom = HORIZONTAL_ZOOMS[framing]
    profile = CAMERA_PROFILES[motion]

    ordered = sorted(frames, key=lambda item: float(item.get("time", 0.0)))
    if float(ordered[0].get("time", start)) > start + 0.001:
        ordered.insert(0, {**ordered[0], "time": start})
    if float(ordered[-1].get("time", end)) < end - 0.001:
        ordered.append({**ordered[-1], "time": end})

    envelopes = [estimate_fancam_body_envelope(item["box"], width, height) for item in ordered]
    times = [float(item.get("time", start)) - start for item in ordered]
    raw_center_x = [float(body["cx"]) for body in envelopes]
    raw_center_y = [float(body["cy"]) for body in envelopes]
    body_widths = [max(2.0, float(body["x2"] - body["x1"])) for body in envelopes]
    body_heights = [max(2.0, float(body["y2"] - body["y1"])) for body in envelopes]

    divisor = math.gcd(width, height)
    aspect_width, aspect_height = width // divisor, height // divisor
    base_unit = min(width / aspect_width, height / aspect_height)
    requested_unit = max(2, int(round(base_unit / requested_zoom)))
    safe_unit = int(math.ceil(max(
        float(np.percentile(body_widths, 90)) * 1.20 / aspect_width,
        float(np.percentile(body_heights, 90)) * 1.16 / aspect_height,
    )))
    unit = max(requested_unit, safe_unit)
    unit = max(2, min(unit, int(math.floor(base_unit))))
    crop_width = int(aspect_width * unit)
    crop_height = int(aspect_height * unit)
    applied_zoom = min(width / crop_width, height / crop_height)

    filtered_x = median_filter(raw_center_x)
    filtered_y = median_filter(raw_center_y)
    smooth_x = exponential_smooth(filtered_x, times, profile["smoothing_seconds"])
    smooth_y = exponential_smooth(filtered_y, times, profile["smoothing_seconds"] * 1.15)
    camera_x = constrained_camera_axis(
        smooth_x,
        times,
        crop_width * profile["dead_zone_x"],
        profile["max_speed_x"],
        profile["max_acceleration_x"],
        crop_width / 2.0,
        width - crop_width / 2.0,
    )
    camera_y = constrained_camera_axis(
        smooth_y,
        times,
        crop_height * profile["dead_zone_y"],
        profile["max_speed_y"],
        profile["max_acceleration_y"],
        crop_height / 2.0,
        height - crop_height / 2.0,
    )

    raw_focus_widths = [
        clamp_number(
            max(crop_width * 0.20, body_width * 2.00),
            crop_width * 0.19,
            crop_width * 0.38,
        )
        for body_width in body_widths
    ]
    focus_centers = exponential_smooth(filtered_x, times, 0.22)
    focus_widths = exponential_smooth(median_filter(raw_focus_widths), times, 0.52)
    detailed = []
    for timestamp, subject_x, crop_center_x, crop_center_y, focus_width in zip(
        times, focus_centers, camera_x, camera_y, focus_widths
    ):
        crop_x = clamp_number(crop_center_x - crop_width / 2.0, 0.0, width - crop_width)
        crop_y = clamp_number(crop_center_y - crop_height / 2.0, 0.0, height - crop_height)
        focus_width = clamp_number(focus_width, 2.0, crop_width)
        focus_x = clamp_number(subject_x - crop_x - focus_width / 2.0, 0.0, crop_width - focus_width)
        detailed.append({
            "t": round(timestamp, 4),
            "source_time": round(start + timestamp, 4),
            "crop_x": round(crop_x, 3),
            "crop_y": round(crop_y, 3),
            "focus_x": round(focus_x, 3),
            "focus_width": round(focus_width, 3),
        })
    simplified = simplify_focus_points(
        detailed,
        fields=("crop_x", "crop_y", "focus_x", "focus_width"),
    )
    feather_pixels = round(max(12.0, min(crop_width, crop_height) * 0.028), 1)
    clear_zone_percent = float(np.median([point["focus_width"] for point in detailed])) / crop_width * 100.0
    return {
        "aspect": f"{aspect_width}:{aspect_height}",
        "width": width,
        "height": height,
        "start": round(start, 4),
        "end": round(end, 4),
        "duration": round(end - start, 4),
        "style": "horizontal-focus",
        "style_label": "横屏两侧柔焦",
        "mask_shape": "full-height-soft-band",
        "feather_pixels": feather_pixels,
        "blur_label": "强柔焦",
        "framing": framing,
        "framing_label": framing_label,
        "requested_zoom": requested_zoom,
        "applied_zoom": round(applied_zoom, 3),
        "motion": motion,
        "motion_label": profile["label"],
        "crop_width": crop_width,
        "crop_height": crop_height,
        "clear_zone_percent": round(clear_zone_percent, 1),
        "vertical_masking": False,
        "raw_samples": len(ordered),
        "focus_keyframes": len(simplified),
        "keyframes": simplified,
    }


def camera_plan_qa(video_path: Path, tracking: dict, plan: dict, limit: int = 5) -> list:
    points = plan["keyframes"]
    indices = sorted({int(round((len(points) - 1) * ratio)) for ratio in np.linspace(0.0, 1.0, limit)})
    tracked = tracking.get("keyframes") or []
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    readable_end = frame_count / fps - 1.0 / fps if frame_count > 0 else plan["end"]
    output = []
    for index in indices:
        point = points[index]
        source_time = min(float(point["source_time"]), max(plan["start"], readable_end))
        capture.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        nearest = min(tracked, key=lambda item: abs(float(item["time"]) - source_time))
        box = nearest["box"]
        x, y = int(round(point["x"])), int(round(point["y"]))
        crop_width, crop_height = plan["crop_width"], plan["crop_height"]
        thickness = max(2, round(frame.shape[1] / 420))
        cv2.rectangle(frame, (x, y), (x + crop_width, y + crop_height), (100, 255, 215), thickness)
        cv2.rectangle(
            frame,
            (int(box["x1"]), int(box["y1"])),
            (int(box["x2"]), int(box["y2"])),
            (111, 88, 255),
            thickness,
        )
        output.append({
            "time": round(source_time, 3),
            "image": encode_jpeg(frame),
        })
    capture.release()
    return output


def interpolate_focus_point(points: list, timestamp: float) -> dict:
    if timestamp <= float(points[0]["source_time"]):
        return points[0]
    if timestamp >= float(points[-1]["source_time"]):
        return points[-1]
    for first, second in zip(points, points[1:]):
        first_time = float(first["source_time"])
        second_time = float(second["source_time"])
        if first_time <= timestamp <= second_time:
            ratio = (timestamp - first_time) / max(0.001, second_time - first_time)
            return {
                field: float(first[field]) + (float(second[field]) - float(first[field])) * ratio
                for field in ("crop_x", "crop_y", "focus_x", "focus_width")
            }
    return points[-1]


def horizontal_focus_qa(video_path: Path, plan: dict, limit: int = 5) -> list:
    points = plan["keyframes"]
    indices = sorted({int(round((len(points) - 1) * ratio)) for ratio in np.linspace(0.0, 1.0, limit)})
    capture = cv2.VideoCapture(str(video_path))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    readable_end = frame_count / fps - 1.0 / fps if frame_count > 0 else plan["end"]
    output = []
    for index in indices:
        source_time = min(float(points[index]["source_time"]), max(plan["start"], readable_end))
        capture.set(cv2.CAP_PROP_POS_MSEC, source_time * 1000.0)
        ok, frame = capture.read()
        if not ok:
            continue
        point = interpolate_focus_point(points, source_time)
        crop_x = int(round(point["crop_x"]))
        crop_y = int(round(point["crop_y"]))
        crop = frame[crop_y:crop_y + plan["crop_height"], crop_x:crop_x + plan["crop_width"]]
        if crop.size == 0:
            continue
        blur_sigma = max(12.0, min(crop.shape[:2]) * 0.032)
        blurred = cv2.GaussianBlur(crop, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
        mask = np.zeros(crop.shape[:2], dtype=np.uint8)
        x1 = int(round(point["focus_x"]))
        x2 = int(round(point["focus_x"] + point["focus_width"]))
        cv2.rectangle(mask, (x1, 0), (x2, crop.shape[0]), 255, -1)
        mask = cv2.GaussianBlur(
            mask,
            (0, 0),
            sigmaX=float(plan["feather_pixels"]),
            sigmaY=float(plan["feather_pixels"]),
        )
        alpha = mask.astype(np.float32)[:, :, None] / 255.0
        composited = np.clip(crop.astype(np.float32) * alpha + blurred.astype(np.float32) * (1.0 - alpha), 0, 255)
        output.append({"time": round(source_time, 3), "image": encode_jpeg(composited.astype(np.uint8))})
    capture.release()
    return output


def rendered_video_poster(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_MSEC, 120.0)
    ok, frame = capture.read()
    capture.release()
    return encode_jpeg(frame) if ok else None


def piecewise_linear_expression(points: list, field: str) -> str:
    values = [(float(point["t"]), float(point[field])) for point in points]
    expression = f"{values[-1][1]:.3f}"
    for index in range(len(values) - 2, -1, -1):
        first_time, first_value = values[index]
        second_time, second_value = values[index + 1]
        span = max(0.001, second_time - first_time)
        interpolation = f"{first_value:.3f}+({second_value-first_value:.3f})*((t-{first_time:.3f})/{span:.3f})"
        expression = f"if(lt(t,{second_time:.3f}),{interpolation},{expression})"
    return expression


def limit_expression_points(points: list, fields: tuple, max_points: int = 24) -> list:
    """Bound FFmpeg expression depth while preserving the largest path bends.

    Deeply nested drawbox expressions can exceed FFmpeg's filter parser/evaluator
    limit on longer clips.  Remove the point with the smallest interpolation error
    until the expression is safely bounded; endpoints are always retained.
    """
    selected = list(points)
    while len(selected) > max(2, max_points):
        removal_index = None
        removal_error = None
        for index in range(1, len(selected) - 1):
            previous = selected[index - 1]
            current = selected[index]
            following = selected[index + 1]
            span = max(0.001, float(following["t"]) - float(previous["t"]))
            ratio = (float(current["t"]) - float(previous["t"])) / span
            error = max(
                abs(
                    float(current[field])
                    - (
                        float(previous[field])
                        + (float(following[field]) - float(previous[field])) * ratio
                    )
                )
                for field in fields
            )
            if removal_error is None or error < removal_error:
                removal_error = error
                removal_index = index
        if removal_index is None:
            break
        selected.pop(removal_index)
    return selected


def probe_media(path: Path) -> dict:
    ffprobe = find_media_tool("ffprobe")
    if ffprobe is None:
        return {}
    completed = subprocess.run(
        [str(ffprobe), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    data = json.loads(completed.stdout or "{}")
    streams = data.get("streams") or []
    video = next((stream for stream in streams if stream.get("codec_type") == "video"), {})
    audio = next((stream for stream in streams if stream.get("codec_type") == "audio"), None)
    return {
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "video_codec": video.get("codec_name"),
        "has_audio": bool(audio),
        "audio_codec": audio.get("codec_name") if audio else None,
        "duration": round(float((data.get("format") or {}).get("duration") or 0.0), 3),
        "file_size": path.stat().st_size,
    }


def cleanup_render_outputs() -> None:
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    generated = sorted(OUTPUTS.glob("onetake-*.mp4"), key=lambda path: path.stat().st_mtime, reverse=True)
    cutoff = time.time() - 24 * 60 * 60
    for index, path in enumerate(generated):
        if index >= 16 or path.stat().st_mtime < cutoff:
            try:
                path.unlink()
            except OSError:
                pass


def even_dimension(value: float) -> int:
    return max(2, int(round(value / 2.0)) * 2)


def render_reframed_video(video_path: Path, tracking: dict, framing: str, motion: str,
                           mode: str, source_name: str, output_mode: str = "vertical") -> dict:
    status = render_status()
    if not status["ready"]:
        raise AnalysisError(
            "renderer_missing",
            "没有找到本地视频渲染组件。",
            "请安装 FFmpeg，或把它放到项目 tools/ffmpeg/bin 目录后重启。",
        )
    mode = mode if mode in {"preview", "export"} else "preview"
    output_mode = output_mode if output_mode in {"vertical", "horizontal-focus"} else "vertical"
    plan = (
        build_horizontal_focus_plan(tracking, framing, motion)
        if output_mode == "horizontal-focus"
        else build_camera_plan(tracking, framing, motion)
    )
    if output_mode == "horizontal-focus":
        source_width, source_height = int(plan["width"]), int(plan["height"])
        preview_scale = min(1.0, 960.0 / source_width, 540.0 / source_height)
        render_scale = preview_scale if mode == "preview" else 1.0
        output_width = even_dimension(source_width * render_scale)
        output_height = even_dimension(source_height * render_scale)
    else:
        output_width = 540 if mode == "preview" else 1080
        output_height = 960 if mode == "preview" else 1920
    crf = 23 if mode == "preview" else 17
    preset = "veryfast" if mode == "preview" else "fast"
    identifier = uuid.uuid4().hex[:12]
    cleanup_render_outputs()
    mode_slug = "horizontal" if output_mode == "horizontal-focus" else "vertical"
    output_name = f"onetake-{identifier}-{mode_slug}-{mode}.mp4"
    output_path = OUTPUTS / output_name
    ffmpeg = find_media_tool("ffmpeg")

    if output_mode == "horizontal-focus":
        scale_x = output_width / float(plan["crop_width"])
        scale_y = output_height / float(plan["crop_height"])
        expression_points = limit_expression_points(
            plan["keyframes"],
            fields=("crop_x", "crop_y"),
            max_points=24,
        )
        plan["render_expression_keyframes"] = len(expression_points)
        expressions = {
            field: piecewise_linear_expression(expression_points, field)
            for field in ("crop_x", "crop_y")
        }
        # The crop path already keeps the subject near the camera centre.  A
        # camera-locked clear band therefore produces the intended follow
        # effect without asking drawbox to evaluate two deeply nested dynamic
        # expressions.  Some Windows FFmpeg builds reject those expressions on
        # 40s+ clips even after keyframe reduction.
        focus_centres = [
            (float(point["focus_x"]) + float(point["focus_width"]) / 2.0) * scale_x
            for point in plan["keyframes"]
        ]
        focus_widths = [float(point["focus_width"]) * scale_x for point in plan["keyframes"]]
        render_focus_width = even_dimension(float(np.median(focus_widths)))
        render_focus_width = max(2, min(output_width, render_focus_width))
        render_focus_centre = float(np.median(focus_centres))
        render_focus_x = int(round(clamp_number(
            render_focus_centre - render_focus_width / 2.0,
            0.0,
            output_width - render_focus_width,
        )))
        plan["render_focus_mode"] = "camera-locked-static-band"
        plan["render_focus_x"] = render_focus_x
        plan["render_focus_width"] = render_focus_width
        feather = max(8.0, float(plan["feather_pixels"]) * min(scale_x, scale_y))
        blur_sigma = max(12.0, min(output_width, output_height) * 0.032)
        graph = (
            f"[0:v]setpts=PTS-STARTPTS,"
            f"crop={plan['crop_width']}:{plan['crop_height']}:x='{expressions['crop_x']}':y='{expressions['crop_y']}':exact=1,"
            f"scale={output_width}:{output_height}:flags=lanczos,setsar=1,split=3[sharpin][blurin][maskin];"
            f"[sharpin]hqdn3d=.45:.45:1.2:1.2,unsharp=3:3:.12:3:3:0[sharp];"
            f"[blurin]gblur=sigma={blur_sigma:.2f}:steps=2[blurred];"
            f"[maskin]format=gray,geq=lum=0,"
            f"drawbox=x={render_focus_x}:y=0:"
            f"w={render_focus_width}:h={output_height}:color=white:t=fill,"
            f"gblur=sigma={feather:.2f}:steps=2[mask];"
            f"[blurred][sharp][mask]maskedmerge,format=yuv420p[v]"
        )
    else:
        x_expression = piecewise_linear_expression(plan["keyframes"], "x")
        y_expression = piecewise_linear_expression(plan["keyframes"], "y")
        graph = (
            f"[0:v]setpts=PTS-STARTPTS,"
            f"crop={plan['crop_width']}:{plan['crop_height']}:x='{x_expression}':y='{y_expression}':exact=1,"
            f"hqdn3d=.8:.8:2.4:2.4,scale={output_width}:{output_height}:flags=lanczos,"
            f"unsharp=3:3:.14:3:3:0,setsar=1[v]"
        )
    script_path = None
    started = time.perf_counter()
    try:
        with tempfile.NamedTemporaryFile(
            prefix="onetake-filter-", suffix=".txt", mode="w", encoding="utf-8", delete=False
        ) as script:
            script.write(graph)
            script_path = Path(script.name)
        command = [
            str(ffmpeg), "-hide_banner", "-loglevel", "error", "-y",
            "-ss", str(plan["start"]), "-t", str(plan["duration"]), "-i", str(video_path),
            "-filter_complex_script", str(script_path),
            "-map", "[v]", "-map", "0:a?",
            "-c:v", "libx264", "-crf", str(crf), "-preset", preset, "-pix_fmt", "yuv420p",
            "-c:a", "aac", "-b:a", "160k", "-shortest", "-movflags", "+faststart",
            str(output_path),
        ]
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0 or not output_path.exists():
            detail = (completed.stderr or completed.stdout or "FFmpeg exited without output").strip()[-1200:]
            raise AnalysisError("render_failed", "直拍视频生成失败。", detail)
    finally:
        if script_path is not None:
            try:
                script_path.unlink(missing_ok=True)
            except OSError:
                pass

    media = probe_media(output_path)
    source_stem = Path(source_name or "video").stem
    suffix = "preview" if mode == "preview" else ("horizontal-focus" if output_mode == "horizontal-focus" else "9x16")
    return {
        "render_id": identifier,
        "mode": mode,
        "output_mode": output_mode,
        "output_url": f"/outputs/{output_name}",
        "download_name": f"{source_stem}-{suffix}.mp4",
        "elapsed_seconds": round(time.perf_counter() - started, 3),
        "camera": {key: value for key, value in plan.items() if key != "keyframes"},
        "camera_qa_frames": (
            horizontal_focus_qa(video_path, plan)
            if output_mode == "horizontal-focus"
            else camera_plan_qa(video_path, tracking, plan)
        ),
        "poster": rendered_video_poster(output_path),
        "media": media,
    }


def analyze_video(video_path: Path, start: float, end: float) -> dict:
    if not ai_status()["ready"]:
        DETECTOR._load()

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise AnalysisError(
            "video_unreadable",
            "AI 服务无法读取这个视频。",
            "请优先使用 H.264 编码的 MP4 文件。",
        )
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0)
    frame_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = frame_count / fps if fps > 0 else end
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    if width <= 0 or height <= 0:
        capture.release()
        raise AnalysisError("video_metadata", "无法读取视频尺寸。", "请转换为 MP4 后重试。")

    safe_start = max(0.0, min(float(start), max(duration - 0.05, 0.0)))
    safe_end = max(safe_start + 0.05, min(float(end), duration))
    records = []
    for timestamp in sample_times(safe_start, safe_end):
        capture.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)
        ok, frame = capture.read()
        if not ok:
            continue
        boxes = decorate_performer_roles(
            merge_person_fragments(DETECTOR.detect(frame), width, height),
            width,
            height,
        )
        measurements = frame_measurements(frame, boxes)
        performer_boxes = [box for box in boxes if box.get("role") == "performer"]
        records.append(
            {
                "time": timestamp,
                "frame": frame,
                "boxes": boxes,
                "performer_boxes": performer_boxes,
                **measurements,
            }
        )
    capture.release()

    if not records:
        raise AnalysisError("no_frames", "没有成功抽取候选画面。", "请调整保留区间后重试。")
    if not any(record["boxes"] for record in records):
        raise AnalysisError(
            "no_people",
            "候选画面中没有检测到人物。",
            "请保留人物更清晰、更完整的片段，或更换视频后重试。",
        )

    sharp_values = [math.log1p(record["sharpness_raw"]) for record in records]
    sharp_min = min(sharp_values)
    sharp_span = max(max(sharp_values) - sharp_min, 1e-6)
    max_people = max(len(record["boxes"]) for record in records)
    max_performers = max(len(record["performer_boxes"]) for record in records)
    performer_counts = [len(record["performer_boxes"]) for record in records if record["performer_boxes"]]
    expected_performers = max(1, int(round(float(np.median(performer_counts))))) if performer_counts else 1

    for record, log_sharpness in zip(records, sharp_values):
        count_score = max(
            0.0,
            1.0 - abs(len(record["performer_boxes"]) - expected_performers) / max(expected_performers, 1),
        )
        performer_confidence = (
            sum(box["confidence"] for box in record["performer_boxes"]) / len(record["performer_boxes"])
            if record["performer_boxes"] else record["average_confidence"]
        )
        sharpness_score = (log_sharpness - sharp_min) / sharp_span
        separation_score = max(0.0, 1.0 - min(1.0, record["overlap_penalty"] * 2.2))
        full_body_score = max(0.0, 1.0 - record["cutoff_ratio"])
        score = (
            0.34 * count_score
            + 0.18 * performer_confidence
            + 0.22 * sharpness_score
            + 0.08 * record["brightness_score"]
            + 0.10 * separation_score
            + 0.08 * full_body_score
        )
        if not record["boxes"]:
            score = -1.0
        record["score"] = score
        record["score_parts"] = {
            "people": round(count_score, 4),
            "confidence": round(performer_confidence, 4),
            "sharpness": round(sharpness_score, 4),
            "brightness": round(record["brightness_score"], 4),
            "separation": round(separation_score, 4),
            "full_body": round(full_body_score, 4),
        }

    ranked = sorted(records, key=lambda item: item["score"], reverse=True)
    candidates = []
    for candidate_index, record in enumerate(ranked[:3], 1):
        # Spend the extra detector budget only on the three frames the user can
        # actually choose from.  The initial nine-frame ranking stays fast, while
        # distant/raised-arm performers in the visible choices get tiled recall.
        cached_boxes = [
            {
                "box": box,
                "detector_confidence": float(box.get("confidence", 0.0)),
            }
            for box in record["boxes"]
        ]
        selection_boxes = correction_candidate_boxes(record["frame"], cached_boxes)
        performer_selection_boxes = [
            box for box in selection_boxes if box.get("role") == "performer"
        ]
        candidates.append(
            {
                "id": f"candidate-{candidate_index}",
                "time": round(record["time"], 3),
                "width": width,
                "height": height,
                "score": round(record["score"], 4),
                "score_parts": record["score_parts"],
                "people_count": len(selection_boxes),
                "performer_count": len(performer_selection_boxes),
                "boxes": selection_boxes,
                "detection_mode": "full-frame-ranking + tiled-selection-recall",
                "image": encode_jpeg(record["frame"]),
            }
        )

    return {
        "analysis_id": uuid.uuid4().hex[:12],
        "detector": ai_status(),
        "source": {
            "width": width,
            "height": height,
            "fps": round(fps, 3),
            "duration": round(duration, 3),
            "start": round(safe_start, 3),
            "end": round(safe_end, 3),
            "sample_count": len(records),
        },
        "scoring": {
            "main_performers": 0.34,
            "confidence": 0.18,
            "sharpness": 0.22,
            "brightness": 0.08,
            "separation": 0.10,
            "full_body": 0.08,
        },
        "scene_context": {
            "mode": "fancam-stage",
            "description": "优先识别中央表演区、持续在场且非前景遮挡的主舞者；周边人物仍可点击。",
            "performer_threshold": 0.70,
            "max_performers_seen": max_performers,
            "expected_performers": expected_performers,
            "max_people_seen": max_people,
        },
        "candidates": candidates,
    }


class AppHandler(SimpleHTTPRequestHandler):
    server_version = "OneClickFancam/0.6"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def guess_type(self, path: str) -> str:
        content_type = super().guess_type(path)
        if content_type.startswith("text/") or content_type in {"application/javascript", "application/json"}:
            return f"{content_type}; charset=utf-8"
        return content_type

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/health":
            self.send_json(HTTPStatus.OK, {"service": "一键直拍", **ai_status()})
            return
        super().do_GET()

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/analyze", "/api/track", "/api/render"}:
            self.send_json(HTTPStatus.NOT_FOUND, {"error": {"message": "接口不存在。"}})
            return
        content_length = int(self.headers.get("Content-Length", "0") or 0)
        if content_length <= 0 or content_length > MAX_UPLOAD_BYTES:
            self.send_json(
                HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                {"error": {"message": "视频为空或超过 1.5 GB。", "hint": "请先裁短或压缩视频。"}},
            )
            return

        temporary_path = None
        started = time.perf_counter()
        try:
            form = cgi.FieldStorage(
                fp=self.rfile,
                headers=self.headers,
                environ={
                    "REQUEST_METHOD": "POST",
                    "CONTENT_TYPE": self.headers.get("Content-Type", ""),
                    "CONTENT_LENGTH": str(content_length),
                },
            )
            video_field = form["video"] if "video" in form else None
            if video_field is None or not getattr(video_field, "file", None):
                raise AnalysisError("video_missing", "没有收到视频文件。", "请返回第一步重新选择视频。")
            start = float(form.getfirst("start", "0"))
            end = float(form.getfirst("end", "0"))
            if end <= start:
                raise AnalysisError("range_invalid", "保留区间无效。", "请确保终点晚于起点。")

            suffix = Path(getattr(video_field, "filename", "video.mp4") or "video.mp4").suffix.lower()
            if suffix not in {".mp4", ".mov", ".mkv", ".webm", ".avi"}:
                suffix = ".mp4"
            with tempfile.NamedTemporaryFile(prefix="one-click-fancam-", suffix=suffix, delete=False) as output:
                temporary_path = Path(output.name)
                while True:
                    chunk = video_field.file.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)

            if path == "/api/analyze":
                result = analyze_video(temporary_path, start, end)
            elif path == "/api/track":
                selection = json.loads(form.getfirst("selection", "{}"))
                if not isinstance(selection, dict) or not isinstance(selection.get("box"), dict):
                    raise AnalysisError(
                        "selection_missing",
                        "没有收到已确认的人物锚点。",
                        "请返回人物选择画面，重新点击并确认目标人物。",
                    )
                corrections = json.loads(form.getfirst("corrections", "[]"))
                if not isinstance(corrections, list):
                    raise AnalysisError(
                        "corrections_invalid",
                        "人工纠偏点格式无效。",
                        "请清空暂存的纠偏点后重新选择。",
                    )
                requested_tracker = form.getfirst("tracker_mode", "auto").strip().lower()
                configured_tracker = os.environ.get("ONETAKE_TRACKER_MODE", "lite").strip().lower()
                tracker_mode = configured_tracker if requested_tracker == "auto" else requested_tracker
                if tracker_mode == "sam2":
                    result = track_video_high_precision(
                        temporary_path, start, end, float(selection.get("timestamp", start)),
                        selection["box"], corrections=corrections,
                    )
                else:
                    result = track_video(
                        temporary_path,
                        start,
                        end,
                        float(selection.get("timestamp", start)),
                        selection["box"],
                        reuse_tracking_id=form.getfirst("reuse_tracking_id", ""),
                        corrections=corrections,
                    )
            else:
                tracking = json.loads(form.getfirst("tracking", "{}"))
                if not isinstance(tracking, dict) or not isinstance(tracking.get("keyframes"), list):
                    raise AnalysisError(
                        "tracking_missing",
                        "没有收到已确认的人物轨迹。",
                        "请返回身份跟踪阶段，完成分析后再生成运镜。",
                    )
                result = render_reframed_video(
                    temporary_path,
                    tracking,
                    form.getfirst("framing", "standard"),
                    form.getfirst("motion", "balanced"),
                    form.getfirst("mode", "preview"),
                    getattr(video_field, "filename", "video.mp4") or "video.mp4",
                    form.getfirst("output_mode", "vertical"),
                )
            result["elapsed_seconds"] = round(time.perf_counter() - started, 3)
            self.send_json(HTTPStatus.OK, result)
        except AnalysisError as exc:
            self.send_json(
                HTTPStatus.UNPROCESSABLE_ENTITY,
                {"error": {"code": exc.code, "message": exc.message, "hint": exc.hint}},
            )
        except (ValueError, TypeError):
            self.send_json(
                HTTPStatus.BAD_REQUEST,
                {"error": {"message": "请求参数不完整。", "hint": "请返回第一步重新确认片段。"}},
            )
        except Exception as exc:  # pragma: no cover - diagnostics for local demo
            self.send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": {"message": "本地视频处理失败。", "hint": str(exc)}},
            )
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink(missing_ok=True)
                except OSError:
                    pass


def open_browser(url: str) -> None:
    try:
        webbrowser.open(url)
    except Exception:
        pass


def serve(host: str, port: int, should_open: bool) -> None:
    url = f"http://127.0.0.1:{port}/"
    server = ThreadingHTTPServer((host, port), AppHandler)
    print("=" * 64)
    print(f"一键直拍已启动 · Build {BUILD_ID}")
    print(f"访问地址：{url}")
    print("请保持此窗口开启；关闭窗口会停止 AI 服务。")
    status = ai_status()
    print(f"AI 状态：{'就绪' if status['ready'] else '未就绪'} · {status['detector']}")
    print("=" * 64)
    if should_open:
        threading.Timer(0.8, open_browser, args=(url,)).start()
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n一键直拍已停止。")
    finally:
        server.server_close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=4173)
    parser.add_argument("--open", action="store_true", dest="should_open")
    parser.add_argument("--analyze", type=Path, help="Run a local analysis smoke test")
    parser.add_argument("--track", type=Path, help="Run a local identity-tracking smoke test")
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=4.0)
    parser.add_argument("--selection-time", type=float, default=2.0)
    parser.add_argument("--box", help="Tracking seed box as x1,y1,x2,y2")
    args = parser.parse_args()

    if args.analyze:
        result = analyze_video(args.analyze.resolve(), args.start, args.end)
        summary = {
            "analysis_id": result["analysis_id"],
            "source": result["source"],
            "candidates": [
                {
                    "time": candidate["time"],
                    "people_count": candidate["people_count"],
                    "score": candidate["score"],
                    "confidences": [box["confidence"] for box in candidate["boxes"]],
                }
                for candidate in result["candidates"]
            ],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    if args.track:
        if not args.box:
            raise SystemExit("--track requires --box x1,y1,x2,y2")
        values = [float(value.strip()) for value in args.box.split(",")]
        if len(values) != 4:
            raise SystemExit("--box must contain x1,y1,x2,y2")
        result = track_video(
            args.track.resolve(),
            args.start,
            args.end,
            args.selection_time,
            dict(zip(("x1", "y1", "x2", "y2"), values)),
        )
        summary = {key: value for key, value in result.items() if key not in {"qa_frames", "keyframes"}}
        summary["keyframe_preview"] = result["keyframes"][:3] + result["keyframes"][-3:]
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    serve(args.host, args.port, args.should_open)


if __name__ == "__main__":
    main()
