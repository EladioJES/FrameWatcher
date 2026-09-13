"""Detection and annotation of single video frames (no GUI).

Model and detection settings mirror ~/projekts/vision/app/detect.py, but run
model.predict on isolated frames: no tracking IDs and no crossing counters.
"""
from pathlib import Path

import cv2
from ultralytics import YOLO

MODELS_DIR = Path("~/projekts/vision/app").expanduser()
DEFAULT_MODEL = "weights_ncnn_model"  # the model detect.py loads
CROP_SIZES = [(1280, 720), (800, 600), (640, 480), (320, 240)]
JPEG_QUALITY = 95


def available_models():
    return sorted(p.name for p in MODELS_DIR.glob("*_ncnn_model") if p.is_dir())


def load_model(name=DEFAULT_MODEL):
    return YOLO(MODELS_DIR / name, task="detect")


def put_text(img, text, org, scale, color, thickness):
    # dark outline keeps text readable on the bright floor, especially on small crops
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def read_frame(cap, idx):
    """Seek to frame idx and decode it. Returns None if the frame can't be read."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ret, frame = cap.read()
    return frame if ret else None


def center_crop(frame, w, h):
    H, W = frame.shape[:2]
    if w > W or h > H:
        raise ValueError(f"crop {w}x{h} larger than frame {W}x{H}")
    x0 = (W - w) // 2
    y0 = (H - h) // 2
    return frame[y0:y0 + h, x0:x0 + w].copy()


def annotate(model, frame, frame_idx=None):
    results = model.predict(frame, classes=[0], conf=0.1, iou=0.5, verbose=False)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    mid_y = h // 2
    cv2.line(annotated, (0, mid_y), (w, mid_y), (0, 255, 0), 2)

    in_upper = 0
    in_lower = 0
    for box in results[0].boxes.xyxy:
        center_y = float((box[1] + box[3]) / 2)
        if center_y < mid_y:
            in_upper += 1
        else:
            in_lower += 1

    # detect.py uses scale 0.7 on 640-wide frames; shrink on narrow crops so text fits
    scale = max(0.45, 0.7 * min(1.0, w / 640))
    thickness = 2 if scale >= 0.6 else 1
    put_text(annotated, f"In upper: {in_upper}", (10, int(40 * scale / 0.7) + 10),
             scale, (0, 255, 0), thickness)
    put_text(annotated, f"In lower: {in_lower}", (10, h - 20), scale, (0, 255, 0), thickness)

    if frame_idx is not None:
        label = f"frame {frame_idx}"
        (tw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        put_text(annotated, label, (w - tw - 10, h - 20), scale, (255, 255, 255), thickness)
    return annotated


def output_path(out_dir, video_path, idx, size, full_size):
    """<out>/<stem>/<stem>_f000123.jpg for the full frame, else ..._f000123_WxH.jpg."""
    stem = Path(video_path).stem
    name = f"{stem}_f{idx:06d}"
    if tuple(size) != tuple(full_size):
        name += f"_{size[0]}x{size[1]}"
    return Path(out_dir) / stem / f"{name}.jpg"


def save_jpg(path, img):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]):
        raise OSError(f"failed to write {path}")
