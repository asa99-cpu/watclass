import cv2
import numpy as np
import pytesseract
import re

# -----------------------------
# BAR COLOR MASK
# -----------------------------
def bars_color_mask(hsv):
    yellow = cv2.inRange(hsv, (20, 60, 120), (45, 255, 255))
    orange = cv2.inRange(hsv, (8, 80, 120), (20, 255, 255))
    red1   = cv2.inRange(hsv, (0, 100, 110), (10, 255, 255))
    red2   = cv2.inRange(hsv, (170, 100, 110), (179, 255, 255))
    green  = cv2.inRange(hsv, (40, 40, 80), (90, 255, 255))
    return yellow | orange | red1 | red2 | green

# -----------------------------
# FIND ROI AROUND BARS
# -----------------------------
def roi_from_mask(mask):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    pad = 2
    x1, x2 = max(0, xs.min()-pad), min(mask.shape[1]-1, xs.max()+pad)
    y1, y2 = max(0, ys.min()-pad), min(mask.shape[0]-1, ys.max()+pad)
    return x1, y1, x2, y2

# -----------------------------
# SPLIT BARS LEFT→RIGHT
# -----------------------------
def split_bars(roi_mask):
    H, W = roi_mask.shape
    colsum = roi_mask.sum(axis=0)
    thr = max(1, int(0.05 * H * 255))
    binary = (colsum >= thr).astype(np.uint8)

    row = (binary * 255).astype(np.uint8)[None, :]
    kernel1d = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, W//200), 1))
    row = cv2.morphologyEx(row, cv2.MORPH_CLOSE, kernel1d, iterations=1)
    binary = (row[0] > 0).astype(np.uint8)

    bars = []
    in_run = False
    for i, v in enumerate(binary):
        if v and not in_run:
            start = i; in_run = True
        if not v and in_run:
            bars.append((start, i-1)); in_run = False
    if in_run:
        bars.append((start, W-1))
    return bars

# -----------------------------
# ZERO LINE = MODE OF BAR BOTTOMS
# -----------------------------
def zero_line_from_bars(roi_mask, bars):
    bottoms = []
    for sx, ex in bars:
        sub = roi_mask[:, sx:ex+1]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) > 0:
            bottoms.append(rows.max())
    if not bottoms:
        return None
    counts = {}
    for b in bottoms: counts[b] = counts.get(b, 0) + 1
    return max(counts, key=counts.get)

# -----------------------------
# OCR RIGHT Y-AXIS LABELS
# -----------------------------
def ocr_y_ticks(rgb):
    H, W, _ = rgb.shape
    right = rgb[:, int(W*0.80):W, :].copy()

    gray = cv2.cvtColor(right, cv2.COLOR_RGB2GRAY)
    bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 3))
    bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker, iterations=1)

    config = "--oem 3 --psm 6 -c tessedit_char_whitelist=0123456789"
    data = pytesseract.image_to_data(bw, output_type=pytesseract.Output.DICT, config=config)

    ticks = []
    for i in range(len(data["text"])):
        txt = data["text"][i]
        if not txt or not re.match(r"^\d+$", txt):
            continue
        val = int(txt)
        y, h = data["top"][i], data["height"][i]
        y_center = y + h//2
        y_snap = snap_to_gridline(rgb, y_center, search_half=6)
        ticks.append((val, y_snap if y_snap is not None else y_center))

    out = {}
    for v, y in ticks:
        out.setdefault(v, []).append(y)
    ticks_clean = [(v, int(np.median(out[v]))) for v in out.keys()]
    ticks_clean.sort(key=lambda t: t[0])
    return ticks_clean

def snap_to_gridline(rgb, y_guess, search_half=6):
    H, W, _ = rgb.shape
    y0 = max(0, y_guess - search_half)
    y1 = min(H-1, y_guess + search_half)
    crop = rgb[y0:y1+1, :, :]
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_RGB2GRAY)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    energy = np.abs(gy).mean(axis=1)
    rel = int(np.argmax(energy))
    return y0 + rel

# -----------------------------
# FIT AXIS MAPPING
# -----------------------------
def fit_axis_mapping_from_ticks(ticks):
    if len(ticks) < 2:
        return None
    vals = np.array([v for v, _ in ticks], dtype=np.float32)
    ys   = np.array([y for _, y in ticks], dtype=np.float32)
    A = np.vstack([np.ones_like(vals), vals]).T
    x, _, _, _ = np.linalg.lstsq(A, ys, rcond=None)
    a, b = x[0], x[1]
    return float(a), float(b)

def value_from_y(y, a, b):
    return (y - a) / b

# -----------------------------
# MAIN DETECTOR
# -----------------------------
def detect_and_measure(image):
    img = np.array(image)
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)

    bmask = bars_color_mask(hsv)
    H, W = bmask.shape
    vker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(3, H//120)))
    bmask = cv2.morphologyEx(bmask, cv2.MORPH_CLOSE, vker, iterations=2)

    bounds = roi_from_mask(bmask)
    if bounds is None:
        return None, None
    x1, y1, x2, y2 = bounds
    roi_mask = bmask[y1:y2+1, x1:x2+1]

    bars = split_bars(roi_mask)
    if not bars:
        return None, None

    zero_rel = zero_line_from_bars(roi_mask, bars)
    if zero_rel is None:
        return None, None
    zero_abs = y1 + zero_rel

    ticks = ocr_y_ticks(img)
    ticks_use, has_zero = [], False
    for v, y in ticks:
        if v == 0:
            has_zero = True
            ticks_use.append((0, zero_abs))
        elif y < zero_abs:
            ticks_use.append((v, y))
    if not has_zero:
        ticks_use.append((0, zero_abs))
    ticks_use.sort(key=lambda t: t[0])

    fit = fit_axis_mapping_from_ticks(ticks_use)
    if fit is None:
        return None, None
    a, b = fit

    vis = img.copy()
    values, rects = [], []
    for sx, ex in bars:
        sub = roi_mask[:, sx:ex+1]
        rows = np.where(sub.any(axis=1))[0]
        if len(rows) == 0:
            continue
        top_rel = rows.min()
        top_abs = y1 + top_rel
        v = value_from_y(top_abs, a, b)
        v = max(0.0, round(float(v), 2))
        values.append(v)
        rects.append((x1+sx, top_abs, x1+ex, zero_abs))

    for (xL, yT, xR, yB), v in zip(rects, values):
        cv2.rectangle(vis, (xL, yT), (xR, yB), (128, 0, 255), 2)
        cv2.putText(vis, f"{v}", (xL, max(0, yT-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (128, 0, 255), 1, cv2.LINE_AA)

    for val, y in ticks_use:
        color = (0, 190, 0) if val == 0 else (0, 160, 0)
        cv2.line(vis, (x1, int(y)), (x2, int(y)), color, 1 + (1 if val == 0 else 0))

    return values, vis
