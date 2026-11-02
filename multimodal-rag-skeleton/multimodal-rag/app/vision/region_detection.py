"""
OpenCV-based visual region detection (Review-2 upgrade: visual grounding).

Instead of indexing one embedding per whole page image, this detects
distinct visual regions on a rendered page (figures, charts, diagrams,
table-like blocks) so each can be embedded and retrieved individually, with
a precise pixel bounding box the frontend can zoom to and highlight. This is
the piece that lets a citation say "this exact region of page 4" instead of
just "page 4" — and it's what the narrated walkthrough pans/zooms toward.

Save this file as: app/vision/region_detection.py
(create the app/vision/ folder and an empty __init__.py in it)
"""
from pathlib import Path

import cv2

MIN_REGION_AREA_FRACTION = 0.01   # ignore specks smaller than 1% of the page
MAX_REGION_AREA_FRACTION = 0.85   # ignore near-full-page false positives
MIN_ASPECT_RATIO = 0.15           # drop extremely thin slivers (stray lines/rules)

BBox = tuple[int, int, int, int]  # (x, y, width, height) in pixels


def detect_regions(image_path: str) -> list[BBox]:
    """Detects candidate figure/chart/diagram regions on a rasterized page
    image. Returns a list of (x, y, w, h) bounding boxes in the original
    image's pixel coordinates, largest-area first.

    Approach: Canny edge detection, then a morphological closing to merge
    nearby edges (e.g. a chart's separate axis lines and bars) into solid
    blobs, then external-contour detection to find bounding rectangles.
    This is a heuristic, not a trained detector — it needs no training data
    and works well on the clean, rendered PDF pages typical of this
    project, in exchange for occasionally catching a false-positive region
    or missing a very low-contrast figure."""
    img = cv2.imread(image_path)
    if img is None:
        return []

    height, width = img.shape[:2]
    page_area = height * width

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    regions: list[BBox] = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area_fraction = (w * h) / page_area
        if area_fraction < MIN_REGION_AREA_FRACTION or area_fraction > MAX_REGION_AREA_FRACTION:
            continue
        aspect_ratio = min(w, h) / max(w, h)
        if aspect_ratio < MIN_ASPECT_RATIO:
            continue
        regions.append((x, y, w, h))

    regions.sort(key=lambda r: r[2] * r[3], reverse=True)
    return regions


def crop_region(image_path: str, bbox: BBox, output_path: str) -> str:
    """Crops one detected region out of the full page image and saves it as
    its own file, so it can be embedded (via CLIP) and indexed independently
    of the full page it came from."""
    img = cv2.imread(image_path)
    x, y, w, h = bbox
    crop = img[y:y + h, x:x + w]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, crop)
    return output_path


def draw_highlight(image_path: str, bbox: BBox, output_path: str) -> str:
    """Draws a highlight rectangle on a COPY of the full page image (never
    mutates the original) — used by the 'show me where' citation feature
    and the narrated walkthrough's pan/highlight frames."""
    img = cv2.imread(image_path)
    x, y, w, h = bbox
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 140, 255), thickness=4)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
    return output_path