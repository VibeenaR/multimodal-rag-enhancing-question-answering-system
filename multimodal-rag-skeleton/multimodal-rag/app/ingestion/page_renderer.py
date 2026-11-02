"""
Rasterizes every page of an uploaded PDF to a PNG file (Review-2 upgrade:
visual grounding). Kept deliberately separate from pdf_parser.py's chunk
extraction — this renders EVERY page, regardless of what text/table/image
chunks were found on it, because the highlight/walkthrough feature needs a
full-page image to draw a bounding box on, even for pages that are pure
text with no extracted figures.

Save this file as: app/ingestion/page_renderer.py
(goes alongside your existing pdf_parser.py)
"""
from pathlib import Path

import fitz  # PyMuPDF — already a dependency via pdf_parser.py

PAGE_IMAGE_DIR = Path("data/page_images")
RENDER_DPI = 150  # good balance of legibility vs. file size/render time


def render_pdf_pages(pdf_path: Path, document_id: str) -> dict[int, str]:
    """Rasterizes every page of the given PDF to a PNG, saved under
    data/page_images/<document_id>/page_<n>.png. Returns a dict mapping
    1-indexed page number -> saved file path (matching the page_number
    convention already used by Chunk elsewhere in this project)."""
    out_dir = PAGE_IMAGE_DIR / document_id
    out_dir.mkdir(parents=True, exist_ok=True)

    zoom = RENDER_DPI / 72  # PDF points are 72 per inch
    matrix = fitz.Matrix(zoom, zoom)

    page_paths: dict[int, str] = {}
    doc = fitz.open(pdf_path)
    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            pix = page.get_pixmap(matrix=matrix)
            page_number = page_index + 1
            out_path = out_dir / f"page_{page_number}.png"
            pix.save(str(out_path))
            page_paths[page_number] = str(out_path)
    finally:
        doc.close()

    return page_paths


def get_page_image_path(document_id: str, page_number: int) -> Path | None:
    """Looks up the already-rendered page image for a document/page, or
    None if it hasn't been rendered (e.g. the document predates this
    feature and hasn't been re-uploaded)."""
    path = PAGE_IMAGE_DIR / document_id / f"page_{page_number}.png"
    return path if path.exists() else None