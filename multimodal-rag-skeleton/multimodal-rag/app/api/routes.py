"""
Serving layer (Review-1 Section 5, "FastAPI service ... upload document,
question submission, answer retrieval"). Keeps a single in-memory
VectorStore for the demo; swap `get_store` for a DB/disk-backed singleton
if you need persistence across restarts (build_index.py already persists
to data/indexes/, so on startup you can VectorStore.load() that path).

Review-2 upgrade: /upload now also rasterizes every page (page_renderer.py)
and runs OpenCV region detection (region_detection.py) on each page image,
indexing each detected figure/chart/diagram region as its own retrievable
chunk with a bounding box. Two new endpoints serve page images — plain and
with a highlight box drawn on — for the frontend's visual grounding and
narrated walkthrough features.
"""
import logging
import re
import shutil
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from app.config import settings
from app.embeddings.image_embedder import embed_image
from app.embeddings.text_embedder import embed_text
from app.generation.structural_answers import try_structural_answer
from app.generation.vlm_generator import generate_answer
from app.indexing.vector_store import VectorStore
from app.ingestion.page_renderer import get_page_image_path, render_pdf_pages
from app.ingestion.pdf_parser import parse_pdf
from app.models.schemas import AskRequest, AskResponse, Chunk, EmbeddedChunk, UploadResponse
from app.retrieval.retriever import retrieve
from app.vision.region_detection import crop_region, detect_regions, draw_highlight

router = APIRouter()
log = logging.getLogger("uvicorn.error")  # shows full tracebacks in the server terminal

# --- Simple process-wide store; see docstring for persistence notes. ---
_store: VectorStore | None = None

MAX_REGIONS_PER_PAGE = 6  # keeps only the biggest figure/chart regions; tiny ones are mostly noise
REGION_CROP_DIR = Path("data/region_crops")
HIGHLIGHT_CACHE_DIR = Path("data/highlight_cache")


def get_store() -> VectorStore:
    global _store
    if _store is None:
        index_path = settings.index_dir
        if (index_path / "metadata.pkl").exists():
            _store = VectorStore.load(index_path)
        else:
            _store = VectorStore()
    return _store


def _friendly_document_id(original_filename: str, store: VectorStore) -> str:
    """Derives a readable document ID from the uploaded filename (e.g.
    'Q3 Report.pdf' -> 'Q3_Report') instead of the random temp-file name
    the upload was briefly saved under. Appends a numeric suffix on
    collision so re-uploading a same-named file doesn't overwrite the
    existing one in the index."""
    stem = Path(original_filename).stem
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", stem).strip("_") or "document"

    existing_ids = {c.document_id for c in (store.text_meta + store.image_meta)}
    if safe not in existing_ids:
        return safe

    counter = 2
    while f"{safe}_{counter}" in existing_ids:
        counter += 1
    return f"{safe}_{counter}"


def _index_visual_regions(document_id: str, page_paths: dict[int, str]) -> list[EmbeddedChunk]:
    """Runs OpenCV region detection on every rendered page image and embeds
    each detected region (figure/chart/diagram) as its own Chunk, tagged
    with its bounding box. This is what gives citations page-region
    precision instead of whole-page precision, and what the narrated
    walkthrough pans/zooms toward."""
    embedded: list[EmbeddedChunk] = []

    for page_number, page_path in page_paths.items():
        regions = detect_regions(page_path)
        for bbox in regions[:MAX_REGIONS_PER_PAGE]:  # largest regions first
            crop_filename = f"{document_id}_p{page_number}_{uuid.uuid4().hex[:8]}.png"
            crop_path = REGION_CROP_DIR / document_id / crop_filename
            crop_region(page_path, bbox, str(crop_path))

            chunk = Chunk(
                chunk_id=f"{document_id}-region-{uuid.uuid4().hex[:8]}",
                document_id=document_id,
                page_number=page_number,
                modality="image",
                content="",
                image_path=str(crop_path),
                bbox=bbox,
            )
            vec = embed_image(chunk.image_path)
            embedded.append(EmbeddedChunk(chunk=chunk, vector=vec))

    return embedded


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/documents")
def list_documents():
    """Returns every document currently in the FAISS store (built via
    build_index.py or /upload), so the frontend can let the user scope a
    question to one specific document instead of searching all of them."""
    store = get_store()
    all_chunks = store.text_meta + store.image_meta

    by_doc: dict[str, dict] = {}
    for chunk in all_chunks:
        entry = by_doc.setdefault(
            chunk.document_id,
            {"document_id": chunk.document_id, "page_count": 0, "chunk_count": 0},
        )
        entry["page_count"] = max(entry["page_count"], chunk.page_number)
        entry["chunk_count"] += 1

    return {"documents": sorted(by_doc.values(), key=lambda d: d["document_id"])}


@router.get("/documents/{document_id}/pages/{page_number}")
def get_page_image(document_id: str, page_number: int):
    """Serves the full rasterized page image, used by the frontend to
    display the document alongside an answer and to draw highlight
    overlays client-side."""
    path = get_page_image_path(document_id, page_number)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found.")
    return FileResponse(str(path), media_type="image/png")


@router.get("/documents/{document_id}/pages/{page_number}/highlight")
def get_highlighted_page_image(
    document_id: str,
    page_number: int,
    x: int = Query(...),
    y: int = Query(...),
    w: int = Query(...),
    h: int = Query(...),
):
    """Serves the page image with a highlight box drawn server-side at the
    given bbox — an alternative to client-side canvas overlay, useful for
    a plain <img> based 'show me where' view or for generating walkthrough
    frames ahead of time."""
    path = get_page_image_path(document_id, page_number)
    if path is None:
        raise HTTPException(status_code=404, detail="Page image not found.")

    out_path = HIGHLIGHT_CACHE_DIR / document_id / f"page_{page_number}_{x}_{y}_{w}_{h}.png"
    if not out_path.exists():
        draw_highlight(str(path), (x, y, w, h), str(out_path))
    return FileResponse(str(out_path), media_type="image/png")


@router.get("/debug/document/{document_id}")
def debug_document(document_id: str):
    """Shows what was actually extracted from an uploaded PDF. If 'text_chars'
    is tiny, the PDF is scanned (pictures of text) and has nothing to search."""
    store = get_store()
    chunks = [c for c in (store.text_meta + store.image_meta) if c.document_id == document_id]
    if not chunks:
        raise HTTPException(status_code=404, detail="No such document in the index.")
    text_chunks = [c for c in chunks if c.modality == "text"]
    return {
        "document_id": document_id,
        "text_chunks": len(text_chunks),
        "table_chunks": sum(1 for c in chunks if c.modality == "table"),
        "image_chunks": sum(1 for c in chunks if c.modality == "image"),
        "text_chars": sum(len(c.content) for c in text_chunks),
        "avg_chunk_chars": round(sum(len(c.content) for c in text_chunks) / max(1, len(text_chunks))),
        "first_text_samples": [c.content[:300] for c in text_chunks[:3]],
    }


@router.get("/debug/retrieval")
def debug_retrieval(question: str, document_id: str | None = None, top_k: int = 6):
    """Shows exactly which chunks would be sent to Gemini for a question."""
    evidence = retrieve(get_store(), question, k=top_k, document_id=document_id)
    return {
        "question": question,
        "mode": evidence[0].mode if evidence else None,
        "chunks_sent": len(evidence),
        "documents": sorted({e.chunk.document_id for e in evidence}),
        "chunks": [
            {
                "n": i + 1,
                "document": e.chunk.document_id,
                "page": e.chunk.page_number,
                "type": e.chunk.modality,
                "has_region_box": e.chunk.bbox is not None,
                "chars": len(e.chunk.content),
                "preview": e.chunk.content[:200],
            }
            for i, e in enumerate(evidence)
        ],
    }


@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = Path(tmp.name)

    try:
        try:
            chunks = parse_pdf(tmp_path)
        except Exception as exc:
            log.exception("PDF parsing failed")
            raise HTTPException(
                status_code=422,
                detail=f"Could not read this PDF ({type(exc).__name__}: {exc}). It may be damaged or password-protected.",
            )
        store = get_store()

        # Replace the random temp-file document_id with one derived from the
        # actual uploaded filename, so the UI shows something readable
        # instead of e.g. "tmpak3jjmin".
        document_id = _friendly_document_id(file.filename, store)
        for chunk in chunks:
            chunk.document_id = document_id

        # Review-2: rasterize every page, then run OpenCV region detection
        # on each page image for the visual-grounding/walkthrough feature.
        # Done while tmp_path still exists, since render_pdf_pages reads
        # the original PDF. This part is optional: if it fails, the upload
        # still succeeds with text/table/image search, just without regions.
        try:
            page_paths = render_pdf_pages(tmp_path, document_id)
        except Exception:
            log.exception("Page rendering failed; continuing without page images")
            page_paths = {}
    finally:
        tmp_path.unlink(missing_ok=True)

    try:
        embedded = []
        skipped_images = 0
        for chunk in chunks:
            if chunk.modality == "image" and chunk.image_path:
                try:
                    vec = embed_image(chunk.image_path)
                except Exception:
                    # Some PDF images (odd colour spaces, masks, tiny icons) can't be
                    # opened by CLIP. Skip that one image instead of failing the upload.
                    log.exception("Skipping an image that could not be embedded: %s", chunk.image_path)
                    skipped_images += 1
                    continue
            else:
                vec = embed_text(chunk.content)
            embedded.append(EmbeddedChunk(chunk=chunk, vector=vec))
        chunks = [c for c in chunks if c.modality != "image" or any(e.chunk is c for e in embedded)]

        try:
            region_embedded = _index_visual_regions(document_id, page_paths)
        except Exception:
            log.exception("Visual region detection failed; continuing without regions")
            region_embedded = []
        embedded.extend(region_embedded)

        if not embedded:
            raise HTTPException(
                status_code=422,
                detail="No searchable content was found in this PDF. If it is a scan (pictures of text), it needs OCR first.",
            )

        store.add(embedded)
        store.save(settings.index_dir)
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("Upload indexing failed")
        raise HTTPException(status_code=500, detail=f"Upload failed ({type(exc).__name__}): {exc}")

    return UploadResponse(
        document_id=document_id,
        filename=file.filename,
        page_count=len({c.page_number for c in chunks}) or len(page_paths),
        text_chunks=sum(1 for c in chunks if c.modality == "text"),
        table_chunks=sum(1 for c in chunks if c.modality == "table"),
        image_chunks=sum(1 for c in chunks if c.modality == "image"),
        regions_detected=len(region_embedded),
    )


@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    store = get_store()

    # Counting questions ("how many images/tables/pages") are answered
    # directly from index metadata — exact and guaranteed correct, rather
    # than trusting the VLM to count from retrieved chunks.
    structural = try_structural_answer(request.question, store, request.document_id)
    if structural is not None:
        return structural

    evidence = retrieve(
        store, request.question, k=request.top_k, document_id=request.document_id
    )
    try:
        return generate_answer(request.question, evidence)
    except Exception as exc:
        # Surface the real cause instead of a bare 500 — most commonly a
        # missing/invalid API key, no credit/quota on the account, or a
        # model name your key doesn't have access to.
        raise HTTPException(
            status_code=502,
            detail=f"Generation failed ({type(exc).__name__}): {exc}",
        )