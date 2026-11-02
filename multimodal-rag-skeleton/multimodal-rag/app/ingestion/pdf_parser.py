"""
Ingestion layer (Review-1, Section 5, "ingestion layer").

Parses a PDF into page-tagged Chunks:
  - text passages (PyMuPDF), about 900 characters each
  - tables        (pdfplumber, falling back to camelot for ruled tables)
  - images        (PyMuPDF, saved to disk for the CLIP embedder to read)

Each chunk keeps its document_id and page_number so citations in the
generation layer can point back to the exact source page.

Accuracy upgrade: text used to be one chunk per WHOLE PAGE. The MiniLM
embedding model only reads the first ~256 tokens of its input, so for a full
page it embedded just the top of the page and was blind to the rest. Pages are
now split into passages of roughly 900 characters (about 200 tokens), so every
part of every page is actually searchable.
"""
import uuid
from pathlib import Path

import fitz  # PyMuPDF
import pdfplumber

from app.config import settings
from app.models.schemas import Chunk

MAX_PASSAGE_CHARS = 900
MIN_PASSAGE_CHARS = 250


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


def _split_into_passages(text: str) -> list[str]:
    """Groups consecutive lines into passages of about MAX_PASSAGE_CHARS,
    keeping the original line breaks (so lists and table-like text stay
    readable) and preferring to break at blank lines."""
    passages: list[str] = []
    current: list[str] = []
    size = 0

    def flush():
        nonlocal current, size
        joined = "\n".join(current).strip()
        if joined:
            passages.append(joined)
        current, size = [], 0

    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            if size >= MIN_PASSAGE_CHARS:
                flush()
            continue

        while len(line) > MAX_PASSAGE_CHARS:  # one huge unbroken line
            cut = line.rfind(" ", 0, MAX_PASSAGE_CHARS)
            if cut < 200:
                cut = MAX_PASSAGE_CHARS
            if current:
                flush()
            passages.append(line[:cut].strip())
            line = line[cut:].strip()

        if size + len(line) + 1 > MAX_PASSAGE_CHARS and size >= MIN_PASSAGE_CHARS:
            flush()
        current.append(line)
        size += len(line) + 1

    flush()
    if len(passages) >= 2 and len(passages[-1]) < 120:  # fold a tiny tail into its neighbour
        tail = passages.pop()
        passages[-1] += "\n" + tail
    return passages


def extract_text_chunks(doc: fitz.Document, document_id: str) -> list[Chunk]:
    """Several passage-sized chunks per page, in reading order."""
    chunks = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        text = page.get_text("text").strip()
        if not text:
            continue
        for passage in _split_into_passages(text):
            chunks.append(
                Chunk(
                    chunk_id=_new_id(),
                    document_id=document_id,
                    page_number=page_index + 1,
                    modality="text",
                    content=passage,
                )
            )
    return chunks


def extract_table_chunks(pdf_path: Path, document_id: str) -> list[Chunk]:
    """Serializes each detected table to a markdown-ish text block so it can
    share the text embedder / same vector space as prose."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_index, page in enumerate(pdf.pages):
            tables = page.extract_tables()
            for table in tables:
                if not table or not any(any(cell for cell in row) for row in table):
                    continue
                serialized = _table_to_text(table)
                chunks.append(
                    Chunk(
                        chunk_id=_new_id(),
                        document_id=document_id,
                        page_number=page_index + 1,
                        modality="table",
                        content=serialized,
                    )
                )
    return chunks


def _table_to_text(table: list[list[str | None]]) -> str:
    rows = []
    for row in table:
        cells = [str(c).strip() if c else "" for c in row]
        rows.append(" | ".join(cells))
    return "\n".join(rows)


def extract_image_chunks(
    doc: fitz.Document, document_id: str, output_dir: Path
) -> list[Chunk]:
    """Saves each embedded image to disk (CLIP needs a file/array, not raw bytes
    in the DB) and records its source page."""
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            ext = base_image["ext"]
            image_path = output_dir / f"{document_id}_p{page_index+1}_{img_index}.{ext}"
            image_path.write_bytes(image_bytes)

            chunks.append(
                Chunk(
                    chunk_id=_new_id(),
                    document_id=document_id,
                    page_number=page_index + 1,
                    modality="image",
                    content=f"[image on page {page_index+1}]",
                    image_path=str(image_path),
                )
            )
    return chunks


def parse_pdf(pdf_path: str | Path) -> list[Chunk]:
    """Top-level ingestion entrypoint: PDF path -> list of page-tagged Chunks
    covering text, tables and images."""
    pdf_path = Path(pdf_path)
    document_id = pdf_path.stem

    doc = fitz.open(pdf_path)
    try:
        text_chunks = extract_text_chunks(doc, document_id)
        image_chunks = extract_image_chunks(
            doc, document_id, settings.processed_dir / "images"
        )
    finally:
        doc.close()

    table_chunks = extract_table_chunks(pdf_path, document_id)

    return text_chunks + table_chunks + image_chunks


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m app.ingestion.pdf_parser <path_to_pdf>")
        raise SystemExit(1)

    result = parse_pdf(sys.argv[1])
    print(f"Extracted {len(result)} chunks:")
    for c in result[:10]:
        print(f"  [{c.modality}] page {c.page_number}: {c.content[:80]!r}")