"""
Ingests every PDF in data/raw_pdfs/ (your 15-20 custom-corpus documents,
Review-1 Section 7) into the shared vector store and persists it to
data/indexes/.

Usage:
    python scripts/build_index.py
"""
from tqdm import tqdm

from app.config import settings
from app.embeddings.image_embedder import embed_image
from app.embeddings.text_embedder import embed_text
from app.indexing.vector_store import VectorStore
from app.ingestion.pdf_parser import parse_pdf
from app.models.schemas import EmbeddedChunk
from app.utils.logging import get_logger

logger = get_logger(__name__)


def build_index() -> VectorStore:
    pdf_paths = sorted(settings.raw_pdf_dir.glob("*.pdf"))
    if not pdf_paths:
        logger.warning(
            f"No PDFs found in {settings.raw_pdf_dir}. "
            "Add your custom corpus there before running this script."
        )

    store = VectorStore()

    for pdf_path in pdf_paths:
        logger.info(f"Parsing {pdf_path.name}...")
        chunks = parse_pdf(pdf_path)

        embedded = []
        for chunk in tqdm(chunks, desc=f"Embedding {pdf_path.name}"):
            if chunk.modality == "image" and chunk.image_path:
                vec = embed_image(chunk.image_path)
            else:
                vec = embed_text(chunk.content)
            embedded.append(EmbeddedChunk(chunk=chunk, vector=vec))

        store.add(embedded)
        logger.info(
            f"  -> {len(chunks)} chunks "
            f"({sum(1 for c in chunks if c.modality=='text')} text, "
            f"{sum(1 for c in chunks if c.modality=='table')} table, "
            f"{sum(1 for c in chunks if c.modality=='image')} image)"
        )

    store.save(settings.index_dir)
    logger.info(f"Index saved to {settings.index_dir}")
    return store


if __name__ == "__main__":
    build_index()
