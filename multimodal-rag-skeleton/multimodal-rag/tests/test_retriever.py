"""
Sanity test for the retrieval layer using a tiny synthetic index, so it
doesn't depend on real PDFs or the CLIP/MiniLM downloads finishing first.
Run with: pytest tests/test_retriever.py
"""
import numpy as np

from app.indexing.vector_store import VectorStore
from app.models.schemas import Chunk, EmbeddedChunk


def test_search_text_returns_nearest_by_vector():
    store = VectorStore(text_dim=4, image_dim=4)

    chunks = [
        Chunk("c1", "doc1", 1, "text", "the sky is blue"),
        Chunk("c2", "doc1", 2, "text", "cats are mammals"),
    ]
    vectors = [
        np.array([1.0, 0.0, 0.0, 0.0], dtype="float32"),
        np.array([0.0, 1.0, 0.0, 0.0], dtype="float32"),
    ]
    store.add([EmbeddedChunk(chunk=c, vector=v) for c, v in zip(chunks, vectors)])

    query = np.array([0.9, 0.1, 0.0, 0.0], dtype="float32")
    results = store.search_text(query, k=1)

    assert len(results) == 1
    assert results[0][0].chunk_id == "c1"
