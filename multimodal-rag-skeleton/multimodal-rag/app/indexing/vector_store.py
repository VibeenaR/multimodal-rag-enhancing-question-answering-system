"""
Indexing layer (Review-1 Section 5, "stored together in a single vector
database called FAISS"). See the note in image_embedder.py for why this is
implemented as two FAISS indexes (text+table, image) sharing one retrieval
API rather than one literal index — that's the accurate version of "unified
retrieval regardless of modality" the class diagram's VectorIndex class
describes.
"""
import pickle
from dataclasses import asdict
from pathlib import Path

import faiss
import numpy as np

from app.config import settings
from app.models.schemas import Chunk, EmbeddedChunk


class VectorStore:
    def __init__(self, text_dim: int = None, image_dim: int = None):
        self.text_dim = text_dim or settings.text_embedding_dim
        self.image_dim = image_dim or settings.image_embedding_dim

        self.text_index = self._new_index(self.text_dim)
        self.image_index = self._new_index(self.image_dim)

        # Parallel metadata lists — position i in text_meta corresponds to
        # vector i in text_index, same for image.
        self.text_meta: list[Chunk] = []
        self.image_meta: list[Chunk] = []

    def _new_index(self, dim: int) -> faiss.Index:
        if settings.faiss_index_type == "HNSW":
            index = faiss.IndexHNSWFlat(dim, settings.hnsw_m)
            index.hnsw.efConstruction = 200
            return index
        # IVF_PQ needs training data before use; simplest robust default
        # for a corpus this size is HNSW, but this branch is left in per
        # the architecture diagram's "HNSW/IVF-PQ" note.
        quantizer = faiss.IndexFlatL2(dim)
        return faiss.IndexIVFPQ(quantizer, dim, 100, 8, 8)

    def add(self, embedded_chunks: list[EmbeddedChunk]):
        text_vecs, image_vecs = [], []
        for ec in embedded_chunks:
            if ec.chunk.modality == "image":
                image_vecs.append(ec.vector)
                self.image_meta.append(ec.chunk)
            else:  # "text" or "table" share the MiniLM space
                text_vecs.append(ec.vector)
                self.text_meta.append(ec.chunk)

        if text_vecs:
            self.text_index.add(np.array(text_vecs, dtype="float32"))
        if image_vecs:
            self.image_index.add(np.array(image_vecs, dtype="float32"))

    def search_text(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        return self._search(self.text_index, self.text_meta, query_vec, k)

    def search_images(self, query_vec: np.ndarray, k: int) -> list[tuple[Chunk, float]]:
        return self._search(self.image_index, self.image_meta, query_vec, k)

    @staticmethod
    def _search(index, meta, query_vec, k) -> list[tuple[Chunk, float]]:
        if index.ntotal == 0:
            return []
        k = min(k, index.ntotal)
        distances, indices = index.search(
            np.array([query_vec], dtype="float32"), k
        )
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            results.append((meta[idx], float(dist)))
        return results

    def save(self, out_dir: str | Path):
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.text_index, str(out_dir / "text.faiss"))
        faiss.write_index(self.image_index, str(out_dir / "image.faiss"))
        with open(out_dir / "metadata.pkl", "wb") as f:
            pickle.dump(
                {
                    "text_meta": [asdict(c) for c in self.text_meta],
                    "image_meta": [asdict(c) for c in self.image_meta],
                    "text_dim": self.text_dim,
                    "image_dim": self.image_dim,
                },
                f,
            )

    @classmethod
    def load(cls, in_dir: str | Path) -> "VectorStore":
        in_dir = Path(in_dir)
        with open(in_dir / "metadata.pkl", "rb") as f:
            data = pickle.load(f)

        store = cls(text_dim=data["text_dim"], image_dim=data["image_dim"])
        store.text_index = faiss.read_index(str(in_dir / "text.faiss"))
        store.image_index = faiss.read_index(str(in_dir / "image.faiss"))
        store.text_meta = [Chunk(**c) for c in data["text_meta"]]
        store.image_meta = [Chunk(**c) for c in data["image_meta"]]
        return store
