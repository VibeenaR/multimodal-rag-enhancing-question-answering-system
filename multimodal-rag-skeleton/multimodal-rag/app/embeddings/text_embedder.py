"""
Text/table embedding via sentence-transformers (MiniLM-L6), per Review-1
Section 5 ("embedding layer ... via sentence-transformer models").
"""
from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings


@lru_cache(maxsize=1)
def _get_model() -> SentenceTransformer:
    # Cached so the (relatively heavy) model is loaded once per process,
    # not once per request.
    return SentenceTransformer(settings.text_embedding_model)


def embed_text(texts: str | list[str]) -> np.ndarray:
    """Returns an (N, text_embedding_dim) float32, L2-normalized array."""
    single = isinstance(texts, str)
    if single:
        texts = [texts]

    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,  # cosine similarity via dot product
        show_progress_bar=False,
        convert_to_numpy=True,
    ).astype("float32")

    return vectors[0] if single else vectors
