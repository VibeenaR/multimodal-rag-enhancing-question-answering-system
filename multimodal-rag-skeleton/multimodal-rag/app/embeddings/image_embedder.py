"""
Image embedding via CLIP ViT-B/32, per Review-1 Section 5
("embedding layer associates the visual data via CLIP").

NOTE on the "unified vector space" claim in Review-1:
CLIP's text tower and MiniLM produce embeddings of different dimensionality
(512 vs 384) and were trained on different objectives, so they are NOT
numerically comparable out of the box. Two honest ways to handle this for
Review-2 — pick one and say so explicitly in your report:

  (A) Two FAISS indexes, one per modality, queried in parallel with CLIP's
      own text encoder for the image index and MiniLM for the text index,
      then merge/re-rank results by score. (Implemented in vector_store.py
      as the default — it's the more defensible/accurate framing.)

  (B) Train a small linear projection head that maps MiniLM's 384-d space
      into CLIP's 512-d space (or vice versa) on paired (caption, image)
      data, so a single index genuinely works. More faithful to the
      original "single shared space" pitch, more work to justify/evaluate.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config import settings

_device = "cuda" if torch.cuda.is_available() else "cpu"


@lru_cache(maxsize=1)
def _get_model_and_preprocess():
    model, _, preprocess = open_clip.create_model_and_transforms(
        settings.clip_model_name, pretrained=settings.clip_pretrained
    )
    model = model.to(_device).eval()
    tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
    return model, preprocess, tokenizer


def embed_image(image_path: str | Path) -> np.ndarray:
    """Returns an (image_embedding_dim,) float32, L2-normalized vector."""
    model, preprocess, _ = _get_model_and_preprocess()
    image = Image.open(image_path).convert("RGB")
    tensor = preprocess(image).unsqueeze(0).to(_device)

    with torch.no_grad():
        features = model.encode_image(tensor)
        features = features / features.norm(dim=-1, keepdim=True)

    return features.squeeze(0).cpu().numpy().astype("float32")


def embed_text_in_clip_space(text: str | list[str]) -> np.ndarray:
    """Embeds text using CLIP's own text tower, so a query can be compared
    against image vectors within CLIP's space (used by the image-side index
    in retriever.py's dual-index approach)."""
    model, _, tokenizer = _get_model_and_preprocess()
    single = isinstance(text, str)
    texts = [text] if single else text

    tokens = tokenizer(texts).to(_device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)

    vectors = features.cpu().numpy().astype("float32")
    return vectors[0] if single else vectors
