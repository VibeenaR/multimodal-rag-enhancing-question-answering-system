"""
Central configuration for the Multimodal RAG system.
All paths and model identifiers referenced elsewhere in the app come from here,
so swapping a model or dataset location is a one-line change.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    project_root: Path = Path(__file__).resolve().parent.parent
    raw_pdf_dir: Path = project_root / "data" / "raw_pdfs"
    processed_dir: Path = project_root / "data" / "processed"
    index_dir: Path = project_root / "data" / "indexes"

    # --- Embedding models (must match Review-1 proposal) ---
    text_embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "openai"

    # --- Embedding dimensions ---
    # MiniLM-L6 -> 384-d, CLIP ViT-B/32 -> 512-d.
    # These differ, so the index either (a) keeps two FAISS indexes under one
    # retrieval API, or (b) projects both into a shared dimension with a
    # learned/linear projection head. Default here is (a) — see vector_store.py.
    text_embedding_dim: int = 384
    image_embedding_dim: int = 512

    # --- FAISS ---
    faiss_index_type: str = "HNSW"  # "HNSW" or "IVF_PQ" per your architecture diagram
    hnsw_m: int = 32

    # --- Retrieval ---
    top_k: int = 3

    # --- Generation ---
    vlm_model: str = "claude-sonnet-4-6"
    anthropic_api_key: str = ""  # set via .env

    # --- Free-tier alternative (Google Gemini) ---
    gemini_model: str = "gemini-3.6-flash"
    gemini_api_key: str = ""  # set via .env

    # Optional pool of Gemini API keys (one per Google Cloud project), used
    # by app/generation/vlm_generator.py to rotate to a fresh key whenever
    # the current one hits its free-tier quota (20/day, 5/min per project).
    # Comma-separated in .env, e.g.:
    #   GEMINI_API_KEYS=key_from_project1,key_from_project2,key_from_project3
    # Leave unset to keep using just gemini_api_key above.
    gemini_api_keys: str = ""  # set via .env (optional)

    # Low temperature = the model sticks to what the evidence says instead of
    # getting creative, which is what you want for document question answering.
    gemini_temperature: float = 0.1

    # --- DocVQA dataset ---
    docvqa_hf_id: str = "pixparse/docvqa-single-page-questions"

    model_config = {"env_file": ".env"}


settings = Settings()