# Multi-Model RAG: Cross-Modal Document QA

Implementation skeleton for SWE3004 Review-2, matching the architecture in
Review-1: **ingestion → multimodal embedding → unified FAISS index →
retrieval → grounded VLM generation with citation → FastAPI/Docker serving**.

## Project layout

```
multimodal-rag/
├── app/
│   ├── config.py                # central settings (model names, paths, k)
│   ├── main.py                  # FastAPI entrypoint
│   ├── models/
│   │   └── schemas.py           # Pydantic request/response + internal dataclasses
│   ├── ingestion/
│   │   └── pdf_parser.py        # PDF -> text blocks, table chunks, image chunks (page-tagged)
│   ├── embeddings/
│   │   ├── text_embedder.py     # sentence-transformers (MiniLM-L6) wrapper
│   │   └── image_embedder.py    # CLIP ViT-B/32 wrapper (image + text-in-CLIP-space)
│   ├── indexing/
│   │   └── vector_store.py      # unified FAISS index (text+image share one space) + metadata store
│   ├── retrieval/
│   │   └── retriever.py         # top-k cosine search over the unified index
│   ├── generation/
│   │   └── vlm_generator.py     # VLM call, grounded on retrieved evidence, returns citation
│   ├── api/
│   │   └── routes.py            # /upload, /ask, /health endpoints
│   └── utils/
│       └── logging.py
├── scripts/
│   ├── download_docvqa.py       # pulls pixparse/docvqa-single-page-questions from HF
│   ├── build_index.py           # ingest data/raw_pdfs/*.pdf -> data/indexes/
│   └── evaluate_anls.py         # ANLS metric over DocVQA val split
├── data/
│   ├── raw_pdfs/                # your 15-20 custom PDFs go here
│   ├── processed/                # cached parsed pages (json)
│   └── indexes/                  # FAISS index + metadata (.faiss / .pkl)
├── tests/
│   ├── test_pdf_parser.py
│   └── test_retriever.py
├── notebooks/
│   └── exploration.ipynb        # empty starter notebook for ad-hoc analysis
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env.example
```

## Why this maps to your Review-1 design

- **Ingestion layer** → `app/ingestion/pdf_parser.py`
- **Embedding layer** (CLIP + sentence-transformers, one shared space) →
  `app/embeddings/`
- **Unified vector DB (FAISS)** → `app/indexing/vector_store.py`
- **Retrieval layer** (top-k cosine) → `app/retrieval/retriever.py`
- **Generation layer** (VLM + citation) → `app/generation/vlm_generator.py`
- **Serving layer** (FastAPI + Docker) → `app/main.py`, `app/api/routes.py`,
  `Dockerfile`

## Dataset

DocVQA is pulled from the HuggingFace dataset your report already names:
**`pixparse/docvqa-single-page-questions`** (train/validation/test splits,
50k+ QA pairs over 12k+ document images). See `scripts/download_docvqa.py`.

For the custom corpus, drop 15-20 real PDFs (reports/manuals/papers with
tables and figures) into `data/raw_pdfs/`.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# 1. Pull DocVQA
python scripts/download_docvqa.py

# 2. Build the index over your custom PDFs
python scripts/build_index.py

# 3. Run the API
uvicorn app.main:app --reload --port 8000

# 4. Evaluate against DocVQA (ANLS)
python scripts/evaluate_anls.py --split validation --limit 200
```

Or via Docker:

```bash
docker compose up --build
```

## Implementation order (suggested, for your remaining timeline)

1. `pdf_parser.py` — get text/table/image extraction with page numbers working
   and unit-tested on 2-3 sample PDFs first.
2. `text_embedder.py` + `image_embedder.py` — confirm both output the *same*
   embedding dimensionality (or you project one into the other's space —
   see note in `image_embedder.py`) before touching FAISS.
3. `vector_store.py` — build the shared index, store `(vector, modality,
   doc_id, page_number, content_ref)` metadata per entry.
4. `retriever.py` — sanity check retrieval quality by hand on a handful of
   questions before wiring in generation.
5. `vlm_generator.py` — grounded generation + citation formatting.
6. `routes.py` / `main.py` — expose it all over FastAPI.
7. `evaluate_anls.py` — run the DocVQA benchmark, record ANLS for your
   Review-2 results table.
8. `Dockerfile` / `docker-compose.yml` — containerize last, once the app
   runs cleanly locally.
