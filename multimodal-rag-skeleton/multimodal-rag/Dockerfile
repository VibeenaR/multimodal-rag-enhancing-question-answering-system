# Multi-Model RAG — container image
# Build:  docker compose build
# Run:    docker compose up
FROM python:3.11-slim

# All model downloads (MiniLM, CLIP) go to /app/.cache, which docker-compose
# keeps in a volume, so they are downloaded only once.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    XDG_CACHE_HOME=/app/.cache \
    HF_HOME=/app/.cache/huggingface \
    TORCH_HOME=/app/.cache/torch \
    PYTHONPATH=/app

WORKDIR /app

# Runtime libraries needed by FAISS (OpenMP), OpenCV and PyMuPDF.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages first so this layer is cached between code changes.
COPY requirements.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install \
        "fastapi" "uvicorn[standard]" "python-multipart" \
        "pydantic" "pydantic-settings" "python-dotenv" \
        "google-generativeai" "pillow" \
        "PyMuPDF" "pdfplumber" \
        "faiss-cpu" "sentence-transformers" "open_clip_torch" \
        "opencv-python-headless<4.10" \
    && pip install "numpy<2" "pyarrow<18"
# The second pip install lists every package the app imports, so a package that is
# missing from requirements.txt (like google-generativeai) is added here instead.
# Packages already installed are skipped.
# numpy<2 is needed by FAISS/OpenCV. Newer pyarrow versions require NumPy 2 and
# crash on import, so pyarrow is pinned to a release that works with NumPy 1.

# Application code (including the static UI in app/static).
COPY app ./app

# Uploaded PDFs, the FAISS index, page images and crops are written here.
# docker-compose mounts a volume on it so they survive restarts.
RUN mkdir -p /app/data /app/.cache
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]