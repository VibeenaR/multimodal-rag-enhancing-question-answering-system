# from fastapi import FastAPI

# from app.api.routes import router

# app = FastAPI(
#     title="Multimodal RAG API",
#     description="Cross-modal document QA over text, tables and images.",
#     version="0.1.0",
# )

# app.include_router(router)


# @app.get("/")
# def root():
#     return {
#         "message": "Multimodal RAG API is running.",
#         "docs": "/docs",
#         "endpoints": ["/upload", "/ask", "/health"],
#     }

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router

app = FastAPI(
    title="Multimodal RAG API",
    description="Cross-modal document QA over text, tables and images.",
    version="0.1.0",
)

app.include_router(router)

# Simple demo frontend — served at /ui, so the JSON root response at "/"
# (used as a quick liveness check) is left untouched.
app.mount("/ui", StaticFiles(directory="app/static", html=True), name="ui")


@app.get("/")
def root():
    return {
        "message": "Multimodal RAG API is running.",
        "docs": "/docs",
        "ui": "/ui",
        "endpoints": ["/upload", "/ask", "/health"],
    }