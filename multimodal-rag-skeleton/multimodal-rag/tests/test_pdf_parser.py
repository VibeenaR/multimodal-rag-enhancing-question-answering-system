"""
Basic smoke tests for the ingestion layer. Point SAMPLE_PDF at one of your
custom-corpus PDFs before running: pytest tests/test_pdf_parser.py
"""
from pathlib import Path

import pytest

from app.ingestion.pdf_parser import parse_pdf

SAMPLE_PDF = Path("data/raw_pdfs/sample.pdf")


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Add a sample PDF to data/raw_pdfs/ first")
def test_parse_pdf_returns_chunks():
    chunks = parse_pdf(SAMPLE_PDF)
    assert len(chunks) > 0
    assert all(c.page_number >= 1 for c in chunks)
    assert all(c.document_id == SAMPLE_PDF.stem for c in chunks)


@pytest.mark.skipif(not SAMPLE_PDF.exists(), reason="Add a sample PDF to data/raw_pdfs/ first")
def test_chunks_have_expected_modalities():
    chunks = parse_pdf(SAMPLE_PDF)
    modalities = {c.modality for c in chunks}
    assert modalities.issubset({"text", "table", "image"})
