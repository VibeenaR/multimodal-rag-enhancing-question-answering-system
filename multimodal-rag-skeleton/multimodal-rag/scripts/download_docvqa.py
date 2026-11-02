"""
Pulls the DocVQA benchmark referenced in Review-1 Section 7 (Methodology):
"pixparse / docvqa-single-page-questions database, ... train, validation,
and test datasets unchanged."

Usage:
    python scripts/download_docvqa.py
    python scripts/download_docvqa.py --split validation --limit 500
"""
import argparse

from datasets import load_dataset

from app.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


def download(split: str, limit: int | None) -> None:
    logger.info(f"Loading '{settings.docvqa_hf_id}' split='{split}' from HuggingFace...")
    ds = load_dataset(settings.docvqa_hf_id, split=split)

    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    logger.info(f"Loaded {len(ds)} examples.")
    logger.info(f"Example fields: {ds.features}")
    if len(ds) > 0:
        sample = ds[0]
        logger.info(f"Sample question: {sample.get('question')!r}")
        logger.info(f"Sample answers: {sample.get('answers')!r}")

    out_dir = settings.processed_dir / "docvqa" / split
    out_dir.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(out_dir))
    logger.info(f"Saved to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="validation",
        choices=["train", "validation", "test"],
        help="Which DocVQA split to pull (train/validation/test).",
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Optional cap on number of examples."
    )
    args = parser.parse_args()
    download(args.split, args.limit)
