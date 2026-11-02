"""
Quantitative evaluation against the DocVQA benchmark using ANLS
(Average Normalized Levenshtein Similarity) — the standard DocVQA metric,
as named in Review-1 Section 7 ("ANLS score to compare the performance
with the available literature").

ANLS per question = max over ground-truth answers of:
    1 - (levenshtein_distance(pred, gt) / max(len(pred), len(gt)))
    (clamped to 0 if that value < 0.5, per the official DocVQA definition)

Usage:
    python scripts/evaluate_anls.py --split validation --limit 200
"""
import argparse

import Levenshtein
from datasets import load_dataset
from tqdm import tqdm

from app.config import settings
from app.generation.vlm_generator import generate_answer
from app.indexing.vector_store import VectorStore
from app.retrieval.retriever import retrieve
from app.utils.logging import get_logger

logger = get_logger(__name__)

ANLS_THRESHOLD = 0.5


def _anls_single(prediction: str, ground_truths: list[str]) -> float:
    best = 0.0
    pred = prediction.strip().lower()
    for gt in ground_truths:
        gt = gt.strip().lower()
        if not pred and not gt:
            score = 1.0
        else:
            dist = Levenshtein.distance(pred, gt)
            norm = dist / max(len(pred), len(gt), 1)
            score = 1 - norm
        if score >= ANLS_THRESHOLD:
            best = max(best, score)
    return best


def evaluate(split: str, limit: int | None) -> dict:
    logger.info(f"Loading DocVQA split='{split}' for evaluation...")
    ds = load_dataset(settings.docvqa_hf_id, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))

    logger.info(f"Loading vector index from {settings.index_dir}...")
    store = VectorStore.load(settings.index_dir)

    scores = []
    for example in tqdm(ds, desc="Evaluating"):
        question = example["question"]
        ground_truths = example.get("answers", [])

        evidence = retrieve(store, question, k=settings.top_k)
        response = generate_answer(question, evidence)

        score = _anls_single(response.answer, ground_truths)
        scores.append(score)

    mean_anls = sum(scores) / len(scores) if scores else 0.0
    logger.info(f"Mean ANLS over {len(scores)} examples: {mean_anls:.4f}")
    return {"split": split, "n": len(scores), "mean_anls": mean_anls}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"])
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    result = evaluate(args.split, args.limit)
    print(result)
