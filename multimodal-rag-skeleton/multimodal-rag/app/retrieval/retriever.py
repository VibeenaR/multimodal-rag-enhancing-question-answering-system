"""
Retrieval layer (Review-1 Section 5, "top-k search using the combined
database of images and texts").

Why this was rewritten (accuracy upgrade)
-----------------------------------------
The first version had three weaknesses that made answers drift off-topic:

1. It merged MiniLM text distances and CLIP image distances in one sort,
   although the two scores live in different embedding spaces and cannot be
   compared. Textless image crops could outrank the real passages.
2. In "scoped" mode it sent chunks sorted by (page, modality), where
   "image" sorts before "text", and then cut the list at 80 chunks. On a PDF
   with many detected image regions, the text could be cut off entirely.
3. When searching all documents it relied on one dense embedding model only,
   so questions that hinge on an exact word, name or number often missed the
   right passage, and passages from other uploaded PDFs got mixed in.

How retrieval works now
-----------------------
- Text and tables are ranked by HYBRID search: dense similarity (MiniLM)
  plus keyword search (BM25), merged with Reciprocal Rank Fusion.
- The system works out WHICH document the question is about (the scope you
  picked, a document named in the question, or the document whose passages
  score best). Questions that say "compare" or "all documents" search across
  documents instead.
- If that document is small enough, ALL of its text and tables are sent to
  the model, so nothing can be missed (mode "full_document"). Larger
  documents get the best passages plus their neighbours (mode "semantic").
- Images are added only when the question is about a figure, chart, diagram
  or picture (or the PDF has no text at all), ranked by CLIP, a few at most.
"""
import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass

from app.config import settings
from app.embeddings.image_embedder import embed_text_in_clip_space
from app.embeddings.text_embedder import embed_text
from app.indexing.vector_store import VectorStore
from app.models.schemas import Chunk

FULL_DOC_CHAR_BUDGET = 150_000   # about 35k tokens; documents up to this size are sent whole
MIN_PRIMARY_PASSAGES = 8         # best passages to keep when a document is too big to send whole
NEIGHBOR_RADIUS = 1              # also include the passage before/after each best passage
MAX_IMAGES = 4                   # images attached for figure/chart questions
MAX_IMAGES_NO_TEXT = 6           # images attached for scanned PDFs with no extractable text
RRF_K = 60                       # standard Reciprocal Rank Fusion constant
DOC_DOMINANCE_RATIO = 1.3        # top document must beat the runner-up by this factor to be chosen
CANDIDATE_LIMIT = 2000

_VISUAL_INTENT = re.compile(
    r"\b(image|images|figure|figures|fig|chart|charts|graph|graphs|diagram|diagrams|"
    r"picture|pictures|photo|photos|illustration|illustrations|plot|plots|screenshot|"
    r"visual|visuals|flowchart|architecture|logo|map|drawing|infographic)\b",
    re.IGNORECASE,
)
_CROSS_DOC = re.compile(
    r"\b(compare|comparison|compared|difference between|differences between|versus|vs|"
    r"across (?:all )?(?:the )?(?:documents|pdfs|files)|all (?:the )?(?:documents|pdfs|files)|"
    r"each (?:document|pdf|file)|both (?:documents|pdfs|files)|every (?:document|pdf|file))\b",
    re.IGNORECASE,
)
_DEICTIC = re.compile(
    r"\b(?:this|the|that|attached|uploaded|current|given|my)\s+"
    r"(?:document|pdf|file|doc|report|paper|article|book|manual|guide|notes|text)\b"
    r"|\buploaded\b",
    re.IGNORECASE,
)
_STOPWORDS = frozenset(
    "a an the and or but if then of to in on at by for with from as is are was were be been being "
    "this that these those it its i you he she they we what which who whom whose when where why how "
    "do does did done can could should would will shall may might must about into over under than "
    "so such not no nor too very there here also any all some each more most other".split()
)
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)?")


@dataclass
class RetrievedEvidence:
    chunk: Chunk
    score: float  # L2 distance (semantic mode); 0.0 placeholder in full_document mode
    mode: str = "semantic"  # "semantic" | "full_document"


def distance_to_confidence(l2_distance: float) -> float:
    """Converts an L2 distance between two L2-normalized embeddings into a
    0-100 confidence score. For unit vectors, L2^2 = 2*(1 - cosine_sim), so
    cosine_sim = 1 - L2^2/2. Clamped to [0, 100]."""
    cos_sim = 1 - (l2_distance ** 2) / 2
    return round(max(0.0, min(1.0, cos_sim)) * 100, 1)


# ---------------------------------------------------------------------------
# Keyword (BM25) index over text/table chunks
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall((text or "").lower()) if t not in _STOPWORDS and (len(t) > 1 or t.isdigit())]


class _KeywordIndex:
    def __init__(self, chunks: list[Chunk]):
        self.n = len(chunks)
        self.doc_len: list[int] = []
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, chunk in enumerate(chunks):
            tokens = _tokenize(chunk.content)
            self.doc_len.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((i, tf))
        self.avgdl = (sum(self.doc_len) / self.n) if self.n else 1.0

    def search(self, query: str, limit: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(_tokenize(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)
            idf = math.log(1 + (self.n - df + 0.5) / (df + 0.5))
            for i, tf in posting:
                denom = tf + 1.5 * (1 - 0.75 + 0.75 * self.doc_len[i] / (self.avgdl or 1.0))
                scores[i] += idf * tf * 2.5 / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:limit]


_cache: dict = {"key": None, "kw": None, "index_of": None}


def _caches(store: VectorStore):
    """Builds (and reuses) the keyword index and a chunk_id -> position map.
    The store only ever grows, so (identity, size) is a safe cache key."""
    key = (id(store), len(store.text_meta))
    if _cache["key"] != key:
        _cache["kw"] = _KeywordIndex(store.text_meta)
        _cache["index_of"] = {c.chunk_id: i for i, c in enumerate(store.text_meta)}
        _cache["key"] = key
    return _cache["kw"], _cache["index_of"]


def _fused_ranking(store: VectorStore, question: str, limit: int):
    """Hybrid text search. Returns ([(position, fused_score), ...] best first,
    {position: dense L2 distance})."""
    if not store.text_meta:
        return [], {}
    kw, index_of = _caches(store)

    dense_hits = store.search_text(embed_text(question), limit)
    dense = [(index_of[c.chunk_id], dist) for c, dist in dense_hits if c.chunk_id in index_of]
    keyword = kw.search(question, limit)

    fused: dict[int, float] = defaultdict(float)
    for rank, (i, _) in enumerate(dense):
        fused[i] += 1.0 / (RRF_K + rank + 1)
    for rank, (i, _) in enumerate(keyword):
        fused[i] += 1.0 / (RRF_K + rank + 1)

    ranked = sorted(fused.items(), key=lambda kv: -kv[1])
    return ranked, {i: d for i, d in dense}


# ---------------------------------------------------------------------------
# Which document is the question about?
# ---------------------------------------------------------------------------
def _normalize_name(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def _documents_named_in(question: str, doc_ids: list[str]) -> list[str]:
    q = " " + _normalize_name(question) + " "
    return [d for d in doc_ids if len(_normalize_name(d)) >= 3 and f" {_normalize_name(d)} " in q]


def _rank_documents(store: VectorStore, ranked: list[tuple[int, float]], top_n: int = 30):
    per_doc: dict[str, list[float]] = defaultdict(list)
    for i, score in ranked[:top_n]:
        per_doc[store.text_meta[i].document_id].append(score)
    totals = {d: sum(sorted(v, reverse=True)[:3]) for d, v in per_doc.items()}
    return sorted(totals.items(), key=lambda kv: -kv[1])


def _latest_document(store: VectorStore) -> str | None:
    """The most recently added document. Uploads are appended, so it is the
    document of the last chunk in the store."""
    if store.text_meta:
        return store.text_meta[-1].document_id
    if store.image_meta:
        return store.image_meta[-1].document_id
    return None


def _modality_rank(chunk: Chunk) -> int:
    return 0 if chunk.modality == "text" else 1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------
def retrieve(
    store: VectorStore,
    question: str,
    k: int = None,
    document_id: str | None = None,
) -> list[RetrievedEvidence]:
    k = max(k or settings.top_k, MIN_PRIMARY_PASSAGES)
    doc_ids = sorted({c.document_id for c in (store.text_meta + store.image_meta)})
    if not doc_ids:
        return []

    target_docs: list[str] | None = None
    if document_id:
        target_docs = [document_id]
    else:
        named = _documents_named_in(question, doc_ids)
        if len(doc_ids) == 1:
            target_docs = doc_ids
        elif named:
            target_docs = named

    # "Summarize this document", "what does the PDF say", ... name no document, so
    # they mean the one the user uploaded most recently.
    if target_docs is None and _DEICTIC.search(question) and not _CROSS_DOC.search(question):
        latest = _latest_document(store)
        if latest:
            target_docs = [latest]

    if target_docs is None and not _CROSS_DOC.search(question) and store.text_meta:
        ranked, _ = _fused_ranking(store, question, min(len(store.text_meta), CANDIDATE_LIMIT))
        docs = _rank_documents(store, ranked)
        if docs and (len(docs) == 1 or docs[0][1] >= DOC_DOMINANCE_RATIO * docs[1][1]):
            target_docs = [docs[0][0]]

    if target_docs and len(target_docs) == 1:
        return _retrieve_single_document(store, question, target_docs[0], k)
    return _retrieve_across_documents(store, question, k, target_docs)


# ---------------------------------------------------------------------------
# One document
# ---------------------------------------------------------------------------
def _retrieve_single_document(store: VectorStore, question: str, doc: str, k: int) -> list[RetrievedEvidence]:
    in_doc = [(i, c) for i, c in enumerate(store.text_meta) if c.document_id == doc]
    total_chars = sum(len(c.content) for _, c in in_doc)
    evidence: list[RetrievedEvidence] = []

    if in_doc and total_chars <= FULL_DOC_CHAR_BUDGET:
        in_doc.sort(key=lambda ic: (ic[1].page_number, _modality_rank(ic[1]), ic[0]))
        evidence = [RetrievedEvidence(chunk=c, score=0.0, mode="full_document") for _, c in in_doc]
    elif in_doc:
        ranked, dense_dist = _fused_ranking(store, question, min(len(store.text_meta), CANDIDATE_LIMIT))
        best = [i for i, _ in ranked if store.text_meta[i].document_id == doc][:k]
        evidence = _passages_with_neighbors(store, best, dense_dist)

    evidence += _select_images(store, question, {doc}, has_text=bool(in_doc))
    return evidence


# ---------------------------------------------------------------------------
# Several documents (compare / all documents / unclear)
# ---------------------------------------------------------------------------
def _retrieve_across_documents(store: VectorStore, question: str, k: int, allowed: list[str] | None) -> list[RetrievedEvidence]:
    evidence: list[RetrievedEvidence] = []
    docs_used: set[str] = set(allowed or [])

    if store.text_meta:
        ranked, dense_dist = _fused_ranking(store, question, min(len(store.text_meta), CANDIDATE_LIMIT))
        if allowed:
            ranked = [(i, s) for i, s in ranked if store.text_meta[i].document_id in allowed]
        best = [i for i, _ in ranked[:k]]
        evidence = _passages_with_neighbors(store, best, dense_dist)
        docs_used |= {store.text_meta[i].document_id for i in best}

    evidence += _select_images(store, question, docs_used or None, has_text=bool(evidence))
    return evidence


def _passages_with_neighbors(store: VectorStore, best: list[int], dense_dist: dict[int, float]) -> list[RetrievedEvidence]:
    """Best passages plus the passage before and after each one (so an answer
    that starts at the end of one passage and finishes in the next is intact),
    returned in reading order."""
    chosen = set(best)
    n = len(store.text_meta)
    for i in best:
        base = store.text_meta[i]
        if base.modality != "text":
            continue
        for j in range(i - NEIGHBOR_RADIUS, i + NEIGHBOR_RADIUS + 1):
            if 0 <= j < n:
                other = store.text_meta[j]
                if other.modality == "text" and other.document_id == base.document_id:
                    chosen.add(j)

    worst = max(dense_dist.values(), default=1.0)
    ordered = sorted(chosen, key=lambda i: (store.text_meta[i].document_id, store.text_meta[i].page_number, _modality_rank(store.text_meta[i]), i))
    return [RetrievedEvidence(chunk=store.text_meta[i], score=dense_dist.get(i, worst), mode="semantic") for i in ordered]


# ---------------------------------------------------------------------------
# Images (only when they are likely to matter)
# ---------------------------------------------------------------------------
def _select_images(store: VectorStore, question: str, docs: set[str] | None, has_text: bool) -> list[RetrievedEvidence]:
    if not store.image_meta:
        return []
    if has_text and not _VISUAL_INTENT.search(question):
        return []

    limit = MAX_IMAGES if has_text else MAX_IMAGES_NO_TEXT
    hits = store.search_images(embed_text_in_clip_space(question), len(store.image_meta))
    picked: list[RetrievedEvidence] = []
    for chunk, dist in hits:  # already sorted, closest first
        if docs is not None and chunk.document_id not in docs:
            continue
        picked.append(RetrievedEvidence(chunk=chunk, score=dist, mode="semantic"))
        if len(picked) >= limit:
            break
    return picked