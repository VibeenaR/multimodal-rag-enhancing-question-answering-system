"""
Splits a generated, cited answer into an ordered sequence of short
narration steps (Review-2 upgrade: narrated visual walkthrough). Each step
is tied to at most one citation's page/region, so the frontend can
pan/zoom/highlight the source document in sync with spoken narration,
instead of just returning one flat block of answer text.

IMPORTANT: this version does NOT call the VLM a second time. The original
implementation asked Gemini to re-split the answer into steps, which
doubled your API usage per question (one call for the answer, one for the
steps) and burns through the free tier's daily quota twice as fast. Since
your system prompt already asks the model to cite sources inline as
"[1]", "[2]", etc., this version just parses those markers directly out
of the answer text with regex — no extra request, no extra cost, and it's
actually more reliable than hoping a second model call reproduces the
same citations faithfully.

Save this file as: app/generation/step_breakdown.py
(replaces the previous version — same filename, same location)
"""
import re

from app.models.schemas import Citation, WalkthroughStep

# Splits on sentence-ending punctuation followed by whitespace, but not on
# abbreviation-style periods inside something like "e.g." or "Fig. 2" when
# immediately followed by a lowercase letter or digit (best-effort heuristic
# — not a full sentence tokenizer, but good enough for narration chunking).
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9⚠])")

# Matches one or more citation numbers in brackets, e.g. "[1]" or "[2, 3]".
_CITATION_MARKERS = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")

# Strips the trailing "Sources: [...]" summary line your system prompt adds
# — it's not narration content, it's a machine-readable footer.
_SOURCES_LINE = re.compile(r"\n?Sources:\s*\[[^\]]*\]\s*$", re.IGNORECASE)


def _first_citation_number(sentence: str) -> int | None:
    """Returns the first citation number referenced in a sentence, if any
    (e.g. "...takes five steps [2]." -> 2). Sentences citing multiple
    sources use only the first — one visual highlight per narration step
    keeps the walkthrough simple to follow."""
    match = _CITATION_MARKERS.search(sentence)
    if not match:
        return None
    first_num = match.group(1).split(",")[0].strip()
    try:
        return int(first_num)
    except ValueError:
        return None


def build_steps(answer: str, citations: list[Citation]) -> list[WalkthroughStep]:
    """Turns a cited answer into a sequence of (text, page, bbox) steps the
    frontend narrates and visually points at, one at a time — purely by
    parsing the answer's own inline [n] citation markers, with no
    additional API call."""
    cleaned = _SOURCES_LINE.sub("", answer).strip()
    if not cleaned:
        return []

    sentences = [s.strip() for s in _SENTENCE_SPLIT.split(cleaned) if s.strip()]
    if not sentences:
        sentences = [cleaned]

    steps: list[WalkthroughStep] = []
    for sentence in sentences:
        source_num = _first_citation_number(sentence)
        citation = None
        if source_num is not None and 1 <= source_num <= len(citations):
            citation = citations[source_num - 1]

        # Strip the bracketed citation markers from the spoken text itself
        # — "[1]" read aloud by TTS is noise, not information.
        narration_text = _CITATION_MARKERS.sub("", sentence).strip()
        if not narration_text:
            continue

        steps.append(
            WalkthroughStep(
                text=narration_text,
                document_id=citation.document_id if citation else None,
                page_number=citation.page_number if citation else None,
                bbox=citation.bbox if citation else None,
            )
        )

    return steps or [WalkthroughStep(text=cleaned)]