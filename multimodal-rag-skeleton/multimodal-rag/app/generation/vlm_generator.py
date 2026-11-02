"""
Generation layer (Review-1 Section 5, "generation layer ... produces an
answer based on the retrieved data using a vision-language model" and
"That answer is then cited"). Uses Google Gemini's free tier (no billing
required) — swap `_call_vlm` for a different provider's client if needed;
nothing else in the pipeline depends on which VLM is used.

Review-2 upgrade: citations now carry a `bbox` (from region-detected image
chunks) so the frontend can highlight the exact source region, and the
answer is additionally split into narration `steps` via step_breakdown.py
for the visual walkthrough feature.

Key-rotation upgrade: Gemini's free tier caps you at 20 requests/day AND
5 requests/minute, PER GOOGLE CLOUD PROJECT. A single API key can't get
around that — but several free-tier keys, each from a *different* Google
Cloud project, each get their own independent 20/day + 5/min bucket. This
module now holds a small pool of keys and automatically rotates to the
next one whenever the current key hits a quota error (429 /
ResourceExhausted), instead of failing the whole request.

HOW TO ADD KEYS:
Set an environment variable (e.g. in your .env file) called GEMINI_API_KEYS
with your keys separated by commas, one key per Google Cloud project:

    GEMINI_API_KEYS=key_from_project_1,key_from_project_2,key_from_project_3

(To create more keys/projects: console.cloud.google.com -> New Project,
then aistudio.google.com/apikey -> Create API key -> "Create API key in
new project" -> pick that new project.)

If GEMINI_API_KEYS isn't set, this falls back to the single
settings.gemini_api_key you already had configured, so nothing breaks if
you don't set up rotation — you just won't get the extra quota headroom.
"""
import re

import google.generativeai as genai
from google.api_core.exceptions import ResourceExhausted

from app.config import settings
from app.generation.step_breakdown import build_steps
from app.models.schemas import AskResponse, Citation
from app.retrieval.retriever import RetrievedEvidence, distance_to_confidence

SYSTEM_PROMPT = (
    "You are a precise document question-answering assistant. Answer using the "
    "numbered evidence provided below.\n\n"
    "Accuracy rules:\n"
    "- Read ALL the evidence before answering, including tables and images; the "
    "answer is often in a table row or a figure label, not the body text.\n"
    "- Copy numbers, names, dates, units and technical terms exactly as written "
    "in the evidence. Never round, convert or paraphrase a figure.\n"
    "- If sources disagree, say so and report both values.\n"
    "- Answer the question that was asked, directly, in the first sentence. Then "
    "add only the supporting detail that is needed.\n"
    "- For a summary or explanation, cover every main point in the evidence, in "
    "the order the document presents them.\n\n"
    "Formatting rules (the answer is shown on screen and read aloud):\n"
    "- Plain sentences only. Do NOT use markdown: no #, ##, ###, no ---, no "
    "asterisks, no backticks, no tables.\n"
    "- For lists, put each item on its own line starting with '- '.\n"
    "- Do NOT mention page numbers anywhere in the answer. Refer to documents by "
    "name only when needed.\n\n"
    "For any claim that IS supported by the evidence, cite the source number(s) "
    "inline (e.g. 'the process takes five steps [2]').\n\n"
    "Answer ONLY from the evidence. Never use outside knowledge, never guess, and "
    "never describe anything the evidence does not say.\n"
    "If the evidence does not contain the answer, reply that the uploaded document "
    "does not cover it, then briefly say what the document does cover that is "
    "closest to the question. If it covers only part of the question, answer that "
    "part and say which part is missing.\n\n"
    "End your answer with a line: 'Sources: [n, n, ...]' listing only the source "
    "numbers actually used (write 'Sources: []' if none were used)."
)


# --- Key pool setup ---------------------------------------------------
def _load_key_pool() -> list[str]:
    """Reads settings.gemini_api_keys (comma-separated, from GEMINI_API_KEYS
    in .env) if set, otherwise falls back to the single
    settings.gemini_api_key. Blank entries and surrounding whitespace are
    cleaned up."""
    raw = (settings.gemini_api_keys or "").strip()
    if raw:
        keys = [k.strip() for k in raw.split(",") if k.strip()]
        if keys:
            return keys
    return [settings.gemini_api_key]


_KEY_POOL: list[str] = _load_key_pool()
_current_key_index = 0


def _configure_key(index: int) -> None:
    genai.configure(api_key=_KEY_POOL[index])


_configure_key(_current_key_index)


class AllKeysExhaustedError(Exception):
    """Raised when every key in the pool has hit a quota error for this
    request. The original per-key error text is included for the last
    key tried, so the user still sees the real Google error/retry_delay."""


def _build_evidence_parts(evidence: list[RetrievedEvidence]) -> list:
    """Turns retrieved Chunks into Gemini content parts — text/table chunks
    as strings, image chunks as PIL images — each labeled with its source
    so the model can cite correctly. Images are downscaled first: full-page
    PNGs sent at native resolution are the single biggest source of
    request latency, and the VLM doesn't need more than ~1000px on the
    long edge to read page content."""
    import PIL.Image

    MAX_DIM = 1400  # a little larger so small table text and axis labels stay readable

    parts = []
    for i, ev in enumerate(evidence):
        chunk = ev.chunk
        label = f"[Source {i+1}: document={chunk.document_id}, page={chunk.page_number}, type={chunk.modality}]"

        if chunk.modality == "image" and chunk.image_path:
            img = PIL.Image.open(chunk.image_path)
            if max(img.size) > MAX_DIM:
                img.thumbnail((MAX_DIM, MAX_DIM), PIL.Image.LANCZOS)
            parts.append(label)
            parts.append(img)
        else:
            parts.append(f"{label}\n{chunk.content}")

    return parts


def _strip_markdown(text: str) -> str:
    """Safety net in case the model still emits markdown: removes ### headings,
    --- rules and **bold** markers. Keeps the inline [n] citation markers and
    the trailing 'Sources:' line, because step_breakdown.py parses those."""
    t = re.sub(r"(?m)^[ \t]{0,3}#{1,6}[ \t]*", "", text)
    t = re.sub(r"(?m)^[ \t]*(?:[-*_=][ \t]*){3,}$", "", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def _call_vlm(question: str, evidence_parts: list) -> str:
    """Tries each key in the pool in turn, starting from the current one.
    On a quota error (429/ResourceExhausted) it rotates to the next key
    and retries immediately; on any other error it fails fast (no point
    rotating keys for a bad request or a network blip). Once a key works,
    future calls keep starting from that key, since it's presumably still
    within quota."""
    global _current_key_index

    pool_size = len(_KEY_POOL)
    last_error: Exception | None = None

    for attempt in range(pool_size):
        index = (_current_key_index + attempt) % pool_size
        _configure_key(index)
        try:
            model = genai.GenerativeModel(
                model_name=settings.gemini_model,
                system_instruction=SYSTEM_PROMPT,
            )
            response = model.generate_content(
                evidence_parts + [f"Question: {question}"],
                generation_config=genai.GenerationConfig(
                    temperature=settings.gemini_temperature,
                    max_output_tokens=2048,
                ),
            )
            _current_key_index = index  # stick with the key that worked
            return _strip_markdown(response.text)
        except ResourceExhausted as exc:
            last_error = exc
            continue  # try the next key in the pool

    # Every key in the pool is out of quota.
    raise AllKeysExhaustedError(
        f"All {pool_size} Gemini API key(s) are currently rate-limited/out of "
        f"quota. Last error: {last_error}"
    ) from last_error


def generate_answer(question: str, evidence: list[RetrievedEvidence]) -> AskResponse:
    if not evidence:
        # No retrieved evidence at all — still let the model answer, but it
        # will fall entirely into the "general knowledge" branch of the
        # system prompt, clearly labeled as such and with no citations.
        raw_answer = _call_vlm(
            question,
            ["[No matching evidence was found in the indexed documents.]"],
        )
        return AskResponse(
            answer=raw_answer,
            citations=[],
            retrieved_chunk_ids=[],
            retrieval_mode="none",
            steps=build_steps(raw_answer, []),
        )

    parts = _build_evidence_parts(evidence)
    raw_answer = _call_vlm(question, parts)

    retrieval_mode = "full_document" if evidence[0].mode == "full_document" else "semantic_topk"

    citations = [
        Citation(
            document_id=ev.chunk.document_id,
            page_number=ev.chunk.page_number,
            modality=ev.chunk.modality,
            confidence=None if ev.mode == "full_document" else distance_to_confidence(ev.score),
            bbox=ev.chunk.bbox,
        )
        for ev in evidence
    ]

    return AskResponse(
        answer=raw_answer,
        citations=citations,
        retrieved_chunk_ids=[ev.chunk.chunk_id for ev in evidence],
        retrieval_mode=retrieval_mode,
        steps=build_steps(raw_answer, citations),
    )