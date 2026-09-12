"""All Gemini calls: page transcription, query preparation, reranking, answering.

Uses the native generateContent endpoint (not the OpenAI-compatible one) because
PDF input is only available there. Requests go through urllib so the deployed
function carries no HTTP dependency.

Thinking is disabled on every call. With it on, transcription burns the output
budget on reasoning and then truncates mid-document.
"""

import base64
import json
import re
import threading
import time
import urllib.error
import urllib.request

from . import config


class GeminiError(RuntimeError):
    pass


class QuotaError(GeminiError):
    """A 429. Separated because it means 'wait', not 'this request is bad'."""


class _RateLimiter:
    """Token bucket shared by every thread.

    The free tier rejects bursts with a 429, and the published numbers move, so
    the limit is configurable and enforced client-side rather than discovered
    the hard way in the middle of a long ingest.
    """

    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(per_minute, 1)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = max(0.0, self._next - now)
            self._next = max(now, self._next) + self.interval
        if wait:
            time.sleep(wait)

    def penalise(self, seconds: float) -> None:
        """Push every waiting thread back after a 429."""
        with self._lock:
            self._next = max(self._next, time.monotonic() + seconds)


_limiter = _RateLimiter(config.REQUESTS_PER_MINUTE)


def _retry_delay(payload: str, default: float) -> float:
    """Honour the RetryInfo Gemini returns with a 429 instead of guessing."""
    try:
        error = json.loads(payload).get("error", {})
        for detail in error.get("details", []):
            delay = detail.get("retryDelay")
            if isinstance(delay, str) and delay.endswith("s"):
                return min(float(delay[:-1]) + 1.0, 90.0)
    except (ValueError, TypeError):
        pass
    return default


def _post(body: dict, *, timeout: int = 120, attempts: int = 5) -> dict:
    """POST to generateContent, rate-limited, retrying 429s and transient 5xxs."""
    if not config.GEMINI_API_KEY:
        raise GeminiError("GEMINI_API_KEY is not set")
    url = f"{config.GEMINI_BASE}/{config.GEMINI_MODEL}:generateContent"
    data = json.dumps(body).encode("utf-8")
    last = None
    for attempt in range(attempts):
        _limiter.acquire()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "x-goog-api-key": config.GEMINI_API_KEY},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body_text = e.read().decode("utf-8", "replace")
            last = f"HTTP {e.code}: {body_text[:300]}"
            if e.code == 429 and attempt < attempts - 1:
                wait = _retry_delay(body_text, 8.0 * (attempt + 1))
                _limiter.penalise(wait)
                time.sleep(wait)
                continue
            if e.code == 429:
                raise QuotaError(last) from e
            if e.code in (500, 502, 503) and attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise GeminiError(last) from e
        except (urllib.error.URLError, TimeoutError) as e:
            last = str(e)
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
                continue
            raise GeminiError(last) from e
    raise GeminiError(last or "unreachable")


def _text_of(response: dict) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        raise GeminiError(f"no candidates: {json.dumps(response)[:300]}")
    parts = candidates[0].get("content", {}).get("parts") or []
    return "".join(p.get("text", "") for p in parts)


def _json_call(prompt: str, schema: dict, *, max_tokens: int = 8000,
               timeout: int = 120) -> dict:
    """Structured-output call. responseSchema makes Gemini return parseable JSON,
    which removes the regex salvage the previous implementation needed."""
    response = _post({
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "responseMimeType": "application/json",
            "responseSchema": schema,
            "thinkingConfig": config.THINKING_CONFIG,
        },
    }, timeout=timeout)
    raw = _text_of(response).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        raise GeminiError(f"unparseable JSON: {raw[:300]}") from e


# ── Transcription ─────────────────────────────────────────────────────────────

_RULES = """- Reproduce the German text VERBATIM. Umlauts (ä ö ü Ä Ö Ü) and ß must be exact.
  Never substitute accented letters (é, è, à) for umlauts.
- Preserve headings, numbered sections and bullet lists.
- Render every table as a Markdown table, keeping all cells.
- For each flowchart or diagram, add a section starting with "[Diagramm]" and
  transcribe every box label, arrow label and decision text, then describe the
  flow in German.
- Transcribe document numbers, machine names and part numbers character for character.
- Do NOT translate. Do NOT summarise. Do NOT add commentary."""

_PAGE_PROMPT = f"""Transcribe this single scanned page of a German quality-management document.

{_RULES}
- Output only the page content. Transcribe the page exactly once."""

_DOC_PROMPT = f"""Transcribe this scanned German quality-management document, page by page.

{_RULES}
- Begin every page with a line containing exactly: <<<PAGE n>>>
  where n is the page number, starting at 1.
- Transcribe each page exactly ONCE, in order, then stop."""

PAGE_MARKER = re.compile(r"^<{3}\s*PAGE\s+(\d+)\s*>{3}\s*$", re.MULTILINE)


def _strip_runaway(text: str, *, window: int = 12, limit: int = 3) -> str:
    """Cut a degenerate repetition loop.

    Gemini reliably transcribes the page and then sometimes keeps going, repeating
    a block over and over (one 3-page test produced 466 KB). If any window of
    consecutive lines reappears more than `limit` times, keep everything up to its
    second occurrence and drop the rest.
    """
    lines = text.split("\n")
    if len(lines) < window * 2:
        return text
    seen: dict[str, int] = {}
    for i in range(len(lines) - window + 1):
        key = "\n".join(lines[i:i + window]).strip()
        if not key:
            continue
        seen[key] = seen.get(key, 0) + 1
        if seen[key] > limit:
            return "\n".join(lines[:i]).rstrip()
    return text


def _transcribe(pdf: bytes, prompt: str, *, max_tokens: int) -> str:
    response = _post({
        "contents": [{"parts": [
            {"inline_data": {"mime_type": "application/pdf",
                             "data": base64.b64encode(pdf).decode()}},
            {"text": prompt},
        ]}],
        "generationConfig": {
            "temperature": 0,
            "maxOutputTokens": max_tokens,
            "thinkingConfig": config.THINKING_CONFIG,
        },
    }, timeout=300)
    return _strip_runaway(_text_of(response)).strip()


def transcribe_page(page_pdf: bytes) -> str:
    """Transcribe one single-page PDF."""
    return _transcribe(page_pdf, _PAGE_PROMPT, max_tokens=8000)


def transcribe_document(pdf: bytes, n_pages: int) -> dict[int, str]:
    """Transcribe a whole PDF in one request, returning {page_number: text}.

    One request per document instead of one per page keeps the job inside the
    free tier's request quota. The model reliably transcribes every page but
    sometimes keeps going afterwards, repeating itself — `_strip_runaway` cuts
    that, and any page this misses is re-done individually by the caller.
    """
    text = _transcribe(pdf, _DOC_PROMPT, max_tokens=min(2200 * n_pages + 2000, 60000))

    matches = list(PAGE_MARKER.finditer(text))
    if not matches:
        # No delimiters came back. Only trustworthy as a whole if it is one page.
        return {1: text} if n_pages == 1 and text else {}

    pages: dict[int, str] = {}
    for i, match in enumerate(matches):
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        number = int(match.group(1))
        body = text[match.end():end].strip()
        # Keep the first transcription of a page; a repeat is the runaway loop.
        if body and number not in pages:
            pages[number] = body
    return pages


# ── Query preparation ─────────────────────────────────────────────────────────

_QUERY_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "expanded_en": {"type": "STRING"},
        "query_de": {"type": "STRING"},
    },
    "required": ["expanded_en", "query_de"],
}

_query_cache: dict[str, dict] = {}


def prepare_query(question: str) -> dict:
    """Expand acronyms and translate to German in one call.

    Two search strings come back: an enriched English one and a German one. The
    German string matters most for the BM25 half of the hybrid index, which needs
    the actual German surface forms to match anything.

    Cached per question so the same wording always produces the same search.
    """
    key = question.strip().lower()
    if key in _query_cache:
        return _query_cache[key]
    prompt = (
        "You prepare search queries for a German ISO 9001 / electronics-manufacturing "
        "document search engine.\n\n"
        f"USER QUESTION (English): {question}\n\n"
        "Return:\n"
        "- expanded_en: the question rewritten as a richer English search phrase. Keep the "
        "original words, expand any acronym using general ISO 9001 and electronics-"
        "manufacturing knowledge (e.g. MSL -> Moisture Sensitivity Level, dry storage of "
        "moisture-sensitive components), and add a few closely related synonyms. "
        "Do not invent document names. Max 30 words.\n"
        "- query_de: the German translation of that expanded phrase, using the German "
        "technical vocabulary an electronics manufacturer would actually use "
        "(Lötprozess, Bestückung, Wartungsplan, Prüfung, ...). Max 30 words."
    )
    try:
        result = _json_call(prompt, _QUERY_SCHEMA, max_tokens=500, timeout=30)
        result = {
            "expanded_en": (result.get("expanded_en") or question).strip(),
            "query_de": (result.get("query_de") or question).strip(),
        }
    except GeminiError:
        # Search still works on the raw question; degrade rather than fail.
        result = {"expanded_en": question, "query_de": question}
    _query_cache[key] = result
    return result


# ── Reranking ─────────────────────────────────────────────────────────────────

_RERANK_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "scores": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "doc_id": {"type": "STRING"},
                    "relevance": {"type": "INTEGER"},
                },
                "required": ["doc_id", "relevance"],
            },
        }
    },
    "required": ["scores"],
}


def rerank(question: str, candidates: list[dict]) -> dict[str, int]:
    """Score each candidate document 0-10 for whether it can answer the question.

    Replaces the previous fixed `top_score - 0.05` cosine cutoff, which had no
    notion of whether the best match was actually relevant — only of whether it
    beat the runner-up. Returns {doc_id: relevance}.
    """
    if not candidates:
        return {}
    blocks = []
    for c in candidates:
        excerpt = c["excerpt"][:1200]
        blocks.append(f"### doc_id: {c['doc_id']}\nTitel: {c['title']}\n\n{excerpt}")
    prompt = (
        "Rate how well each German document can answer the question.\n\n"
        f"QUESTION (English): {question}\n\n"
        "DOCUMENTS:\n\n" + "\n\n---\n\n".join(blocks) + "\n\n"
        "For every doc_id give relevance 0-10:\n"
        "  10 = directly and specifically answers the question\n"
        "   7 = clearly on topic and contains part of the answer\n"
        "   4 = same general subject area but does not answer it\n"
        "   0 = unrelated\n"
        "Judge only the text shown. Score every doc_id exactly once."
    )
    try:
        result = _json_call(prompt, _RERANK_SCHEMA, max_tokens=1000, timeout=45)
    except GeminiError:
        # Fall back to retrieval order rather than dropping the question.
        return {c["doc_id"]: config.RERANK_MIN_SCORE for c in candidates}
    valid = {c["doc_id"] for c in candidates}
    return {
        s["doc_id"]: max(0, min(10, int(s.get("relevance", 0))))
        for s in result.get("scores", [])
        if s.get("doc_id") in valid
    }


# ── Answering ─────────────────────────────────────────────────────────────────

_ANSWER_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "found": {"type": "BOOLEAN"},
        "answer_german": {"type": "STRING"},
        "answer_english": {"type": "STRING"},
        "citations": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "file": {"type": "STRING"},
                    "page": {"type": "INTEGER"},
                    "quote_german": {"type": "STRING"},
                    "quote_english": {"type": "STRING"},
                },
                "required": ["file", "page", "quote_german", "quote_english"],
            },
        },
        "confidence": {"type": "STRING", "enum": ["high", "medium", "low"]},
    },
    "required": ["found", "answer_german", "answer_english", "citations", "confidence"],
}


def answer(question: str, documents: list[dict]) -> dict:
    """Answer from the FULL text of the selected documents.

    The previous implementation truncated each excerpt to 350 characters, so the
    model routinely answered from a third of the evidence. These documents are
    short (median 2 pages) — the whole thing fits comfortably in context.
    """
    blocks = []
    for doc in documents:
        pages = "\n\n".join(
            f"--- {doc['file']} | Seite {p['page']} ---\n{p['text']}"
            for p in doc["pages"]
        )
        blocks.append(f"DOKUMENT: {doc['file']}\nTitel: {doc['title']}\n\n{pages}")
    corpus = "\n\n============================\n\n".join(blocks)

    prompt = (
        "You are an ISO 9001 audit assistant. Answer strictly from the German "
        "documents below. They are the complete text of the relevant documents.\n\n"
        f"QUESTION (English): {question}\n\n"
        f"DOCUMENTS:\n\n{corpus}\n\n"
        "Instructions:\n"
        "- If the documents do not contain the answer, set found=false and leave the "
        "answers empty. Never guess and never use outside knowledge.\n"
        "- answer_german: the answer in German, for the auditor.\n"
        "- answer_english: the same answer in English. Both must state the same facts.\n"
        "- Be specific: name the machines, materials, intervals, tolerances and "
        "document numbers the text gives.\n"
        "- citations: one entry per passage you actually relied on (1-6 entries).\n"
        "  * file: copy the DOKUMENT filename exactly.\n"
        "  * page: the Seite number the passage came from.\n"
        "  * quote_german: copied CHARACTER FOR CHARACTER from that page above. "
        "Do not paraphrase, correct, reformat or shorten it with ellipses. "
        "One to three sentences. This is checked against the source automatically "
        "and a quote that does not match will be discarded.\n"
        "  * quote_english: your English translation of that exact quote.\n"
        "- confidence: high if the documents state the answer explicitly, medium if "
        "it must be inferred, low if the evidence is thin."
    )
    return _json_call(prompt, _ANSWER_SCHEMA, max_tokens=8000, timeout=180)


def extract_title(filename: str, first_page: str) -> str:
    """Human-readable document title from the filename, e.g.
    'AA 6003 - Prozess verbleites Löten.pdf' -> 'AA 6003 - Prozess verbleites Löten'.
    Bracketed ISO clause lists in the filename are dropped — they are metadata, not
    a title, and they pollute the embedding."""
    name = re.sub(r"\.pdf$", "", filename, flags=re.IGNORECASE)
    name = re.sub(r"\s*\([0-9._\s&Kapitel]+\)\s*$", "", name)
    return name.strip() or filename
