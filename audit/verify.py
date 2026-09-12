"""Check that every quoted German sentence really appears in the cited page.

This is what makes "exact" enforceable. The model is told to copy quotes verbatim;
here we confirm it did. Anything unverifiable is dropped rather than shown to an
auditor as if it came out of the document.
"""

import difflib
import re
import unicodedata

from . import config


# Differences that don't change meaning: quote styles, dash styles, soft hyphens.
# Umlauts are deliberately NOT folded — a wrong umlaut is a real mismatch, and
# catching those is half the point of this check.
_CHAR_MAP = {
    "„": '"', "“": '"', "”": '"', "«": '"', "»": '"',
    "‘": "'", "’": "'", "‚": "'",
    "–": "-", "—": "-", "‐": "-",
    "­": "",   # soft hyphen
    "​": "",   # zero-width space
}


def _normalise(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = "".join(_CHAR_MAP.get(c, c) for c in text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _normalise_mapped(text: str) -> tuple[str, list[int]]:
    """Normalise while recording, for each output character, its index in `text`.

    The offset map is what lets a match found in normalised space be reported back
    as the untouched source substring — quotes are often a fragment of a sentence,
    so sentence-level matching alone misses them.
    """
    text = unicodedata.normalize("NFC", text)
    out: list[str] = []
    offsets: list[int] = []
    previous_space = True
    for i, char in enumerate(text):
        mapped = _CHAR_MAP.get(char, char)
        if mapped == "":
            continue
        if mapped.isspace():
            if previous_space:
                continue
            out.append(" ")
            offsets.append(i)
            previous_space = True
            continue
        out.append(mapped.casefold())
        offsets.append(i)
        previous_space = False
    while out and out[-1] == " ":
        out.pop()
        offsets.pop()
    return "".join(out), offsets


def _expand_to_sentence(text: str, start: int, end: int) -> str:
    """Grow a span outwards to the nearest sentence boundaries, so what we show an
    auditor reads as a sentence rather than starting mid-word."""
    left = max((text.rfind(c, 0, start) for c in ".:;!?\n"), default=-1)
    start = left + 1 if left != -1 else 0
    right = min((p for p in (text.find(c, end) for c in ".:;!?\n") if p != -1),
                default=-1)
    end = right + 1 if right != -1 else len(text)
    return text[start:end].strip()


def _best_span(quote: str, page_text: str) -> tuple[float, str]:
    """Closest match for `quote` anywhere in `page_text`.

    Returns (ratio, original_substring). A sliding character window handles quotes
    that are fragments of a sentence; the match is then widened to sentence
    boundaries for display.
    """
    q, _ = _normalise_mapped(quote)
    p, offsets = _normalise_mapped(page_text)
    if not q or not p:
        return 0.0, ""

    if len(p) <= len(q):
        ratio = difflib.SequenceMatcher(None, q, p).ratio()
        return ratio, page_text.strip()

    best_ratio, best_start = 0.0, 0
    step = max(1, len(q) // 8)
    matcher = difflib.SequenceMatcher()
    matcher.set_seq2(q)
    for start in range(0, len(p) - len(q) + step, step):
        matcher.set_seq1(p[start:start + len(q)])
        ratio = matcher.ratio()
        if ratio > best_ratio:
            best_ratio, best_start = ratio, start
            if best_ratio >= 0.995:
                break

    end = min(best_start + len(q), len(offsets) - 1)
    return best_ratio, _expand_to_sentence(page_text, offsets[best_start], offsets[end])


def verify_citation(quote: str, page_text: str) -> tuple[str, float, str]:
    """Return (status, score, source_text).

    status is 'exact', 'fuzzy' or 'unverified'. For a fuzzy match `source_text` is
    the passage as the document actually words it; for an exact match it is the
    quote itself.
    """
    q, p = _normalise(quote), _normalise(page_text)
    if not q:
        return "unverified", 0.0, ""
    if q in p:
        return "exact", 1.0, quote
    ratio, span = _best_span(quote, page_text)
    if ratio >= config.QUOTE_FUZZY_RATIO:
        return "fuzzy", ratio, span
    return "unverified", ratio, span


def verify_answer(result: dict, documents: list[dict]) -> dict:
    """Annotate each citation with its verification status and drop the ones that
    cannot be traced back to the page they claim. Mutates and returns `result`."""
    pages = {
        (doc["file"], p["page"]): p["text"]
        for doc in documents for p in doc["pages"]
    }

    kept, dropped = [], 0
    for citation in result.get("citations", []) or []:
        page_text = pages.get((citation.get("file"), citation.get("page")))
        if page_text is None:
            # Cited a file/page that was never supplied — a fabricated reference.
            dropped += 1
            continue
        quote = citation.get("quote_german", "")
        status, score, source_text = verify_citation(quote, page_text)
        if status == "unverified":
            dropped += 1
            continue
        if status == "fuzzy" and source_text:
            # Show what the document says, not the model's near-miss of it. The
            # model's version is kept so the difference stays inspectable.
            citation["quote_german_as_written"] = quote
            citation["quote_german"] = source_text
        citation["verified"] = status
        citation["match_score"] = round(score, 3)
        kept.append(citation)

    for i, citation in enumerate(kept, 1):
        citation["source_num"] = i
    result["citations"] = kept
    result["citations_dropped"] = dropped

    if dropped and not kept:
        result["warning"] = (
            "None of the model's quotes could be matched to the source documents. "
            "Treat this answer as unverified."
        )
        result["confidence"] = "low"
    elif dropped:
        result["warning"] = (
            f"{dropped} quote(s) did not match the source text and were removed."
        )
    return result
