"""PDF → transcribed pages → chunks → Upstash. Shared by the API and the bulk script.

Every page goes through Gemini vision, including pages that carry an extractable
text layer. Uniform treatment is deliberate: the text layer loses table structure
entirely and never contains the flowchart labels, and 63% of this corpus is
scanned anyway.
"""

import datetime as dt
import hashlib
import io
import re
from concurrent.futures import ThreadPoolExecutor

from pypdf import PdfReader, PdfWriter

from . import chunking, gemini, store


def make_doc_id(filename: str) -> str:
    """Stable id from the filename, so re-uploading a file replaces it in place
    instead of duplicating it."""
    slug = re.sub(r"[^a-z0-9]+", "-", filename.lower()).strip("-")[:48]
    digest = hashlib.sha1(filename.encode("utf-8")).hexdigest()[:8]
    return f"{slug}-{digest}"


def split_pages(pdf_bytes: bytes) -> list[bytes]:
    """One single-page PDF per page.

    Transcribing a page at a time keeps page numbering exact, bounds each response,
    and makes a failure retryable for that page alone. Handing Gemini a whole
    document instead invites a repetition loop — a 3-page test produced 466 KB of
    output before it stopped.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page in reader.pages:
        writer = PdfWriter()
        writer.add_page(page)
        buf = io.BytesIO()
        writer.write(buf)
        pages.append(buf.getvalue())
    return pages


MIN_PAGE_CHARS = 40      # below this a "page" is a header fragment, not a transcription


def transcribe_pdf(pdf_bytes: bytes, *, workers: int = 3,
                   progress=None) -> list[dict]:
    """Transcribe every page of a PDF. Returns [{page, text}] in page order.

    One request for the whole document first — that is 3-4x fewer requests than
    a call per page, which matters because the Gemini free tier limits requests,
    not pages. Pages the document-level pass misses or truncates are then redone
    one at a time, so a partial response degrades into a few extra calls rather
    than a document with holes in it.
    """
    page_pdfs = split_pages(pdf_bytes)
    total = len(page_pdfs)
    if not total:
        return []

    texts: dict[int, str] = {}
    try:
        texts = gemini.transcribe_document(pdf_bytes, total)
    except gemini.GeminiError as e:
        if progress:
            progress(0, total, f"document pass failed ({e}); falling back per page")

    missing = [n for n in range(1, total + 1)
               if len(texts.get(n, "").strip()) < MIN_PAGE_CHARS]
    if progress:
        progress(total - len(missing), total,
                 "document pass" + (f", {len(missing)} page(s) to repair" if missing else ""))

    if missing:
        def repair(number: int):
            try:
                return number, gemini.transcribe_page(page_pdfs[number - 1])
            except gemini.GeminiError as e:
                if progress:
                    progress(number, total, f"page {number} failed: {e}")
                return number, ""

        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            for number, text in pool.map(repair, missing):
                if text.strip():
                    texts[number] = text

    return [{"page": n, "text": texts[n].strip()}
            for n in sorted(texts)
            if texts.get(n, "").strip()]


def ingest_pdf(pdf_bytes: bytes, *, filename: str, blob_url: str | None = None,
               workers: int = 6, progress=None) -> dict:
    """Full pipeline for one PDF. Returns the stored document record."""
    doc_id = make_doc_id(filename)
    pages = transcribe_pdf(pdf_bytes, workers=workers, progress=progress)
    if not pages:
        raise ValueError(f"no text could be transcribed from {filename}")

    title = gemini.extract_title(filename, pages[0]["text"])
    doc = {
        "doc_id": doc_id,
        "file": filename,
        "title": title,
        "pages": pages,
        "blob_url": blob_url,
        "indexed_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }

    # Clear any previous version's chunks before writing the new ones, so a
    # re-upload with fewer pages doesn't leave orphans behind.
    store.index().delete(prefix=f"{doc_id}:")
    chunks = chunking.document_to_chunks(doc)
    store.upsert_chunks(chunks)
    store.save_document(doc)

    doc["n_chunks"] = len(chunks)
    return doc
