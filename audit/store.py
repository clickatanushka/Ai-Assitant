"""Persistence — everything lives in a single Upstash Vector index.

The index is **dense, Custom, 1536 dimensions, COSINE**. Upstash no longer offers
BAAI/bge-m3 as a hosted model, and it refuses to mix a Custom dense model with its
hosted BM25 sparse one ("either set both models to CUSTOM or set both models to a
pre-defined model"), so dense vectors are computed here with gemini-embedding-2 and
the keyword half is handled by `retrieve.lexical_boost` instead of by BM25.

Two namespaces, so one service covers both jobs:

  ""      chunk vectors — what search runs over. Each page's first chunk also
          carries that page's full text, which is what answers are generated from
          and what citation quotes are verified against.
  "docs"  one small manifest per document (filename, title, page count, blob url).
          Never searched; it exists so the Documents tab can list what is indexed
          without scanning every chunk.

Page text sits on the chunks rather than in the manifest because a whole document
reaches 41 KB in this corpus, uncomfortably close to Upstash's 48 KB metadata cap,
while the largest single page is 3.7 KB.

Vercel Blob holds the original PDFs. Blob *writes* happen in api/blob.ts through
the official JS SDK (there is no documented REST API for other languages); Python
only ever reads a blob back over plain HTTPS, which needs no SDK at all.
"""

import urllib.request

from upstash_vector import Index

from . import config, gemini

_index: Index | None = None

DOC_NS = "docs"          # namespace holding the per-document manifests


def index() -> Index:
    global _index
    if _index is None:
        if not config.UPSTASH_VECTOR_URL or not config.UPSTASH_VECTOR_TOKEN:
            raise RuntimeError("UPSTASH_VECTOR_REST_URL / _TOKEN are not set")
        _index = Index(url=config.UPSTASH_VECTOR_URL, token=config.UPSTASH_VECTOR_TOKEN)
    return _index


def _placeholder_vector() -> list[float]:
    """Manifests still need a vector because the index is dense. They live in their
    own namespace and are never queried, so the value is irrelevant — but an
    all-zero vector has no direction, which a cosine index can reject."""
    return [1.0] + [0.0] * (config.EMBED_DIMS - 1)


# ── Documents ─────────────────────────────────────────────────────────────────

def save_document(doc: dict) -> None:
    index().upsert(
        vectors=[{
            "id": doc["doc_id"],
            "vector": _placeholder_vector(),
            "metadata": {
                "doc_id": doc["doc_id"],
                "file": doc["file"],
                "title": doc["title"],
                "n_pages": len(doc.get("pages", [])),
                "chars": sum(len(p["text"]) for p in doc.get("pages", [])),
                "blob_url": doc.get("blob_url") or "",
                "indexed_at": doc.get("indexed_at") or "",
            },
        }],
        namespace=DOC_NS,
    )


def _manifest(doc_id: str) -> dict:
    found = index().fetch(ids=[doc_id], include_metadata=True, namespace=DOC_NS)
    first = (found or [None])[0]
    return (first.metadata if first else {}) or {}


def get_document(doc_id: str) -> dict | None:
    """Rebuild a document from its chunks.

    Only each page's first chunk carries `page_text`, so pages come back whole
    rather than as overlapping windows.
    """
    results = index().fetch(prefix=f"{doc_id}:", include_metadata=True)
    pages: dict[int, str] = {}
    fallback: dict = {}
    for r in results or []:
        if not r or not r.metadata:
            continue
        fallback = fallback or r.metadata
        text = r.metadata.get("page_text")
        if text:
            pages[int(r.metadata["page"])] = text
    if not pages:
        return None

    m = _manifest(doc_id)
    return {
        "doc_id": doc_id,
        "file": m.get("file") or fallback.get("file", ""),
        "title": m.get("title") or fallback.get("title", ""),
        "blob_url": m.get("blob_url") or None,
        "indexed_at": m.get("indexed_at") or None,
        "pages": [{"page": n, "text": pages[n]} for n in sorted(pages)],
    }


def get_documents(doc_ids: list[str]) -> list[dict]:
    return [d for d in (get_document(i) for i in doc_ids) if d]


def list_documents() -> list[dict]:
    """Summaries for the Documents tab, straight from the manifest namespace."""
    out, cursor = [], ""
    while True:
        page = index().range(cursor=cursor, limit=100, include_metadata=True,
                             namespace=DOC_NS)
        for v in page.vectors or []:
            m = v.metadata or {}
            out.append({
                "doc_id": m.get("doc_id", v.id),
                "file": m.get("file", v.id),
                "title": m.get("title", ""),
                "n_pages": m.get("n_pages", 0),
                "chars": m.get("chars", 0),
                "blob_url": m.get("blob_url") or None,
                "indexed_at": m.get("indexed_at") or None,
            })
        cursor = page.next_cursor or ""
        if not cursor:
            break
    return sorted(out, key=lambda d: d["file"].lower())


def delete_document(doc_id: str) -> bool:
    """Drop a document's chunks and its manifest."""
    existed = bool(_manifest(doc_id))
    index().delete(prefix=f"{doc_id}:")
    index().delete(ids=[doc_id], namespace=DOC_NS)
    return existed


# ── Chunks ────────────────────────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], *, batch_size: int = 50) -> int:
    """Embed with gemini-embedding-2 and upsert. The index uses a Custom dense
    model, so we supply the vector rather than having Upstash derive it."""
    if not chunks:
        return 0
    idx = index()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = gemini.embed_documents([c["embed_text"] for c in batch])
        idx.upsert(vectors=[
            {"id": c["id"], "vector": v, "metadata": c["metadata"]}
            for c, v in zip(batch, vectors)
        ])
    return len(chunks)


def search(query: str, *, top_k: int) -> list[dict]:
    """Dense search over the chunk namespace."""
    results = index().query(
        vector=gemini.embed_query(query),
        top_k=top_k,
        include_metadata=True,
    )
    return [{"id": r.id, "score": r.score, **(r.metadata or {})} for r in results]


def index_stats() -> dict:
    try:
        info = index().info()
        return {"vectors": getattr(info, "vector_count", None)}
    except Exception:
        return {"vectors": None}


def document_count() -> int | None:
    try:
        return len(list_documents())
    except Exception:
        return None


# ── Blob (read-only from Python) ──────────────────────────────────────────────

def fetch_blob(url: str, *, timeout: int = 120, max_bytes: int = 60_000_000) -> bytes:
    """Download a PDF from Blob.

    The store is private, so an unauthenticated request gets a 403; the read-write
    token has to travel as a bearer header. That is the point — the PDFs are
    confidential and must not be anonymously fetchable.
    """
    if not url.startswith("https://"):
        raise ValueError("blob url must be https")
    headers = {"User-Agent": "audit-assistant"}
    if config.BLOB_TOKEN:
        headers["Authorization"] = f"Bearer {config.BLOB_TOKEN}"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"blob larger than {max_bytes} bytes")
    return data
