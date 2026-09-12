"""Persistence: Upstash Vector for search, Upstash Redis for document text.

Upstash Vector holds a HYBRID index: a Custom dense half plus Upstash's hosted BM25
sparse half. Upstash no longer offers BAAI/bge-m3 (the model the old local build
used), so dense vectors are computed here with gemini-embedding-2 and upserted
directly, while the raw text still goes along for BM25 to index. Measured 5/5 on
English-question -> German-document retrieval over this corpus, which is the
property everything else depends on.

Redis holds the full transcription of every document. Answers are generated from
complete documents rather than fragments, and citation quotes are verified against
the exact page text stored here.

Vercel Blob holds the original PDFs. Blob *writes* happen in api/blob.ts through
the official JS SDK (there is no documented REST API for other languages); Python
only ever reads a blob back over plain HTTPS, which needs no SDK at all.
"""

import json
import urllib.request

from upstash_redis import Redis
from upstash_vector import Index
from upstash_vector.types import FusionAlgorithm

from . import config, gemini

_index: Index | None = None
_redis: Redis | None = None

DOC_SET = "docs"          # Redis set of every doc_id
DOC_KEY = "doc:{}"        # Redis key holding one document record


def index() -> Index:
    global _index
    if _index is None:
        if not config.UPSTASH_VECTOR_URL or not config.UPSTASH_VECTOR_TOKEN:
            raise RuntimeError("UPSTASH_VECTOR_REST_URL / _TOKEN are not set")
        _index = Index(url=config.UPSTASH_VECTOR_URL, token=config.UPSTASH_VECTOR_TOKEN)
    return _index


def redis() -> Redis:
    global _redis
    if _redis is None:
        if not config.UPSTASH_REDIS_URL or not config.UPSTASH_REDIS_TOKEN:
            raise RuntimeError("UPSTASH_REDIS_REST_URL / _TOKEN are not set")
        _redis = Redis(url=config.UPSTASH_REDIS_URL, token=config.UPSTASH_REDIS_TOKEN)
    return _redis


# ── Documents (Redis) ─────────────────────────────────────────────────────────

def save_document(doc: dict) -> None:
    r = redis()
    r.set(DOC_KEY.format(doc["doc_id"]), json.dumps(doc, ensure_ascii=False))
    r.sadd(DOC_SET, doc["doc_id"])


def get_document(doc_id: str) -> dict | None:
    raw = redis().get(DOC_KEY.format(doc_id))
    return json.loads(raw) if raw else None


def get_documents(doc_ids: list[str]) -> list[dict]:
    if not doc_ids:
        return []
    keys = [DOC_KEY.format(d) for d in doc_ids]
    raws = redis().mget(*keys)
    return [json.loads(r) for r in raws if r]


def list_documents() -> list[dict]:
    """Summaries for the Documents tab — no page text, which would be megabytes."""
    doc_ids = redis().smembers(DOC_SET) or []
    docs = get_documents(sorted(doc_ids))
    return sorted((
        {
            "doc_id": d["doc_id"],
            "file": d["file"],
            "title": d["title"],
            "n_pages": len(d.get("pages", [])),
            "chars": sum(len(p["text"]) for p in d.get("pages", [])),
            "blob_url": d.get("blob_url"),
            "indexed_at": d.get("indexed_at"),
        }
        for d in docs
    ), key=lambda d: d["file"].lower())


def delete_document(doc_id: str) -> bool:
    """Remove a document and every chunk it produced.

    Chunk ids are all prefixed `{doc_id}:`, so one prefix delete clears the vector
    side without having to reconstruct individual ids."""
    doc = get_document(doc_id)
    index().delete(prefix=f"{doc_id}:")
    r = redis()
    r.delete(DOC_KEY.format(doc_id))
    r.srem(DOC_SET, doc_id)
    return doc is not None


# ── Chunks (Upstash Vector) ───────────────────────────────────────────────────

def upsert_chunks(chunks: list[dict], *, batch_size: int = 50) -> int:
    """Upsert chunks with dense vectors we compute ourselves.

    The index is hybrid with a Custom dense model, so the vector is supplied here
    while `data` is still sent for Upstash to build the BM25 sparse side from.
    Upstash no longer offers BAAI/bge-m3 as a hosted model, which is why the dense
    half moved to gemini-embedding-2 on our side.
    """
    if not chunks:
        return 0
    idx = index()
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start:start + batch_size]
        vectors = gemini.embed_documents([c["embed_text"] for c in batch])
        idx.upsert(vectors=[
            {"id": c["id"], "vector": v, "data": c["embed_text"],
             "metadata": c["metadata"]}
            for c, v in zip(batch, vectors)
        ])
    return len(chunks)


def search(query: str, *, top_k: int) -> list[dict]:
    """One hybrid query: our dense vector plus Upstash's BM25, fused with RRF."""
    results = index().query(
        vector=gemini.embed_query(query),
        data=query,                       # drives the BM25 sparse half
        top_k=top_k,
        include_metadata=True,
        fusion_algorithm=FusionAlgorithm.RRF,
    )
    return [
        {"id": r.id, "score": r.score, **(r.metadata or {})}
        for r in results
    ]


def index_stats() -> dict:
    try:
        info = index().info()
        return {"vectors": getattr(info, "vector_count", None)}
    except Exception:
        return {"vectors": None}


# ── Blob (read-only from Python) ──────────────────────────────────────────────

def fetch_blob(url: str, *, timeout: int = 120, max_bytes: int = 60_000_000) -> bytes:
    """Download a PDF that api/blob.ts (or the Vercel CLI) already stored."""
    if not url.startswith("https://"):
        raise ValueError("blob url must be https")
    req = urllib.request.Request(url, headers={"User-Agent": "audit-assistant"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = resp.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ValueError(f"blob larger than {max_bytes} bytes")
    return data
