"""The question → answer pipeline.

Shape of it:
    prepare query (EN + DE) → two hybrid searches → RRF fusion → group by document
    → LLM rerank → full text of the winners → answer → verify every quote

Two things here are what make answers land on the right document and stay honest:
the German query variant (BM25 cannot match German text from English tokens), and
the rerank threshold, which lets the system say "not in these documents" instead of
always returning its least-bad guess.
"""

from . import config, gemini, store, verify


def _rrf_fuse(result_lists: list[list[dict]], k: int = config.RRF_K) -> list[dict]:
    """Reciprocal rank fusion across the English and German result lists.

    Rank-based rather than score-based on purpose: the two queries produce scores
    on different scales, and RRF only cares about ordering.
    """
    fused: dict[str, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            entry = fused.get(item["id"])
            if entry is None:
                entry = fused[item["id"]] = dict(item, rrf=0.0)
            entry["rrf"] += 1.0 / (k + rank + 1)
    return sorted(fused.values(), key=lambda c: -c["rrf"])


def _group_by_document(chunks: list[dict]) -> list[dict]:
    """Collapse chunks to documents, keeping each document's best chunk as the
    excerpt the reranker will judge it on."""
    docs: dict[str, dict] = {}
    for chunk in chunks:
        doc_id = chunk.get("doc_id")
        if not doc_id:
            continue
        entry = docs.get(doc_id)
        if entry is None:
            docs[doc_id] = {
                "doc_id": doc_id,
                "file": chunk.get("file", ""),
                "title": chunk.get("title", ""),
                "score": chunk["rrf"],
                "excerpt": chunk.get("text", ""),
                "pages": {chunk.get("page")},
            }
        else:
            entry["score"] = max(entry["score"], chunk["rrf"])
            entry["pages"].add(chunk.get("page"))
            if len(entry["excerpt"]) < 900:
                entry["excerpt"] += "\n\n" + chunk.get("text", "")
    ranked = sorted(docs.values(), key=lambda d: -d["score"])
    for doc in ranked:
        doc["pages"] = sorted(p for p in doc["pages"] if p is not None)
    return ranked


def retrieve(question: str) -> dict:
    """Everything up to (not including) answering. Separated so the eval harness
    can measure retrieval on its own."""
    prepared = gemini.prepare_query(question)

    searches = []
    for query in (prepared["expanded_en"], prepared["query_de"]):
        try:
            searches.append(store.search(query, top_k=config.SEARCH_TOP_K))
        except Exception:
            searches.append([])
    if not any(searches):
        return {"query": prepared, "candidates": [], "selected": [], "scores": {}}

    candidates = _group_by_document(_rrf_fuse(searches))[:config.RERANK_CANDIDATES]
    scores = gemini.rerank(question, candidates)

    for doc in candidates:
        doc["relevance"] = scores.get(doc["doc_id"], 0)
    ordered = sorted(candidates, key=lambda d: (-d["relevance"], -d["score"]))
    selected = [d for d in ordered if d["relevance"] >= config.RERANK_MIN_SCORE]

    return {
        "query": prepared,
        "candidates": ordered,
        "selected": selected[:config.MAX_ANSWER_DOCS],
        "scores": scores,
    }


NOT_FOUND = {
    "found": False,
    "answer_english": (
        "No document in the indexed set contains an answer to this question."
    ),
    "answer_german": (
        "Keines der indizierten Dokumente enthält eine Antwort auf diese Frage."
    ),
    "citations": [],
    "confidence": "high",
}


def ask(question: str) -> dict:
    """Retrieve, answer from full documents, and verify every quote."""
    retrieval = retrieve(question)
    selected = retrieval["selected"]

    debug = {
        "expanded_en": retrieval["query"]["expanded_en"],
        "query_de": retrieval["query"]["query_de"],
        "considered": [
            {"file": d["file"], "relevance": d["relevance"]}
            for d in retrieval["candidates"]
        ],
    }

    if not selected:
        # Nothing cleared the relevance bar. Saying so is more useful to an auditor
        # than an answer assembled from the closest unrelated document.
        return {**NOT_FOUND, "documents": [], "debug": debug}

    documents = store.get_documents([d["doc_id"] for d in selected])
    if not documents:
        return {**NOT_FOUND, "documents": [], "debug": debug}

    result = gemini.answer(question, documents)
    result = verify.verify_answer(result, documents)

    if not result.get("found"):
        result.setdefault("answer_english", NOT_FOUND["answer_english"])
        result.setdefault("answer_german", NOT_FOUND["answer_german"])

    result["documents"] = [
        {"file": d["file"], "title": d["title"], "blob_url": d.get("blob_url"),
         "relevance": next((s["relevance"] for s in selected
                            if s["doc_id"] == d["doc_id"]), None)}
        for d in documents
    ]
    result["debug"] = debug
    return result
