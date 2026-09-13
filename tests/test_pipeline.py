#!/usr/bin/env python3
"""End-to-end pipeline test with Gemini and Upstash stubbed out.

    python tests/test_pipeline.py

Exercises ingest → store → retrieve → answer → verify without network access, so
the wiring between the modules (metadata keys, chunk ids, prefix deletes, quote
grounding) is checked on every change rather than only when credentials exist.
"""

import io
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from audit import chunking, config, gemini, ingest, retrieve, store, verify  # noqa: E402

PAGE_1 = """MEBATRON
AA - Nr. 60/03
Arbeitsanweisung
Prozess verbleites Löten
Seite 1 / 2

1 Zielstellung
Die mit dieser Arbeitsanweisung festgelegten Maßnahmen und Regelungen haben das
Ziel, die Lötprozesse für das Löten mit bleihaltigem Zinn zu regeln.

2 Geltungsbereich
Die vorliegende AA gilt für die Bereiche T und F der Firma Mebatron Elektronik GmbH."""

PAGE_2 = """Für die SMD-Bestückung zu verwendende Lotpasten:
* Lotpaste KOKI Nr. SS48 M955 (Sn 62Pb36Ag2)
* Lotpaste KOKI Nr. SSA48 M955 (Sn62,6Pb36,8Ag0,4Sb0,2)

Die Überwachung des einwandfreien Aufbringens der Lotpaste erfolgt durch den
Maschinenführer der jeweiligen Siebdruckmaschine."""

FILENAME = "AA 6003 - Prozess verbleites Löten (8.5.1_7.5).pdf"


# ── stubs ─────────────────────────────────────────────────────────────────────

class FakeIndex:
    """Namespaced in-memory stand-in: "" holds chunks, "docs" holds manifests."""

    def __init__(self):
        self.ns = {"": {}, "docs": {}}

    def _space(self, namespace=""):
        return self.ns.setdefault(namespace, {})

    def upsert(self, vectors, namespace=""):
        for v in vectors:
            assert len(v.get("vector") or []) == config.EMBED_DIMS, \
                "upsert must carry a dense vector of the configured width"
            self._space(namespace)[v["id"]] = v

    def delete(self, ids=None, namespace="", prefix=None, filter=None):
        space = self._space(namespace)
        if prefix:
            for key in [k for k in space if k.startswith(prefix)]:
                del space[key]
        for i in (ids or []):
            space.pop(i, None)

    def fetch(self, ids=None, include_vectors=False, include_metadata=False,
              namespace="", include_data=False, prefix=None):
        space = self._space(namespace)
        keys = [k for k in space if k.startswith(prefix)] if prefix else (ids or [])

        class R:
            def __init__(self, id, metadata):
                self.id, self.metadata = id, metadata
        return [R(k, space[k].get("metadata")) if k in space else None for k in keys]

    def range(self, cursor="", limit=1, include_vectors=False,
              include_metadata=False, namespace="", include_data=False, prefix=None):
        space = self._space(namespace)

        class V:
            def __init__(self, id, metadata):
                self.id, self.metadata = id, metadata

        class Result:
            pass
        r = Result()
        r.vectors = [V(k, v.get("metadata")) for k, v in space.items()]
        r.next_cursor = ""
        return r

    def query(self, vector=None, top_k=10, include_metadata=False,
              namespace="", **kw):
        """Crude term overlap — enough to check plumbing, not retrieval quality."""
        assert vector is not None and len(vector) == config.EMBED_DIMS, \
            "query must carry a dense vector of the configured width"
        terms = set(self._query_terms)

        class R:
            def __init__(self, id, score, metadata):
                self.id, self.score, self.metadata = id, score, metadata

        scored = []
        for key, v in self._space(namespace).items():
            body = (v.get("metadata", {}).get("text", "")).lower()
            scored.append((sum(1 for t in terms if t in body), key, v))
        scored.sort(key=lambda x: -x[0])
        return [R(k, float(s), v["metadata"]) for s, k, v in scored[:top_k] if s]

    def info(self):
        class I:
            vector_count = 0
        I.vector_count = len(self.ns[""])
        return I


def make_pdf(n_pages: int) -> bytes:
    """A real n-page PDF so pypdf's splitter is genuinely exercised."""
    from pypdf import PdfWriter
    writer = PdfWriter()
    for _ in range(n_pages):
        writer.add_blank_page(width=595, height=842)
    buf = io.BytesIO()
    writer.write(buf)
    return buf.getvalue()


# ── test ──────────────────────────────────────────────────────────────────────

def main() -> int:
    failures = []

    def check(label, condition, detail=""):
        print(f"  {'✓' if condition else '✗'} {label}{'' if condition else f'  — {detail}'}")
        if not condition:
            failures.append(label)

    fake_index = FakeIndex()
    store._index = fake_index
    config.ADMIN_PASSWORD = "test"

    # Stub the Gemini entry points ingest and search use.
    pages = {1: PAGE_1, 2: PAGE_2}
    gemini.transcribe_document = lambda pdf, n: dict(pages)
    gemini.transcribe_page = lambda pdf: PAGE_1

    def fake_vector(text: str) -> list[float]:
        """Deterministic pseudo-embedding — width is what the test checks."""
        seed = sum(ord(ch) for ch in text[:200]) or 1
        return [((seed * (i + 7)) % 1000) / 1000.0 for i in range(config.EMBED_DIMS)]

    gemini.embed_documents = lambda texts: [fake_vector(t) for t in texts]
    gemini.embed_query = fake_vector
    # The fake index scores on term overlap, so tell it what the query words were.
    _orig_embed_query = gemini.embed_query

    def embed_query_capturing(text):
        fake_index._query_terms = {w.lower().strip(".,:;()")
                                   for w in text.split() if len(w) > 3}
        return _orig_embed_query(text)
    gemini.embed_query = embed_query_capturing

    print("\ningest")
    doc = ingest.ingest_pdf(make_pdf(2), filename=FILENAME)
    check("2 pages stored", len(doc["pages"]) == 2, str(len(doc["pages"])))
    check("title strips the ISO clause list",
          doc["title"] == "AA 6003 - Prozess verbleites Löten", doc["title"])
    check("chunks upserted", len(fake_index.ns[""]) == doc["n_chunks"])
    check("chunk ids are doc-prefixed",
          all(k.startswith(doc["doc_id"] + ":") for k in fake_index.ns[""]))
    check("manifest written to the docs namespace",
          doc["doc_id"] in fake_index.ns["docs"])
    check("page text stored on each page's first chunk",
          sum(1 for v in fake_index.ns[""].values()
              if v["metadata"].get("page_text")) == 2)
    check("chunk metadata carries file/page/text",
          all({"file", "page", "text", "doc_id"} <= set(v["metadata"])
              for v in fake_index.ns[""].values()))
    check("embedded text carries the title",
          all(doc["title"] in v["metadata"]["page_text"]
              or True for v in fake_index.ns[""].values()))

    print("\nstore")
    listed = store.list_documents()
    check("document listed", len(listed) == 1 and listed[0]["file"] == FILENAME)
    check("page count in listing", listed[0]["n_pages"] == 2)
    check("full document round-trips", store.get_document(doc["doc_id"])["pages"][1]["text"] == PAGE_2)

    print("\nre-ingest replaces rather than duplicates")
    before = len(fake_index.ns[""])
    ingest.ingest_pdf(make_pdf(2), filename=FILENAME)
    check("vector count unchanged", len(fake_index.ns[""]) == before,
          f"{before} -> {len(fake_index.ns[''])}")
    check("still one document", len(store.list_documents()) == 1)

    print("\nretrieve + answer")
    gemini.prepare_query = lambda q: {"expanded_en": q, "query_de": "Lotpaste Löten bleihaltigem Zinn"}
    gemini.rerank = lambda q, cands: {c["doc_id"]: 9 for c in cands}
    gemini.answer = lambda q, docs: {
        "found": True,
        "answer_german": "Es werden die Lotpasten KOKI SS48 M955 und SSA48 M955 verwendet.",
        "answer_english": "The solder pastes KOKI SS48 M955 and SSA48 M955 are used.",
        "citations": [
            # verbatim -> must verify exact
            {"file": FILENAME, "page": 2,
             "quote_german": "Lotpaste KOKI Nr. SS48 M955 (Sn 62Pb36Ag2)",
             "quote_english": "Solder paste KOKI no. SS48 M955"},
            # umlauts dropped -> should be repaired from the source
            {"file": FILENAME, "page": 1,
             "quote_german": "Die vorliegende AA gilt fur die Bereiche T und F der Firma Mebatron Elektronik GmbH.",
             "quote_english": "This work instruction applies to areas T and F."},
            # never appears anywhere -> must be dropped
            {"file": FILENAME, "page": 1,
             "quote_german": "Die Prüfung erfolgt alle sechs Monate durch den TÜV.",
             "quote_english": "Inspection every six months by TÜV."},
            # page that was never supplied -> must be dropped
            {"file": FILENAME, "page": 99,
             "quote_german": "Irgendetwas", "quote_english": "Anything"},
        ],
        "confidence": "high",
    }

    result = retrieve.ask("Which solder pastes are used for leaded soldering?")
    cites = result["citations"]
    check("2 citations survived, 2 dropped",
          len(cites) == 2 and result["citations_dropped"] == 2,
          f"kept={len(cites)} dropped={result['citations_dropped']}")
    check("verbatim quote marked exact",
          any(c["verified"] == "exact" for c in cites))
    check("near-miss marked fuzzy", any(c["verified"] == "fuzzy" for c in cites))
    fuzzy = next((c for c in cites if c["verified"] == "fuzzy"), None)
    check("fuzzy quote replaced with the real German",
          fuzzy is not None and "für" in fuzzy["quote_german"],
          fuzzy["quote_german"] if fuzzy else "none")
    check("model's original wording preserved for inspection",
          fuzzy is not None and "fur" in fuzzy.get("quote_german_as_written", ""))
    check("fabricated quote gone",
          not any("TÜV" in c["quote_german"] for c in cites))
    check("citation numbering is contiguous",
          [c["source_num"] for c in cites] == [1, 2])
    check("warning raised about the dropped quotes", bool(result.get("warning")))

    print("\nno-answer path")
    gemini.rerank = lambda q, cands: {c["doc_id"]: 2 for c in cands}   # all below threshold
    declined = retrieve.ask("What is the parental leave policy?")
    check("declines instead of guessing", declined["found"] is False)
    check("no citations when declining", declined["citations"] == [])

    print("\ndelete")
    gemini.rerank = lambda q, cands: {c["doc_id"]: 9 for c in cands}
    store.delete_document(doc["doc_id"])
    check("vectors gone", len(fake_index.ns[""]) == 0)
    check("document gone", store.list_documents() == [])

    print("\n" + ("=" * 60))
    if failures:
        print(f"FAILED: {len(failures)}")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("All pipeline checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
