"""Turn transcribed pages into embeddable chunks.

The previous scheme cut every page into 100-word windows regardless of length,
which split short documents mid-sentence for no reason. Here a page that fits
becomes a single chunk, so the embedding sees a complete thought.
"""

from . import config


def _windows(words: list[str]) -> list[list[str]]:
    step = config.CHUNK_WINDOW - config.CHUNK_OVERLAP
    out = []
    for start in range(0, len(words), step):
        window = words[start:start + config.CHUNK_WINDOW]
        if len(window) < 20 and out:      # trailing scrap folds into the previous window
            break
        out.append(window)
        if start + config.CHUNK_WINDOW >= len(words):
            break
    return out


def page_to_chunks(text: str, *, doc_id: str, file: str, title: str,
                   page: int, n_pages: int) -> list[dict]:
    """Split one page into chunk records ready for upsert."""
    words = text.split()
    if len(words) < 10:
        return []

    groups = [words] if len(words) <= config.CHUNK_MAX_WORDS else _windows(words)

    chunks = []
    for i, group in enumerate(groups):
        body = " ".join(group)
        metadata_extra = {}
        if i == 0:
            # The page's full text rides on its first chunk. Answers are generated
            # from whole pages and citation quotes are verified against them, so
            # the text has to survive somewhere that is not an overlapping window.
            metadata_extra["page_text"] = text.strip()
        chunks.append({
            "id": f"{doc_id}:p{page}:c{i}",
            # The embedded string carries the title and page so a query naming the
            # document ("Wartungsplan SIPLACE") matches even when the body never
            # repeats the title. Generalises the filename-prepending the old
            # build_embeddings() did, and gives BM25 the title tokens too.
            "embed_text": f"{title}\nSeite {page} von {n_pages}\n\n{body}",
            "metadata": {
                "doc_id": doc_id,
                "file": file,
                "title": title,
                "page": page,
                "n_pages": n_pages,
                "text": body,
                **metadata_extra,
            },
        })
    return chunks


def document_to_chunks(doc: dict) -> list[dict]:
    """All chunks for a stored document record."""
    chunks = []
    n_pages = len(doc["pages"])
    for page in doc["pages"]:
        chunks.extend(page_to_chunks(
            page["text"],
            doc_id=doc["doc_id"], file=doc["file"], title=doc["title"],
            page=page["page"], n_pages=n_pages,
        ))
    return chunks
