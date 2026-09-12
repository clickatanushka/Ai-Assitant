#!/usr/bin/env python3
"""One-time bulk load of the existing PDF corpus.

    # Step 1 — transcribe every PDF with Gemini vision into a local cache.
    #          Needs only GEMINI_API_KEY. Safe to re-run; finished files are skipped.
    python scripts/bulk_ingest.py transcribe

    # Step 2 — push the cache into Upstash Vector + Redis.
    #          Needs the Upstash variables too. Cheap and fast, no Gemini calls.
    python scripts/bulk_ingest.py push

    # Or do both at once:
    python scripts/bulk_ingest.py all

Transcription is the slow, paid step, so it is cached separately from the upload.
Getting the Upstash credentials wrong should never mean paying to transcribe twice.
"""

import argparse
import json
import os
import pathlib
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def load_env() -> None:
    """Read .env if present; real environment variables win."""
    env_path = ROOT / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip())


load_env()

PDF_DIR = ROOT / "pdfs"
CACHE = ROOT / "data" / "transcriptions.json"
_lock = threading.Lock()


def find_pdfs() -> list[pathlib.Path]:
    return sorted(
        (p for p in PDF_DIR.rglob("*.pdf") if not p.name.startswith("~$")),
        key=lambda p: p.name.lower(),
    )


def load_cache() -> dict:
    if CACHE.exists():
        return json.loads(CACHE.read_text(encoding="utf-8"))
    return {}


def save_cache(cache: dict) -> None:
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    tmp = CACHE.with_suffix(".tmp")
    tmp.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(CACHE)


def cmd_transcribe(args) -> int:
    from audit import gemini, ingest

    pdfs = find_pdfs()
    cache = load_cache()
    todo = [p for p in pdfs if p.name not in cache or args.force]
    print(f"{len(pdfs)} PDFs found; {len(cache)} already transcribed; {len(todo)} to do")
    if not todo:
        return 0

    done = 0
    started = time.time()

    def work(path: pathlib.Path):
        pages = ingest.transcribe_pdf(path.read_bytes(), workers=args.page_workers)
        return path, pages

    with ThreadPoolExecutor(max_workers=args.file_workers) as pool:
        futures = {pool.submit(work, p): p for p in todo}
        for future in as_completed(futures):
            path = futures[future]
            try:
                _, pages = future.result()
            except Exception as e:
                print(f"  FAILED {path.name}: {e}", flush=True)
                continue
            with _lock:
                cache[path.name] = {
                    "file": path.name,
                    "title": gemini.extract_title(path.name, pages[0]["text"] if pages else ""),
                    "pages": pages,
                }
                done += 1
                if done % 5 == 0:
                    save_cache(cache)
                chars = sum(len(p["text"]) for p in pages)
                rate = done / max(time.time() - started, 1) * 60
                print(f"  [{done}/{len(todo)}] {len(pages):2d}p {chars:6d} chars "
                      f"({rate:.1f}/min)  {path.name[:58]}", flush=True)

    save_cache(cache)
    print(f"\nCache written to {CACHE.relative_to(ROOT)} ({len(cache)} documents)")
    return 0


def cmd_push(args) -> int:
    from audit import chunking, ingest, store

    cache = load_cache()
    if not cache:
        print("No transcription cache. Run `transcribe` first.", file=sys.stderr)
        return 1

    blob_urls = {}
    if args.blob_urls and pathlib.Path(args.blob_urls).exists():
        blob_urls = json.loads(pathlib.Path(args.blob_urls).read_text())

    import datetime as dt
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")

    total_chunks = 0
    for i, (filename, entry) in enumerate(sorted(cache.items()), 1):
        pages = [p for p in entry["pages"] if p["text"].strip()]
        if not pages:
            print(f"  [{i}/{len(cache)}] SKIP (no text) {filename[:55]}")
            continue
        doc = {
            "doc_id": ingest.make_doc_id(filename),
            "file": filename,
            "title": entry["title"],
            "pages": pages,
            "blob_url": blob_urls.get(filename),
            "indexed_at": now,
        }
        chunks = chunking.document_to_chunks(doc)
        store.index().delete(prefix=f"{doc['doc_id']}:")
        store.upsert_chunks(chunks)
        store.save_document(doc)
        total_chunks += len(chunks)
        print(f"  [{i}/{len(cache)}] {len(pages):2d}p {len(chunks):3d} chunks  "
              f"{filename[:55]}", flush=True)

    print(f"\nPushed {len(cache)} documents, {total_chunks} chunks.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    t = sub.add_parser("transcribe", help="Gemini-vision every PDF into data/transcriptions.json")
    t.add_argument("--file-workers", type=int, default=3, help="PDFs in flight at once")
    t.add_argument("--page-workers", type=int, default=4, help="pages in flight per PDF")
    t.add_argument("--force", action="store_true", help="re-transcribe cached files")
    t.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("push", help="upload the cache to Upstash Vector + Redis")
    p.add_argument("--blob-urls", help="optional JSON map {filename: blob url}")
    p.set_defaults(func=cmd_push)

    a = sub.add_parser("all", help="transcribe then push")
    a.add_argument("--file-workers", type=int, default=3)
    a.add_argument("--page-workers", type=int, default=4)
    a.add_argument("--force", action="store_true")
    a.add_argument("--blob-urls")
    a.set_defaults(func=lambda args: cmd_transcribe(args) or cmd_push(args))

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
