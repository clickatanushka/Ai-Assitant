#!/usr/bin/env python3
"""Upload the source PDFs to Vercel Blob so the app can show the real document.

    python scripts/upload_pdfs.py            # upload, then patch the manifests
    python scripts/upload_pdfs.py --dry-run  # list what would be uploaded

Until this runs, "View source" falls back to the transcribed text: correct, but
not the original page an auditor wants to point at.

Prerequisites — a Blob store must exist and be linked to the project:

    Vercel dashboard -> Storage -> Create Database -> Blob
    npx vercel env pull .env.vercel        # brings down BLOB_READ_WRITE_TOKEN

Uploads go through the Vercel CLI (`vercel blob put`), because Vercel Blob has no
documented REST API for other languages. Already-uploaded files are skipped, so
re-running after an interruption is cheap.
"""

import argparse
import json
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bulk_ingest import find_pdfs, load_env  # noqa: E402

load_env()

URL_MAP = ROOT / "data" / "blob_urls.json"
URL_RE = re.compile(r"https://[^\s\"']+\.public\.blob\.vercel-storage\.com/\S+")


def load_map() -> dict:
    if URL_MAP.exists():
        return json.loads(URL_MAP.read_text(encoding="utf-8"))
    return {}


def save_map(m: dict) -> None:
    URL_MAP.parent.mkdir(parents=True, exist_ok=True)
    URL_MAP.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


def upload(path: pathlib.Path) -> str | None:
    """`vercel blob put` prints the blob URL; pull it back out of the output."""
    proc = subprocess.run(
        ["npx", "vercel", "blob", "put", str(path), "--pathname", f"pdfs/{path.name}",
         "--add-random-suffix=false", "--force"],
        capture_output=True, text=True, cwd=ROOT,
    )
    combined = proc.stdout + proc.stderr
    match = URL_RE.search(combined)
    if match:
        return match.group(0)
    print(f"    no URL in CLI output: {combined.strip()[:200]}")
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-push", action="store_true",
                        help="upload only; do not patch the Upstash manifests")
    args = parser.parse_args()

    pdfs = find_pdfs()
    urls = load_map()
    todo = [p for p in pdfs if p.name not in urls]
    total_mb = sum(p.stat().st_size for p in todo) / 1e6
    print(f"{len(pdfs)} PDFs, {len(urls)} already uploaded, "
          f"{len(todo)} to go ({total_mb:.0f} MB)")
    if args.dry_run:
        for p in todo[:10]:
            print(f"    {p.stat().st_size/1e6:5.1f} MB  {p.name[:64]}")
        return 0
    if not todo:
        print("Nothing to upload.")

    for i, path in enumerate(todo, 1):
        url = upload(path)
        if url:
            urls[path.name] = url
            if i % 5 == 0:
                save_map(urls)
            print(f"  [{i}/{len(todo)}] {path.stat().st_size/1e6:5.1f} MB  {path.name[:56]}",
                  flush=True)
        else:
            print(f"  [{i}/{len(todo)}] FAILED {path.name[:56]}", flush=True)
    save_map(urls)
    print(f"\n{len(urls)} URLs in {URL_MAP.relative_to(ROOT)}")

    if args.skip_push:
        return 0

    # Re-push the manifests so each document carries its blob_url. This only
    # rewrites metadata — the vectors and transcriptions are untouched.
    print("\nPatching document manifests with their blob URLs...")
    from audit import store
    patched = 0
    for doc in store.list_documents():
        url = urls.get(doc["file"])
        if not url or doc.get("blob_url") == url:
            continue
        full = store.get_document(doc["doc_id"])
        if not full:
            continue
        full["blob_url"] = url
        store.save_document(full)
        patched += 1
    print(f"  {patched} manifests updated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
