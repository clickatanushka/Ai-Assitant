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
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bulk_ingest import find_pdfs, load_env  # noqa: E402

load_env()

URL_MAP = ROOT / "data" / "blob_urls.json"
URL_RE = re.compile(r"https://[^\s\"']+\.(?:public|private)\.blob\.vercel-storage\.com/\S+")


def load_map() -> dict:
    if URL_MAP.exists():
        return json.loads(URL_MAP.read_text(encoding="utf-8"))
    return {}


def save_map(m: dict) -> None:
    URL_MAP.parent.mkdir(parents=True, exist_ok=True)
    URL_MAP.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")


NOISE = re.compile(r"^(npm notice|<claude-code-hint|Vercel CLI |\s*$)")


def clean(output: str) -> str:
    """Strip npm/CLI chatter so a real error is actually visible. The first
    version of this buried the error behind notice lines and reported
    'no URL in CLI output' 31 times in a row."""
    return "\n".join(l for l in output.splitlines() if not NOISE.match(l)).strip()


def auth_flags() -> list[str]:
    """Vercel Blob accepts either a read-write token or OIDC + store id.

    OIDC is what a linked project gets by default, but it is enabled per
    environment and is typically off for "development" — so running this from a
    laptop fails with "OIDC is enabled for this project, but not for the
    development environment". The read-write token works everywhere, so it wins
    when present.
    """
    token = os.environ.get("BLOB_READ_WRITE_TOKEN", "").strip()
    if token:
        return ["--rw-token", token]

    oidc = os.environ.get("VERCEL_OIDC_TOKEN", "").strip()
    store = os.environ.get("BLOB_STORE_ID", "").strip()
    if oidc and store:
        return ["--oidc-token", oidc, "--store-id", store]

    sys.exit(
        "No Blob credentials.\n\n"
        "Get the read-write token — it is the one that works from a laptop:\n"
        "  Vercel dashboard -> Storage -> your Blob store -> the '.env.local' tab\n"
        "  in the Quickstart panel. Copy the BLOB_READ_WRITE_TOKEN=... line.\n\n"
        "  echo 'BLOB_READ_WRITE_TOKEN=vercel_blob_rw_...' >> .env\n"
        "  .venv/bin/python scripts/upload_pdfs.py\n\n"
        "(OIDC also works, but only in environments where it is enabled — usually\n"
        " Production and Preview, not development.)"
    )


def upload(path: pathlib.Path, auth: list[str]) -> tuple[str | None, str]:
    """`vercel blob put` prints the blob URL; pull it back out of the output.

    Credentials are passed explicitly. Left to itself the CLI picks up a stale
    VERCEL_OIDC_TOKEN from .env.local (written by `vercel link`, and short-lived)
    and fails with an access error that looks nothing like an expiry.
    """
    proc = subprocess.run(
        ["npx", "vercel", "blob", "put", str(path),
         "--pathname", f"pdfs/{path.name}",
         # Private: these are confidential audit documents, and a public blob URL
         # is anonymously readable by anyone who ever sees it. The app serves them
         # back through /api/pdf/<doc_id>, which attaches the token server-side.
         "--access", "private",
         "--allow-overwrite", "true",
         *auth],
        capture_output=True, text=True, cwd=ROOT,
    )
    combined = clean(proc.stdout + proc.stderr)
    match = URL_RE.search(combined)
    return (match.group(0) if match else None), combined


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
        return 0

    auth = auth_flags()
    failures = 0
    for i, path in enumerate(todo, 1):
        url, output = upload(path, auth)
        if url:
            urls[path.name] = url
            if i % 5 == 0:
                save_map(urls)
            print(f"  [{i}/{len(todo)}] {path.stat().st_size/1e6:5.1f} MB  {path.name[:56]}",
                  flush=True)
        else:
            failures += 1
            print(f"  [{i}/{len(todo)}] FAILED {path.name[:56]}\n    {output[:400]}",
                  flush=True)
            # Stop rather than repeat the same failure 80 times: if the first two
            # fail the cause is configuration, not this particular file.
            if failures >= 2 and len(urls) == 0:
                save_map(urls)
                sys.exit("\nStopping — the first uploads all failed, so this is a "
                         "setup problem rather than a bad file. Fix the error above "
                         "and re-run; finished uploads are skipped.")
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
