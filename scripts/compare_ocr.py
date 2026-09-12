#!/usr/bin/env python3
"""Compare the Gemini transcription against the old Tesseract index.

    python scripts/compare_ocr.py [--legacy index.json] [--sample 3]

The old pipeline ran Tesseract over the 63% of pages that carry no text layer,
which broke German umlauts (ä rendered as é, ö as 6, and so on). This quantifies
that damage and shows what the new transcription produced for the same documents,
so the improvement is a measured number rather than an impression.
"""

import argparse
import json
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

GERMAN_UMLAUTS = re.compile(r"[äöüÄÖÜß]")

# How Tesseract actually mangles these scans, taken from the current index:
#   Schwall-Löten  -> Schwall-Léten     ö becomes é
#   Qualität       -> Qualitét          ä becomes é
#   Maßnahmen      -> MaBnahmen         ß becomes a capital B mid-word
#   für / bestückt -> fiir / bestiickte ü becomes ii
#   überprüfen     -> Uberprifen        umlaut simply dropped
# Each pattern is essentially absent from correct German, so counting them
# measures OCR damage rather than style.
DAMAGE_PATTERNS = {
    "é for ä/ö": re.compile(r"[a-zäöüß]é[a-zäöüß]|é[a-z]{2,}"),
    "B for ß":   re.compile(r"[a-zäöü]B[a-zäöü]"),
    "ii for ü":  re.compile(r"[a-z]ii[a-z]"),
    "other accents": re.compile(r"[àâãèêëìîïòóôõùúûý]"),
}
# A run of isolated single letters, e.g. "s e o e o OE HOE RE" — Tesseract
# reading the dot leaders in a table of contents.
GIBBERISH_RUN = re.compile(r"(?:\b[a-zA-Z]\b[ ]+){4,}")


def score(text: str) -> dict:
    result = {
        "chars": len(text),
        "words": len(text.split()),
        "umlauts": len(GERMAN_UMLAUTS.findall(text)),
        "gibberish_runs": len(GIBBERISH_RUN.findall(text)),
    }
    damage = 0
    for name, pattern in DAMAGE_PATTERNS.items():
        hits = len(pattern.findall(text))
        result[f"dmg:{name}"] = hits
        damage += hits
    result["damage"] = damage
    return result


def report(name: str, texts: dict[str, str]) -> dict:
    totals: dict[str, int] = {}
    for text in texts.values():
        for key, value in score(text).items():
            totals[key] = totals.get(key, 0) + value
    words = max(totals.get("words", 0), 1)
    per_1k = lambda v: v / words * 1000                                # noqa: E731
    print(f"\n{name}")
    print(f"  documents          {len(texts)}")
    print(f"  words              {totals.get('words', 0):,}")
    print(f"  correct umlauts    {totals.get('umlauts', 0):,}"
          f"   ({per_1k(totals.get('umlauts', 0)):.1f} per 1k words)")
    print(f"  MANGLED characters {totals.get('damage', 0):,}"
          f"   ({per_1k(totals.get('damage', 0)):.1f} per 1k words)")
    for key in DAMAGE_PATTERNS:
        print(f"      {key:16s} {totals.get(f'dmg:{key}', 0):,}")
    print(f"  gibberish runs     {totals.get('gibberish_runs', 0):,}")
    return totals


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy", default="index.json",
                        help="the old Tesseract index (default: index.json)")
    parser.add_argument("--cache", default="data/transcriptions.json")
    parser.add_argument("--sample", type=int, default=2,
                        help="documents to print a side-by-side excerpt for")
    args = parser.parse_args()

    cache_path = ROOT / args.cache
    if not cache_path.exists():
        print(f"No transcription cache at {args.cache}. Run scripts/bulk_ingest.py transcribe.",
              file=sys.stderr)
        return 1
    cache = json.loads(cache_path.read_text(encoding="utf-8"))
    new_texts = {
        name: "\n".join(p["text"] for p in entry["pages"])
        for name, entry in cache.items() if entry["pages"]
    }

    legacy_path = ROOT / args.legacy
    old_texts: dict[str, str] = {}
    if legacy_path.exists():
        chunks = json.loads(legacy_path.read_text(encoding="utf-8"))
        for chunk in chunks:
            if chunk["file"] in new_texts:
                old_texts.setdefault(chunk["file"], "")
                old_texts[chunk["file"]] += " " + chunk["text"]
    else:
        print(f"(no legacy index at {args.legacy} — showing new figures only)")

    shared = sorted(set(new_texts) & set(old_texts)) if old_texts else sorted(new_texts)
    print(f"Comparing {len(shared)} document(s) present in both.")

    if old_texts:
        report("OLD — Tesseract OCR", {k: old_texts[k] for k in shared})
    report("NEW — Gemini vision", {k: new_texts[k] for k in shared})

    for name in shared[:args.sample]:
        print("\n" + "─" * 74)
        print(name[:74])
        if old_texts:
            print("\n  OLD (Tesseract):")
            print("   ", old_texts[name][:300].strip().replace("\n", " "))
        print("\n  NEW (Gemini):")
        print("   ", new_texts[name][:300].strip().replace("\n", " "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
