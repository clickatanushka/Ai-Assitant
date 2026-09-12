#!/usr/bin/env python3
"""Measure retrieval accuracy against eval/questions.json.

    python scripts/eval.py              # full run
    python scripts/eval.py --scanned    # only the previously-unsearchable documents
    python scripts/eval.py --no-negative

Reports top-1 and top-3 document accuracy, split by whether the answer lives on a
scanned page. The scanned split is the one that matters: those documents used to
reach the index only through Tesseract, with broken umlauts.

The negative questions have no answer in the corpus. A correct system declines
them; one that always returns its closest guess fails here, and that failure is
invisible in accuracy numbers alone.
"""

import argparse
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bulk_ingest import load_env  # noqa: E402  (shares the .env reader)

load_env()

QUESTIONS = ROOT / "eval" / "questions.json"


def matches(filename: str, expected: list[str]) -> bool:
    lowered = filename.lower()
    return any(fragment.lower() in lowered for fragment in expected)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scanned", action="store_true",
                        help="only questions whose source is a scanned document")
    parser.add_argument("--no-negative", action="store_true",
                        help="skip the should-not-answer questions")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    from audit import retrieve

    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.scanned:
        questions = [q for q in questions if q.get("scanned")]

    stats = {"scanned": [0, 0, 0], "native": [0, 0, 0]}   # [n, top1, top3]
    failures = []

    for i, item in enumerate(questions, 1):
        try:
            result = retrieve.retrieve(item["question"])
        except Exception as e:
            print(f"  [{i}/{len(questions)}] ERROR {e}")
            continue

        ranked = [c["file"] for c in result["candidates"]]
        top1 = bool(ranked) and matches(ranked[0], item["expect"])
        top3 = any(matches(f, item["expect"]) for f in ranked[:3])

        bucket = stats["scanned" if item.get("scanned") else "native"]
        bucket[0] += 1
        bucket[1] += top1
        bucket[2] += top3

        mark = "✓" if top1 else ("~" if top3 else "✗")
        print(f"  [{i}/{len(questions)}] {mark} {item['question'][:58]}")
        if args.verbose or not top3:
            for c in result["candidates"][:3]:
                print(f"        rel={c['relevance']:2d}  {c['file'][:62]}")
        if not top3:
            failures.append((item["question"], item["expect"], ranked[:3]))

    print("\n" + "=" * 72)
    total = [0, 0, 0]
    for name in ("scanned", "native"):
        n, t1, t3 = stats[name]
        if not n:
            continue
        total = [a + b for a, b in zip(total, stats[name])]
        print(f"{name:8s}  n={n:2d}   top-1 {t1}/{n} ({t1/n:5.1%})   top-3 {t3}/{n} ({t3/n:5.1%})")
    if total[0]:
        n, t1, t3 = total
        print(f"{'OVERALL':8s}  n={n:2d}   top-1 {t1}/{n} ({t1/n:5.1%})   top-3 {t3}/{n} ({t3/n:5.1%})")

    if not args.no_negative:
        print("\nQuestions that should NOT be answered:")
        declined = 0
        for question in spec.get("negative_questions", []):
            result = retrieve.retrieve(question)
            ok = not result["selected"]
            declined += ok
            best = result["candidates"][0] if result["candidates"] else None
            detail = "declined" if ok else f"ANSWERED from {best['file'][:44]} (rel={best['relevance']})"
            print(f"  {'✓' if ok else '✗'} {question[:50]:52s} {detail}")
        n = len(spec.get("negative_questions", []))
        if n:
            print(f"  declined {declined}/{n}")

    if failures:
        print("\nMissed entirely:")
        for question, expect, got in failures:
            print(f"  {question[:52]}\n    expected {expect}\n    got      {[g[:40] for g in got]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
