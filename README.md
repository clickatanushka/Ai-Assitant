# Audit Assistant

Ask a question in English, get the answer in German **and** English, with the exact
document, page, and a verbatim German quote — searched across a corpus of German
ISO 9001 quality-management documents.

Built to be deployed on Vercel, with new PDFs uploadable through the browser.

---

## ⚠️ Read this first: the API key must belong to the *billed project*

Gemini keys are scoped to a Google Cloud **project**, and billing is attached to a
project too. A key from an unbilled project stays on the free tier no matter how
much credit sits on a different one — it fails like this:

```
GenerateRequestsPerMinutePerProjectPerModel-FreeTier
GenerateRequestsPerDayPerProjectPerModel-FreeTier   limit: 20
```

That is not enough to run this system:

| Job | Requests needed |
|---|---|
| Indexing the existing 82 documents | **82** (one per document) |
| Each question asked | **3** (query prep, rerank, answer) |

So a free-tier key indexes about a quarter of one day's documents, or answers about
six questions per day.

**Fix:** in [Google AI Studio → API Keys](https://aistudio.google.com/apikey), check
the *project* column and use a key belonging to the project that has billing enabled,
creating one there if needed. Cost is small — indexing all 289 pages is roughly 95k
input tokens plus transcription output, about $0.50 one-off, and a question costs
around half a cent.

**On key format:** current keys start with `AQ.` (Auth keys). Google migrated away
from the older `AIza` Standard keys during 2026 and now rejects them, so `AQ.` is
correct — do not go looking for an `AIza` key. One consequence: `AQ.` keys are
rejected by the OpenAI-compatible endpoint (`/v1beta/openai/chat/completions`), which
the old `server.py` used. This version calls the native `generateContent` endpoint
instead, which accepts them.

To confirm a key is actually on the paid tier, send ~10 quick requests: a free-tier
key returns `...PerMinutePerProjectPerModel-FreeTier` after a handful.

You can raise throughput once on a paid tier:

```bash
GEMINI_REQUESTS_PER_MINUTE=60    # default is 8, sized for the free tier
```

---

## What changed, and why

The previous version extracted text with `pdfplumber` and fell back to Tesseract
OCR. Measuring the corpus showed why answers were unreliable:

- 289 pages total; **182 of them (63%) have no extractable text** — they are scans
- **57 of the 82 files are entirely scanned images**
- so **62% of the old search index came from Tesseract**

And Tesseract was corrupting the German. Measured across the **full 82-document
corpus**, old index vs. new transcription:

| | Tesseract (old) | Gemini vision (new) |
|---|---|---|
| Pages readable | 107 of 289 | **274** |
| Mangled characters | **2,137** (24.4 per 1k words) | **8** (0.1 per 1k) |
| — `é` written for `ä`/`ö` | 682 | **0** |
| — `B` written for `ß` | 294 | 7 |
| — `ii` written for `ü` | 1,161 | 1 |
| Gibberish runs | 31 | **0** |
| Correct umlauts | 69.2 per 1k words | **96.9 per 1k words** |

That last row is the one that matters for search: Tesseract was silently losing
about **2,100 umlauts**, roughly a quarter of every one in the corpus.

Tesseract was destroying roughly half of all umlauts. `Schwall-Löten` was indexed
as `Schwall-Léten`, `Maßnahmen` as `MaBnahmen`, `Qualität` as `Qualitét` — none of
which match the correct German under either semantic or keyword search.

Reproduce this yourself with `python scripts/compare_ocr.py --legacy eval/legacy_index.json`.

**Ingestion now goes through Gemini 3.6 Flash vision**, which reads scanned German
correctly, renders tables as Markdown, and transcribes the labels inside the
process-flow diagrams that make up most of the `PB` documents — content Tesseract
could not see at all.

## How a question is answered

> **Model note:** `gemini-2.5-flash` is no longer available to newly created Google
> Cloud projects — it returns a 404 pointing at `gemini-3.6-flash`. The 3.x line also
> replaced `thinkingConfig.thinkingBudget: 0` with `thinkingConfig.thinkingLevel:
> "minimal"`, and rejects the old form. Both live in `audit/config.py`; override the
> model with the `GEMINI_MODEL` env var.

1. **Query preparation** — one Gemini call expands acronyms (`MSL` → *Moisture
   Sensitivity Level, dry storage…*) and translates to German. The German variant
   matters: the BM25 half of the index cannot match German text from English tokens.
2. **Hybrid search** — each variant is embedded with `gemini-embedding-2`
   (`RETRIEVAL_QUERY` task) and sent to Upstash alongside its raw text, so the dense
   and BM25 halves both run and Upstash fuses them with RRF. The two result lists
   are then fused again.
3. **Rerank** — Gemini scores the top 6 documents 0–10. Anything below 6 is
   discarded, so **"not in these documents" is a real answer** rather than the
   closest unrelated match.
4. **Answer from full documents** — the complete text of the winning documents goes
   into the prompt. The old version truncated every excerpt to 350 characters.
5. **Verify every quote** — each German quote is checked against the stored page
   text. Exact matches are marked ✓ verbatim; near matches are replaced with the
   passage as the document actually words it and marked ≈ matched; anything that
   cannot be traced is **dropped** and the answer is flagged.

## Setup

### 1. Services

| Service | What for | Free tier |
|---|---|---|
| [Gemini](https://aistudio.google.com/apikey) | transcription, rerank, answers | see the warning above |
| [Upstash Vector](https://console.upstash.com/vector) | hybrid search index | ~195k vectors, 10k ops/day |
| [Upstash Redis](https://console.upstash.com/redis) | document text store | 10k commands/day |
| [Vercel Blob](https://vercel.com/docs/vercel-blob) | original PDFs (optional) | included with Vercel |

Create the Upstash Vector index as a **hybrid** index:

| Field | Value |
|---|---|
| Type | **Hybrid** |
| Dense Embedding Model | **Custom** |
| Dimensions | **1536** |
| Metric | **COSINE** |
| Sparse Embedding Model | **BM25** |

Upstash no longer offers `BAAI/bge-m3` as a hosted model (the console lists only
*Custom* and `openai/text-embedding-3-small`, and the latter is being deprecated in
October 2026). So the dense half is computed here with **`gemini-embedding-2`** and
upserted directly, while the raw text still goes to Upstash for BM25 to index.

Dimensions must be **1536** — it is Upstash's free-tier ceiling, the model truncates
to it natively, and a mismatch is rejected at upsert. Measured 5/5 on
English-question → German-document retrieval over this corpus.

### 2. Environment

```bash
cp .env.example .env      # then fill it in
```

### 3. Index the existing corpus

Transcription is the slow, paid step, so it is cached separately from the upload —
a wrong Upstash credential should never mean paying to transcribe twice.

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt

.venv/bin/python scripts/bulk_ingest.py transcribe   # Gemini → data/transcriptions.json
.venv/bin/python scripts/bulk_ingest.py push         # cache  → Upstash
```

`transcribe` is resumable: it skips documents already in the cache, so if it stops
on a quota limit, just run it again.

### 4. Deploy

```bash
npm install          # only @vercel/blob, for the large-file upload route
vercel deploy
```

Set the same environment variables in the Vercel project settings.

## Adding documents later

Open the **Documents** tab, drop PDFs in, enter the admin password. Files under
4.5 MB post directly to the function; larger ones go to Vercel Blob first (Vercel
caps function request bodies at 4.5 MB). Delete removes a document's transcription
and all of its search entries.

## Checking it works

```bash
.venv/bin/python tests/test_pipeline.py            # full pipeline, no network needed
.venv/bin/python scripts/compare_ocr.py            # transcription quality vs. the old OCR
.venv/bin/python scripts/eval.py                   # retrieval accuracy
.venv/bin/python scripts/eval.py --scanned         # only the previously-unsearchable documents
```

`eval/questions.json` holds English questions paired with the German document that
should answer them, deliberately weighted toward the scanned files. It also holds
`negative_questions` that have no answer in the corpus — a system that answers
those anyway is failing in a way accuracy alone would not show.

## Layout

```
app.py                  FastAPI entrypoint — every route
audit/config.py         env vars and tuning constants
audit/gemini.py         transcription, query prep, rerank, answering
audit/store.py          Upstash Vector + Redis + blob reads
audit/chunking.py       page → chunks
audit/ingest.py         PDF → transcription → chunks → index
audit/retrieve.py       the question → answer pipeline
audit/verify.py         quote grounding checks
api/blob.ts             the only JS: mints Vercel Blob upload tokens
public/index.html       Q&A, Documents and Dashboard tabs
scripts/bulk_ingest.py  one-time corpus load
scripts/eval.py         retrieval evaluation
scripts/compare_ocr.py  old-vs-new transcription quality
tests/test_pipeline.py  end-to-end test with Gemini and Upstash stubbed
eval/legacy_index.json  the old Tesseract index, kept as the baseline
```

## Notes

- **Vercel's Hobby plan is licensed for non-commercial use.** Using this in a real
  audit needs a Pro plan.
- `pdfs/` is no longer tracked in git — the PDFs live in Vercel Blob. The 167 MB
  still in the repo's history can be purged separately with `git filter-repo`.
- The Gemini key is read from `GEMINI_API_KEY` only. The old Settings modal wrote
  it to `api_key.txt`, which cannot work on Vercel's read-only filesystem.
