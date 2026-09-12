"""Environment configuration and tuning constants.

Every secret comes from an environment variable — nothing is written to disk at
runtime, because Vercel's filesystem is read-only outside /tmp.
"""

import os

# ── Secrets (set these in Vercel → Project → Settings → Environment Variables) ─
GEMINI_API_KEY      = os.environ.get("GEMINI_API_KEY", "")
UPSTASH_VECTOR_URL  = os.environ.get("UPSTASH_VECTOR_REST_URL", "")
UPSTASH_VECTOR_TOKEN= os.environ.get("UPSTASH_VECTOR_REST_TOKEN", "")
UPSTASH_REDIS_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
UPSTASH_REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
BLOB_TOKEN          = os.environ.get("BLOB_READ_WRITE_TOKEN", "")
ADMIN_PASSWORD      = os.environ.get("ADMIN_PASSWORD", "")

# ── Gemini ────────────────────────────────────────────────────────────────────
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_BASE  = "https://generativelanguage.googleapis.com/v1beta/models"

# Thinking is disabled on every call: left on, the model spends the whole output
# budget reasoning and returns an empty transcription. The parameter changed shape
# between model generations — 2.5 took {"thinkingBudget": 0}, 3.x takes
# {"thinkingLevel": "minimal"} and rejects the old form outright — so it lives here
# next to the model name rather than being repeated at each call site.
THINKING_CONFIG = {"thinkingLevel": "minimal"}

# Client-side rate limit. The free tier rejects bursts with a 429 and Google no
# longer publishes the numbers, so this is enforced here rather than discovered
# halfway through a bulk ingest. Raise it if the key is on a paid tier.
REQUESTS_PER_MINUTE = int(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "8"))

# ── Chunking ──────────────────────────────────────────────────────────────────
# The median document here is 2 pages, so a page usually fits in one chunk and
# the embedding sees the whole thought rather than a 100-word slice of it.
# Longer pages fall back to overlapping windows.
CHUNK_MAX_WORDS   = 400   # a page shorter than this becomes a single chunk
CHUNK_WINDOW      = 350   # window size when a page must be split
CHUNK_OVERLAP     = 80

# ── Retrieval ─────────────────────────────────────────────────────────────────
SEARCH_TOP_K      = 20    # chunks fetched per query variant (English + German)
RRF_K             = 60    # reciprocal-rank-fusion constant, the standard default
RERANK_CANDIDATES = 6     # documents sent to the reranker
RERANK_MIN_SCORE  = 6     # 0-10; below this a document is not used at all
MAX_ANSWER_DOCS   = 3     # documents whose full text goes into the answer prompt

# ── Grounding ─────────────────────────────────────────────────────────────────
# A quote that is not found verbatim is retried as a fuzzy match at this ratio
# before being discarded. High enough that only OCR-level noise slips through.
QUOTE_FUZZY_RATIO = 0.85
