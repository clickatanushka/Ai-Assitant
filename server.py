#!/usr/bin/env python3

"""
Audit Assistant Backend — Gemini Flash Edition
Run: python server.py
"""

import os, json, re, math, urllib.request, urllib.error, urllib.parse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PDF_DIR       = "./pdfs"
INDEX_FILE    = "./index.json"
EMB_FILE      = "./embeddings.npy"
API_KEY_FILE  = "./api_key.txt"
PORT          = 8000
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20
TOP_K         = 15
BOOST_CAP     = 0.12   # max lexical filename boost per document (prevents synonym flooding)
EMB_MODEL_NAME = "BAAI/bge-m3"
GEMINI_MODEL  = "gemini-2.5-flash"
GEMINI_URL    = "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"

# Lazy-loaded embedding model (loaded on first use to keep startup fast)
_EMB_MODEL = None
def get_embedding_model():
    global _EMB_MODEL
    if _EMB_MODEL is None:
        from sentence_transformers import SentenceTransformer
        print(f"Loading embedding model '{EMB_MODEL_NAME}' (first time may download ~470MB)...")
        _EMB_MODEL = SentenceTransformer(EMB_MODEL_NAME)
        print("Embedding model ready.")
    return _EMB_MODEL

# ── PDF Extraction ────────────────────────────────────────────────────────────
def extract_text_from_pdf(path):
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})
        if not pages:
            print(f"    → OCR: {os.path.basename(path)}")
            try:
                from pdf2image import convert_from_path
                import pytesseract
                images = convert_from_path(path, dpi=200)
                for i, img in enumerate(images):
                    text = pytesseract.image_to_string(img, lang="deu+eng")
                    if text.strip():
                        pages.append({"page": i + 1, "text": text.strip()})
            except Exception as ocr_err:
                print(f"    OCR failed: {ocr_err}")
        return pages
    except Exception as e:
        print(f"  Error reading {path}: {e}")
        return []

def chunk_pages(pages, filename):
    chunks = []
    for p in pages:
        words = p["text"].split()
        for start in range(0, max(1, len(words) - 20), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_words = words[start:start + CHUNK_SIZE]
            if len(chunk_words) < 15:
                continue
            chunks.append({
                "file": filename,
                "page": p["page"],
                "text": " ".join(chunk_words),
            })
    return chunks

def _call_gemini(payload_bytes, api_key, timeout=45):
    """Single Gemini API call; raises urllib.error.HTTPError on failure."""
    req = urllib.request.Request(
        GEMINI_URL,
        data=payload_bytes,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}",
                 "User-Agent": "Mozilla/5.0"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))

def get_all_pdfs():
    """Walk PDF_DIR recursively; return sorted list of (filename, full_path)."""
    result = []
    for root, _dirs, files in os.walk(PDF_DIR):
        for f in files:
            if f.lower().endswith(".pdf"):
                result.append((f, os.path.join(root, f)))
    result.sort(key=lambda x: x[0].lower())
    return result

def build_index():
    print("Building index from PDFs...")
    all_chunks = []
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_files = get_all_pdfs()
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}/ (including subdirectories)")
        return []
    for i, (filename, path) in enumerate(pdf_files):
        print(f"  [{i+1}/{len(pdf_files)}] {filename}")
        pages = extract_text_from_pdf(path)
        chunks = chunk_pages(pages, filename)
        all_chunks.extend(chunks)
        print(f"    → {len(pages)} pages, {len(chunks)} chunks")
    with open(INDEX_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"\nIndex built: {len(all_chunks)} total chunks from {len(pdf_files)} PDFs")
    return all_chunks

def load_index():
    if os.path.exists(INDEX_FILE):
        with open(INDEX_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    return build_index()

# ── Semantic Search (multilingual embeddings) ─────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def stem(w):
    """Crude German stemmer for lexical matching: strip the participle 'ge-' prefix
    ('gewaschen' -> 'waschen') and reduce to a 6-char root so inflected forms match
    ('kalibriert'/'Kalibrierung' -> 'kalibr', 'gewaschen'/'Waschen' -> 'wasche')."""
    w = w.lower()
    if len(w) > 7 and w.startswith("ge"):
        w = w[2:]
    return w[:6]

def build_idf(chunks):
    """Inverse-document-frequency of every word-stem across the corpus.
    Tells us how *distinctive* a term is: 'dokume' (in most docs) → low weight,
    'stichp' (in one doc) → high weight. Derived purely from the documents — no
    hardcoded keywords. Used to weight the lexical filename boost in search()."""
    doc_stems = {}                       # filename -> set of stems it contains
    for c in chunks:
        s = doc_stems.setdefault(c["file"], set())
        for w in tokenize(c["text"]):
            if len(w) >= 6:
                s.add(stem(w))
    N = max(len(doc_stems), 1)
    df = {}
    for stems in doc_stems.values():
        for st in stems:
            df[st] = df.get(st, 0) + 1
    return {st: math.log(N / cnt) for st, cnt in df.items()}, math.log(N)

def build_embeddings(chunks):
    """Embed every chunk (filename + text) into a normalized vector matrix and cache to disk."""
    import numpy as np
    model = get_embedding_model()
    print(f"Embedding {len(chunks)} chunks...")
    # Prepend the filename so document titles (e.g. 'Stichprobenprüfungen') contribute to the vector
    texts = [f"{c['file']}\n{c['text']}" for c in chunks]
    embs = model.encode(
        texts, batch_size=32, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype(np.float32)
    np.save(EMB_FILE, embs)
    print(f"Search index ready ({embs.shape[0]} vectors, dim {embs.shape[1]})")
    return embs

def load_embeddings(chunks):
    """Load cached embeddings if they match the current index, else rebuild."""
    import numpy as np
    if os.path.exists(EMB_FILE):
        try:
            embs = np.load(EMB_FILE)
            if embs.shape[0] == len(chunks):
                print(f"Loaded cached embeddings ({embs.shape[0]} vectors)")
                return embs
            print("Embedding cache stale (chunk count changed) — rebuilding.")
        except Exception as e:
            print(f"Embedding cache unreadable ({e}) — rebuilding.")
    return build_embeddings(chunks)

def translate_to_german(text):
    """Translate the English query to German (free Google Translate) for the lexical boost."""
    try:
        from deep_translator import GoogleTranslator
        return GoogleTranslator(source='en', target='de').translate(text)
    except Exception as e:
        print(f"[translate] {e}")
        return text

_EXPANSION_CACHE = {}   # query -> expanded query, so the same question is always identical

def expand_query(question, api_key):
    """Expand abbreviations/terse queries into a fuller search query using the LLM's
    general domain knowledge (e.g. 'MSL' -> 'Moisture Sensitivity Level, dry storage of
    moisture-sensitive ICs'). This is world knowledge, NOT memorized document mappings —
    it just turns a terse query into words the semantic search can match. The result is
    cached per query so the SAME question always yields the SAME search (deterministic).
    Falls back to the original question on any error."""
    if not api_key:
        return question
    key = question.strip().lower()
    if key in _EXPANSION_CACHE:
        return _EXPANSION_CACHE[key]
    payload = json.dumps({
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You expand short audit/manufacturing search queries for a document search engine. "
                "Given a query (possibly a bare acronym or a few words), rewrite it as ONE richer "
                "search phrase: keep the original words, expand any acronyms using general "
                "ISO 9001 / electronics-manufacturing knowledge, and add a few closely-related "
                "synonyms. Do NOT invent document names. Output ONLY the expanded phrase, max 30 words."
            )},
            {"role": "user", "content": question}
        ],
        "temperature": 0,
        "max_tokens": 200,
        "reasoning_effort": "none",   # gemini-2.5-flash 'thinking' would eat the token budget
    }).encode("utf-8")
    try:
        data = _call_gemini(payload, api_key, timeout=20)
        expanded = data["choices"][0]["message"]["content"].strip()
        result = f"{question}. {expanded}" if expanded else question
        _EXPANSION_CACHE[key] = result
        return result
    except Exception as e:
        print(f"[expand_query] {e}")
        return question

def search(query, chunks, embeddings, top_k=TOP_K):
    """Hybrid search = semantic similarity + an IDF-weighted lexical boost when a
    German query keyword appears in a document's filename. The multilingual model
    matches an English query directly against German text; the lexical boost rescues
    documents whose distinctive topic word is in the title. Each matched word is
    weighted by IDF — how rare it is across the corpus — so generic words like
    'dokumentiert' barely count while distinctive ones like 'Stichprobenprüfung' do.
    The boost is CAPPED per document so a flood of generic synonyms (e.g. an expanded
    query matching 'Personal/Kompetenz' in an unrelated title) can't dominate ranking."""
    import numpy as np
    model = get_embedding_model()
    qv = model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0].astype(np.float32)
    scores = (embeddings @ qv).astype(np.float32).copy()  # cosine similarity

    # IDF-weighted lexical boost. Stem-match handles German inflection; weight = how
    # distinctive the word is. Capped so stacked generic matches can't take over.
    german = translate_to_german(query)
    gstems = {stem(w) for w in tokenize(german) if len(w) >= 6}
    weighted = [(st, IDF.get(st, IDF_MAX)) for st in gstems]
    weighted = [(st, w) for st, w in weighted if w > 0.4]  # drop near-universal terms
    if weighted:
        for i, c in enumerate(chunks):
            fn = c["file"].lower()
            boost = min(sum(w for st, w in weighted if st in fn) * 0.06, BOOST_CAP)
            if boost:
                scores[i] += boost

    order = np.argsort(-scores)[:top_k]
    result = []
    for i in order:
        c = chunks[int(i)].copy()
        c["_score"] = float(scores[int(i)])
        result.append(c)
    return result

# ── Gemini API ────────────────────────────────────────────────────────────────
def get_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()
    return os.environ.get("GEMINI_API_KEY", "")

def ask_llm(question, relevant_chunks):
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key. Click Settings and add your Gemini API key."}

    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        short_text = chunk["text"][:350].strip()
        context_parts.append(f"[Source {i+1}: {chunk['file']}, Page {chunk['page']}]\n{short_text}")
    context = "\n\n---\n\n".join(context_parts)

    source_list = "\n".join(
        f"  Source {i+1}: {c['file']}, Page {c['page']}"
        for i, c in enumerate(relevant_chunks)
    )

    prompt = (
        "You are an expert ISO 9001 audit assistant.\n\n"
        f"QUESTION: {question}\n\n"
        "RELEVANT DOCUMENT EXCERPTS (in German — translate citations to English yourself):\n" + context + "\n\n"
        f"SOURCES TO CITE (include ALL of these in citations array):\n{source_list}\n\n"
        "Instructions:\n"
        "- Answer comprehensively based on the documents.\n"
        "- Only set answer_english to NO_ANSWER_FOUND if documents have zero relevant info.\n"
        "- In answer text, reference as [Source N, p.X].\n"
        "- Citations array MUST have one entry per source above.\n"
        "  Copy exact filename and page. Pick the most informative sentence.\n"
        "  original_german: most informative sentence from the source.\n"
        "  translated_english: English translation of that sentence. NEVER leave empty.\n"
        "- Respond ONLY with raw JSON, no markdown.\n\n"
        "{\n"
        '  "answer_english": "...",\n'
        '  "answer_german": "...",\n'
        '  "citations": [{"source_num":1,"file":"...","page":1,"original_german":"...","translated_english":"..."}],\n'
        '  "confidence": "high"\n'
        "}"
    )

    payload = json.dumps({
        "model": GEMINI_MODEL,
        "messages": [
            {"role": "system", "content": "You are an audit assistant. Respond only with valid JSON. No markdown, no backticks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": 6000,
        "reasoning_effort": "none",  # don't let 'thinking' eat the output budget (truncated answers)
    }).encode("utf-8")

    try:
        import time
        req = urllib.request.Request(
            GEMINI_URL,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )
        # Retry transient server overloads (503/500/502) — Gemini "high demand" spikes
        data = None
        for attempt in range(4):
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as e:
                if e.code in (500, 502, 503) and attempt < 3:
                    wait = 2 * (attempt + 1)
                    print(f"  Gemini {e.code} (overloaded), retrying in {wait}s...")
                    time.sleep(wait)
                    continue
                raise
        raw = data["choices"][0]["message"]["content"].strip()
        raw = raw.replace("```json", "").replace("```", "").strip()
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
        if not raw:
            return {"error": "Empty response from AI model - try again"}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # Try to salvage partial response
            try:
                import ast
                # Fix common issues: trailing commas, unescaped quotes
                raw = re.sub(r',\s*}', '}', raw)
                raw = re.sub(r',\s*]', ']', raw)
                return json.loads(raw)
            except:
                # Return whatever we can extract
                answer_en = re.search(r'"answer_english"\s*:\s*"([^"]*)"', raw)
                answer_de = re.search(r'"answer_german"\s*:\s*"([^"]*)"', raw)
                return {
                    "answer_english": answer_en.group(1) if answer_en else "Could not parse full response - try asking again",
                    "answer_german": answer_de.group(1) if answer_de else "Antwort konnte nicht verarbeitet werden",
                    "citations": [],
                    "confidence": "low"
                }
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8")
        try:
            msg = json.loads(body).get("error", {}).get("message", body)
        except:
            msg = body
        return {"error": f"Groq API error: {msg}"}
    except json.JSONDecodeError as e:
        return {"error": f"Could not parse response: {e}"}
    except Exception as e:
        return {"error": str(e)}

# ── File lookup (handles special chars) ───────────────────────────────────────
def find_pdf(fname):
    all_pdfs = get_all_pdfs()
    # Exact match
    for f, path in all_pdfs:
        if f == fname:
            return path
    # Fuzzy: normalize underscores and case
    norm = re.sub(r'_+', '_', fname).lower()
    for f, path in all_pdfs:
        if re.sub(r'_+', '_', f).lower() == norm:
            return path
    # Partial match
    for f, path in all_pdfs:
        if f.lower() in fname.lower() or fname.lower() in f.lower():
            return path
    return None

# ── HTTP Server ───────────────────────────────────────────────────────────────
INDEX      = []
EMBEDDINGS = None
IDF        = {}     # 8-char stem -> inverse document frequency (term distinctiveness)
IDF_MAX    = 4.0    # weight for a stem unseen in the corpus (treated as very rare)
REINDEX_LOCK = threading.Lock()

class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass

    def send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def send_file(self, path, ctype):
        with open(path, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", len(data))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(data)

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_GET(self):
        global INDEX, EMBEDDINGS, IDF, IDF_MAX
        path = urllib.parse.unquote(urlparse_path(self.path))

        if path in ("/", "/index.html"):
            self.send_file("index.html", "text/html; charset=utf-8")
        elif path.endswith(".html"):
            # Serve any local .html page (dashboard.html, operations2.html, sales.html …)
            fname = path.lstrip("/")
            if ".." not in fname and os.path.exists(fname):
                self.send_file(fname, "text/html; charset=utf-8")
            else:
                self.send_response(404); self.end_headers()
        elif path.startswith("/pdf/"):
            try:
                pdf_id = int(path[5:])
                all_pdfs = get_all_pdfs()
                if 0 <= pdf_id < len(all_pdfs):
                    _fname, fpath = all_pdfs[pdf_id]
                    self.send_file(fpath, "application/pdf")
                else:
                    self.send_response(404); self.end_headers()
            except:
                self.send_response(404); self.end_headers()
        elif path == "/api/pdflist":
            files = [f for f, _ in get_all_pdfs()]
            self.send_json({"files": files})
        elif path == "/api/status":
            pdf_count = len(get_all_pdfs()) if os.path.exists(PDF_DIR) else 0
            self.send_json({"indexed": len(INDEX), "pdf_count": pdf_count, "has_api_key": bool(get_api_key())})
        elif path == "/api/reindex":
            with REINDEX_LOCK:
                if os.path.exists(INDEX_FILE):
                    os.remove(INDEX_FILE)
                if os.path.exists(EMB_FILE):
                    os.remove(EMB_FILE)
                INDEX = build_index()
                if INDEX:
                    EMBEDDINGS = build_embeddings(INDEX)
                    IDF, IDF_MAX = build_idf(INDEX)
            self.send_json({"ok": True, "chunks": len(INDEX)})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        global INDEX, EMBEDDINGS
        path = urlparse_path(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/api/ask":
            question = body.get("question", "").strip()
            if not question:
                return self.send_json({"error": "No question provided"}, 400)
            if not INDEX:
                return self.send_json({"error": "No documents indexed. Click Re-index PDFs."}, 400)
            # Expand abbreviations/terse queries with the LLM's world knowledge
            # (e.g. 'MSL' -> 'Moisture Sensitivity Level, dry storage of ICs'), then
            # run hybrid semantic search. The multilingual model matches the expanded
            # English query directly against the German document vectors.
            api_key = get_api_key()
            # Only spend a Gemini call to expand terse/acronym queries (e.g. 'msl').
            # Full-sentence questions already retrieve well without expansion, so we
            # skip the extra call and save daily quota.
            if len(question.split()) <= 4:
                search_query = expand_query(question, api_key)
                print(f"Query: '{question}'  →  expanded: '{search_query}'")
            else:
                search_query = question
                print(f"Query: '{question}'")
            relevant = search(search_query, INDEX, EMBEDDINGS, top_k=40)
            if not relevant:
                return self.send_json({"error": "No relevant content found."})

            # ── File selection ──────────────────────────────────────────────────
            # Pick the top 3 highest-scoring distinct documents (a question can
            # legitimately be answered by more than one document).
            def _doc_num(fname):
                m = re.match(r'^([A-Za-z]{2}\s*\d+)', fname)
                return m.group(1).lower().replace(' ', '') if m else fname[:15].lower()

            best_per_file = {}
            for chunk in relevant:
                fname = chunk["file"]
                if fname not in best_per_file or chunk["_score"] > best_per_file[fname]["_score"]:
                    best_per_file[fname] = chunk
            sorted_files = sorted(best_per_file.values(), key=lambda c: c["_score"], reverse=True)
            seen_nums = {}
            deduped = []
            for c in sorted_files:
                dn = _doc_num(c["file"])
                if dn not in seen_nums:
                    seen_nums[dn] = True
                    deduped.append(c)
            # Keep only documents competitive with the best match: if one document
            # clearly dominates (big score gap) we show just it; if several are close
            # (ambiguous query) we show up to 3. Avoids padding clear answers with noise.
            if deduped:
                top_score = deduped[0]["_score"]
                selected_files = [c["file"] for c in deduped[:3]
                                  if c["_score"] >= top_score - 0.05][:3]
            else:
                selected_files = []
            selected_set = set(selected_files)
            print(f"  Top scores: " + " | ".join(f"{c['_score']:.0f} {c['file'][:30]}" for c in sorted_files[:4]))
            print(f"  Selected: {[f[:45] for f in selected_files]}")

            llm_chunks = [c for c in relevant if c["file"] in selected_set][:10]
            result = ask_llm(question, llm_chunks)
            result["chunks_used"] = len(llm_chunks)

            # ── Citation building ──────────────────────────────────────────────
            # Strategy:
            #  Pass 0 — pages mentioned inline in the answer text [filename, p.X]
            #            (the LLM gets these right in prose even when JSON is wrong)
            #  Pass 1 — pages from the LLM citations JSON
            #  Pass 2 — TF-IDF top pages to fill remaining slots (max 2 per file)
            llm_cit_map = {
                f"{c.get('file','')}|{c.get('page','')}": c
                for c in result.get("citations", [])
            }
            seen_fp = set()
            pages_per_file = {}
            final_cits = []

            # Build a lookup: (file, page) → chunk, for fast access
            relevant_chunk_map = {}
            for chunk in relevant:
                key = (chunk["file"], chunk["page"])
                if key not in relevant_chunk_map:
                    relevant_chunk_map[key] = chunk

            # Map "Source N" numbers back to filenames (LLM sometimes uses [Source N, p.X])
            source_num_to_file = {i+1: c["file"] for i, c in enumerate(llm_chunks)}

            def _add_pass0(fname, page):
                ckey = f"{fname}|{page}"
                if ckey in seen_fp or pages_per_file.get(fname, 0) >= 2:
                    return
                chunk = relevant_chunk_map.get((fname, page))
                if not chunk:
                    return
                seen_fp.add(ckey)
                pages_per_file[fname] = pages_per_file.get(fname, 0) + 1
                c = dict(llm_cit_map.get(ckey, {}))
                c["source_num"] = len(final_cits) + 1
                c["file"] = fname
                c["page"] = page
                if not c.get("original_german"):
                    c["original_german"] = chunk["text"][:80].strip()
                if not c.get("translated_english"):
                    c["translated_english"] = chunk["text"][:80].strip()
                final_cits.append(c)

            # Pass 0: pages mentioned in answer text — handles both formats:
            #   [filename, p.X]  and  [Source N, p.X] / [Source N, Page X]
            answer_text = (result.get("answer_english","") + " " + result.get("answer_german",""))
            # Format A: [filename fragment, p.X]
            for m in re.finditer(r'\[([A-Z]{2}[^,\]]{3,60}),\s*p(?:age)?\.?\s*(\d+)\]', answer_text, re.IGNORECASE):
                partial = m.group(1).strip()
                page = int(m.group(2))
                for fname in selected_set:
                    if partial[:12].lower() in fname.lower():
                        _add_pass0(fname, page)
                        break
            # Format B: [Source N, p.X] or [Source N, Page X]
            for m in re.finditer(r'\[Source\s+(\d+)[,\s]+[Pp](?:age)?\.?\s*(\d+)\]', answer_text):
                src_n = int(m.group(1))
                page = int(m.group(2))
                fname = source_num_to_file.get(src_n)
                if fname and fname in selected_set:
                    _add_pass0(fname, page)

            # Pass 1: LLM-cited pages (correct specific pages)
            for c in result.get("citations", []):
                fname = c.get("file", "")
                if fname not in selected_set:
                    continue
                if pages_per_file.get(fname, 0) >= 2:
                    continue
                key = f"{fname}|{c.get('page', '')}"
                if key in seen_fp:
                    continue
                seen_fp.add(key)
                pages_per_file[fname] = pages_per_file.get(fname, 0) + 1
                c2 = dict(c)
                c2["source_num"] = len(final_cits) + 1
                if not c2.get("translated_english"):
                    c2["translated_english"] = "[Translation not available for this page]"
                final_cits.append(c2)

            # Pass 2: fill remaining slots with TF-IDF top pages
            for chunk in relevant:
                fname = chunk["file"]
                if fname not in selected_set:
                    continue
                if pages_per_file.get(fname, 0) >= 2:
                    continue
                key = f"{fname}|{chunk['page']}"
                if key in seen_fp:
                    continue
                seen_fp.add(key)
                pages_per_file[fname] = pages_per_file.get(fname, 0) + 1
                if key in llm_cit_map:
                    c = dict(llm_cit_map[key])
                    c["source_num"] = len(final_cits) + 1
                    final_cits.append(c)
                else:
                    final_cits.append({
                        "source_num": len(final_cits) + 1,
                        "file": chunk["file"],
                        "page": chunk["page"],
                        "original_german": chunk["text"][:80].strip(),
                        "translated_english": "[Translation not available for this page]"
                    })
            result["citations"] = final_cits
            self.send_json(result)
        elif path == "/api/save_key":
            key = body.get("key", "").strip()
            if key:
                with open(API_KEY_FILE, "w") as f:
                    f.write(key)
                self.send_json({"ok": True})
            else:
                self.send_json({"error": "Empty key"}, 400)
        else:
            self.send_response(404); self.end_headers()

def urlparse_path(full_path):
    return full_path.split('?')[0]

def main():
    global INDEX, EMBEDDINGS, IDF, IDF_MAX
    os.makedirs(PDF_DIR, exist_ok=True)
    INDEX = load_index()
    if INDEX:
        EMBEDDINGS = load_embeddings(INDEX)
        IDF, IDF_MAX = build_idf(INDEX)
    else:
        print(f"No index. Add PDFs to '{PDF_DIR}/' and click Re-index.")
    print(f"\n✅ Server running → open http://localhost:{PORT}\n")
    HTTPServer(("", PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()