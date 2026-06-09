#!/usr/bin/env python3

"""
Audit Assistant Backend — Groq + OCR Edition
Run: python server.py
"""

import os, json, re, math, urllib.request, urllib.error, urllib.parse, threading
from http.server import HTTPServer, BaseHTTPRequestHandler

PDF_DIR       = "./pdfs"
INDEX_FILE    = "./index.json"
API_KEY_FILE  = "./api_key.txt"
PORT          = 8000
CHUNK_SIZE    = 100
CHUNK_OVERLAP = 20
TOP_K         = 15
GROQ_MODEL    = "llama-3.3-70b-versatile"

# Local abbreviation expansions — Groq reliably misidentifies SMT/PCB acronyms.
# When any token matches here, Groq is skipped entirely for the query expansion.
ABBREV_MAP = {
    "msl":  "MSL MSD Trockenlagerung Trockenlagerschrank feuchtigkeitsempfindlichkeit feuchteempfindlichkeit",
    "esd":  "ESD Elektrostatik elektrostatisch Entladung antistatisch ESD-Schutz Erdung",
    "iqc":  "IQC Wareneingangsprüfung Wareneingangskontrolle Eingangsprüfung Warenannahme Wareneingang",
    "smt":  "SMT Oberflächenmontage SMD Bestückung Reflow Lötpaste Schablone",
    "pcb":  "PCB Leiterplatte Platine Schaltung Leiterplattenbestückung",
    "aoi":  "AOI optische Inspektion Sichtprüfung automatisch Kamera Bildverarbeitung",
    "ict":  "ICT In-Circuit-Test Leiterplattentest Elektrische Prüfung Nadeltest",
    "spc":  "SPC statistische Prozesskontrolle Regelkarte Prozessregelung Qualitätsregelung",
    "ppm":  "PPM Fehlerrate Ausschuss Qualitätskennzahl Reklamationsquote",
    "bga":  "BGA Löten Reflow Lotperlen Ball-Grid Röntgen",
    "fifo": "FIFO Reihenfolge Lagerung Umlauf Verbrauchsreihenfolge",
    "rohs": "RoHS bleifrei Umweltschutz Gefahrstoffe Lötzinn Richtlinie",
    "fmea": "FMEA Fehleranalyse Risikoanalyse Fehler-Möglichkeit Risikoprioritätszahl",
    "ppap": "PPAP Erstmuster Erstmusterprüfung Bemusterung Produktionsfreigabe",
    "8d":   "8D 8D-Report Reklamation Fehleranalyse Abstellmaßnahme Ursachenanalyse",
    "oee":  "OEE Gesamtanlageneffektivität Anlagenverfügbarkeit Leistung Verfügbarkeit",
}

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

# ── TF-IDF Search ─────────────────────────────────────────────────────────────
def tokenize(text):
    return re.findall(r'\b\w+\b', text.lower())

def build_tfidf(chunks):
    print("Building search index...")
    N = len(chunks)
    df = {}
    chunk_tfs = []
    for chunk in chunks:
        words = tokenize(chunk["text"])
        tf = {}
        for w in words:
            tf[w] = tf.get(w, 0) + 1
        chunk_tfs.append(tf)
        for w in set(words):
            df[w] = df.get(w, 0) + 1
    idf = {w: math.log((N + 1) / (cnt + 1)) for w, cnt in df.items()}
    print(f"Search index ready ({len(df)} unique terms, {N} chunks)")
    return chunk_tfs, idf

def _compound_root(w, doc_word_set):
    """Match if w and a doc word share a compound prefix that is ≥6 chars AND covers
    at least 60% of the longer word.  The 60% guard prevents a short common prefix
    (e.g. 'feuchtigkeit' ⊂ 'feuchtigkeitsempfindlichkeit') from triggering a match
    when the words are semantically very different in scope."""
    for dw in doc_word_set:
        shorter = min(len(w), len(dw))
        longer  = max(len(w), len(dw))
        if shorter >= 6 and shorter >= longer * 0.60:
            if w.startswith(dw) or dw.startswith(w):
                return dw
    return None

def search(query, chunks, chunk_tfs, idf, top_k=TOP_K):
    q_words = tokenize(query)
    scores = []
    for i, (chunk, tf) in enumerate(zip(chunks, chunk_tfs)):
        doc_len = max(sum(tf.values()), 1)
        doc_word_set = set(tf.keys())

        # TF-IDF: exact match + 70% credit for compound-root match.
        # This lets "Kalibrierungsplan" (query) score off "Kalibrierung" (doc token).
        tfidf = 0.0
        for w in q_words:
            if w in tf:
                tfidf += (tf[w] / doc_len) * idf.get(w, 0)
            else:
                root = _compound_root(w, doc_word_set)
                if root:
                    tfidf += (tf[root] / doc_len) * idf.get(root, 0) * 0.7
        score = tfidf * 1000

        fname = chunk["file"].lower()
        text_lower = chunk["text"].lower()
        # Filename match bonus
        fname_hits = sum(1 for w in q_words if len(w) > 1 and w in fname)
        score += fname_hits * 150
        sig = [w for w in q_words if len(w) > 2]
        if sig and all(w in fname for w in sig):
            score += 500

        # Word-boundary hit count (prevents "plan" matching inside "Produktionsplanung")
        hit_count = 0
        for w in q_words:
            if len(w) <= 1:
                continue
            if re.search(r'\b' + re.escape(w) + r'\b', text_lower):
                hit_count += 1
            elif _compound_root(w, doc_word_set):
                hit_count += 0.5
        score += hit_count * 25
        if len(q_words) > 0 and hit_count / len(q_words) > 0.4:
            score += 300
        scores.append((score, i))
    scores.sort(reverse=True)
    result = []
    for score, i in scores[:top_k]:
        if score > 0:
            c = chunks[i].copy()
            c["_score"] = score
            result.append(c)
    return result

# ── Groq API ──────────────────────────────────────────────────────────────────
def get_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()
    return os.environ.get("GROQ_API_KEY", "")

def expand_query_to_keywords(question, api_key):
    """Return German search keywords for the question.

    If the question contains a known SMT/PCB abbreviation, use only the local
    ABBREV_MAP — Groq reliably misidentifies these and adds noise that drowns
    out the correct documents.  For all other questions, call Groq.
    """
    tokens = [t.strip("?!.,;:").lower() for t in question.split()]
    local_parts = [ABBREV_MAP[t] for t in tokens if t in ABBREV_MAP]

    if local_parts:
        # Skip Groq — local expansion is cleaner and Groq would corrupt it
        return " ".join(local_parts)

    # No known abbreviation → ask Groq for German keyword expansion
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": (
                "You are an expert in German quality management and ISO 9001 audit documents. "
                "Given an English question, output 8–12 German words/compound-words that would "
                "literally appear inside German audit process documents (Prozessbeschreibungen) "
                "that answer this question. Include synonyms, compound forms, and abbreviations. "
                "Return ONLY a space-separated list. No sentences, no punctuation, no English."
            )},
            {"role": "user", "content": question}
        ],
        "temperature": 0,
        "max_tokens": 120
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}", "User-Agent": "Mozilla/5.0"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"].strip()
    except:
        return question

def ask_llm(question, relevant_chunks):
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key. Click Settings and add your Groq API key."}

    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        short_text = chunk['text'][:350].strip()
        context_parts.append(f"[Source {i+1}: {chunk['file']}, Page {chunk['page']}]\n{short_text}")
    context = "\n\n---\n\n".join(context_parts)

    prompt = (
        "You are an expert audit assistant. The documents below are German audit documents.\n\n"
        f"QUESTION (in English): {question}\n\n"
        "RELEVANT DOCUMENT EXCERPTS:\n" + context + "\n\n"
        "Instructions:\n"
        "- If the documents contain ANY relevant information, provide a comprehensive answer.\n"
        "- IMPORTANT: Include ALL sources that are relevant, not just one.\n"
        "- Only cite sources that directly answer the question.\n"
        "- Only set answer_english to NO_ANSWER_FOUND if documents have zero relevant info.\n"
        "- Respond ONLY with raw JSON, no markdown, no backticks.\n\n"
        "JSON structure:\n"
        "{\n"
        '  "answer_english": "Full answer in English citing [Source N, Page X]",\n'
        '  "answer_german": "Same answer in formal German",\n'
        '  "citations": [\n'
        '    {\n'
        '      "source_num": 1,\n'
        '      "file": "exact filename.pdf",\n'
        '      "page": 1,\n'
        '      "original_german": "brief 10-word German excerpt only",\n'
        '      "translated_english": "English translation"\n'
        '    }\n'
        '  ],\n'
        '  "confidence": "high"\n'
        "}"
    )

    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [
            {"role": "system", "content": "You are an audit assistant. Respond only with valid JSON. No markdown, no backticks."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "seed": 42,
        "max_tokens": 2500,
    }).encode("utf-8")

    try:
        req = urllib.request.Request(
            "https://api.groq.com/openai/v1/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "Mozilla/5.0"
            },
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
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
INDEX     = []
CHUNK_TFS = []
IDF       = {}
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
        global INDEX, CHUNK_TFS, IDF
        path = urllib.parse.unquote(urlparse_path(self.path))

        if path in ("/", "/index.html"):
            self.send_file("index.html", "text/html; charset=utf-8")
        elif path == "/dashboard.html":
            self.send_file("dashboard.html", "text/html; charset=utf-8")
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
                INDEX = build_index()
                if INDEX:
                    CHUNK_TFS, IDF = build_tfidf(INDEX)
            self.send_json({"ok": True, "chunks": len(INDEX)})
        else:
            self.send_response(404); self.end_headers()

    def do_POST(self):
        global INDEX, CHUNK_TFS, IDF
        path = urlparse_path(self.path)
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/api/ask":
            question = body.get("question", "").strip()
            if not question:
                return self.send_json({"error": "No question provided"}, 400)
            if not INDEX:
                return self.send_json({"error": "No documents indexed. Click Re-index PDFs."}, 400)
            api_key = get_api_key()
            german_keywords = expand_query_to_keywords(question, api_key)
            print(f"Query: '{question}' → Keywords: '{german_keywords}'")
            combined_query = question + " " + german_keywords
            relevant = search(combined_query, INDEX, CHUNK_TFS, IDF)
            if not relevant:
                return self.send_json({"error": "No relevant content found."})
            result = ask_llm(question, relevant)
            result["chunks_used"] = len(relevant)

            # Determine which files to show as sources.
            # Strategy: rank files by their highest-scoring chunk, then walk down
            # the list stopping when a consecutive file drops >20% from the prior.
            # Cap at 3 unique files — audit queries rarely have more than 3 true sources.
            best_per_file = {}
            for chunk in relevant:
                fname = chunk["file"]
                if fname not in best_per_file or chunk["_score"] > best_per_file[fname]["_score"]:
                    best_per_file[fname] = chunk
            sorted_files = sorted(best_per_file.values(), key=lambda c: c["_score"], reverse=True)

            # Always show at least 2 documents (the top-2 are almost always both relevant).
            # From position 3 onward, stop if the next file drops >20% from the previous.
            selected_files = [f["file"] for f in sorted_files[:min(2, len(sorted_files))]]
            for i in range(2, min(len(sorted_files), 5)):
                prev_score = sorted_files[i-1]["_score"]
                curr_score = sorted_files[i]["_score"]
                if curr_score / prev_score < 0.80:
                    break
                selected_files.append(sorted_files[i]["file"])

            selected_set = set(selected_files)
            print(f"  Scores: " + " | ".join(f"{c['_score']:.0f} {c['file'][:25]}" for c in sorted_files[:4]))
            print(f"  Selected files: {[f[:30] for f in selected_files]}")

            # Build final citations: one entry per unique (file, page) in selected files,
            # enriched with LLM translations where the LLM happened to cite that page.
            llm_cit_map = {
                f"{c.get('file','')}|{c.get('page','')}": c
                for c in result.get("citations", [])
            }
            seen_fp = set()
            final_cits = []
            for chunk in relevant:
                if chunk["file"] not in selected_set:
                    continue
                key = f"{chunk['file']}|{chunk['page']}"
                if key in seen_fp:
                    continue
                seen_fp.add(key)
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
                        "translated_english": ""
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
    global INDEX, CHUNK_TFS, IDF
    os.makedirs(PDF_DIR, exist_ok=True)
    INDEX = load_index()
    if INDEX:
        CHUNK_TFS, IDF = build_tfidf(INDEX)
    else:
        print(f"No index. Add PDFs to '{PDF_DIR}/' and click Re-index.")
    print(f"\n✅ Server running → open http://localhost:{PORT}\n")
    HTTPServer(("", PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()