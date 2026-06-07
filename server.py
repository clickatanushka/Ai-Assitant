#!/usr/bin/env python3
"""
Audit Assistant Backend — Gemini + OCR Edition
Run: python server.py
"""

import os, json, re, math, urllib.request, urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse

PDF_DIR       = "./pdfs"
INDEX_FILE    = "./index.json"
API_KEY_FILE  = "./api_key.txt"
PORT          = 8000
CHUNK_SIZE    = 150
CHUNK_OVERLAP = 20
TOP_K         = 10
GROQ_MODEL    = "llama-3.3-70b-versatile"

# ── PDF Extraction (text + OCR fallback) ─────────────────────────────────────
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
            if len(chunk_words) < 20:
                continue
            chunk_text = " ".join(chunk_words)
            chunks.append({
                "file": filename,
                "page": p["page"],
                "text": chunk_text,
                "snippet": chunk_text[:200].replace("\n", " ")
            })
    return chunks

def build_index():
    print("Building index from PDFs...")
    all_chunks = []
    os.makedirs(PDF_DIR, exist_ok=True)
    pdf_files = [f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]
    if not pdf_files:
        print(f"No PDFs found in {PDF_DIR}/")
        return []
    for i, filename in enumerate(sorted(pdf_files)):
        path = os.path.join(PDF_DIR, filename)
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
            data = json.load(f)
        if len(data) < 200:
            print(f"Index has only {len(data)} chunks — looks stale. Delete index.json and re-index.")
        return data
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

def search(query, chunks, chunk_tfs, idf, top_k=TOP_K):
    q_words = tokenize(query)
    q_lower = query.lower()
    scores = []
    for i, (chunk, tf) in enumerate(zip(chunks, chunk_tfs)):
        # Base TF-IDF score
        score = sum(tf.get(w, 0) * idf.get(w, 0) for w in q_words)
        fname = chunk["file"].lower()
        text_lower = chunk["text"].lower()

        # Count how many query words appear in filename
        fname_hits = sum(1 for w in q_words if len(w) > 1 and w in fname)
        score += fname_hits * 150

        # Extra bonus if ALL significant query words are in filename
        sig_words = [w for w in q_words if len(w) > 2]
        if sig_words and all(w in fname for w in sig_words):
            score += 500

        # Bonus for query words in text
        for w in q_words:
            if len(w) > 3 and w in text_lower:
                score += 3

        scores.append((score, i))
    scores.sort(reverse=True)
    return [chunks[i] for score, i in scores[:top_k] if score > 0]

# ── Groq API ──────────────────────────────────────────────────────────────────
def get_api_key():
    if os.path.exists(API_KEY_FILE):
        with open(API_KEY_FILE) as f:
            return f.read().strip()
    return os.environ.get("GROQ_API_KEY", "")

def ask_llm(question, relevant_chunks):
    api_key = get_api_key()
    if not api_key:
        return {"error": "No API key. Click Settings and add your Groq API key."}

    context_parts = []
    for i, chunk in enumerate(relevant_chunks):
        # Send only first 300 chars of each chunk to save tokens
        short_text = chunk['text'][:500].strip()
        context_parts.append(
            f"[Source {i+1}: {chunk['file']}, Page {chunk['page']}]\n{short_text}"
        )
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert audit assistant. The documents below are German audit documents.

QUESTION (in English): {question}

RELEVANT DOCUMENT EXCERPTS:
{context}

Instructions:
- List ALL sources that contain relevant information, not just one.
- If the documents contain ANY partial information related to the question, provide that information as the answer.
- Only set answer_english to "NO_ANSWER_FOUND" if the documents contain absolutely zero relevant information.
- Always respond ONLY with raw JSON, no markdown, no backticks.

JSON structure:
{{
  "answer_english": "...",
  "answer_german": "...",
"citations": [
    {{
      "source_num": 1,
      "file": "filename.pdf",
      "page": 5,
      "original_german": "exact German excerpt from this source",
      "translated_english": "English translation of that excerpt"
    }}
  ],
  "confidence": "high"
}}"""

    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = json.dumps({
        "model": GROQ_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 1500
    }).encode("utf-8")

    try:
        req = urllib.request.Request(url, data=payload, headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
            "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120.0.0.0 Safari/537.36"
        }, method="POST")
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
        return json.loads(raw)
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

# ── HTTP Server ───────────────────────────────────────────────────────────────
INDEX     = []
CHUNK_TFS = []
IDF       = {}

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
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            self.send_file("index.html", "text/html; charset=utf-8")
        elif path == "/dashboard.html":
            self.send_file("dashboard.html", "text/html; charset=utf-8")
        elif path.startswith("/pdfs/"):
            import urllib.parse
            fname = urllib.parse.unquote(path[6:])
            fpath = os.path.join(PDF_DIR, fname)
            if os.path.exists(fpath):
                self.send_file(fpath, "application/pdf")
            else:
                self.send_response(404); self.end_headers()
        elif path == "/api/status":
            pdf_count = len([f for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")]) if os.path.exists(PDF_DIR) else 0
            self.send_json({"indexed": len(INDEX), "pdf_count": pdf_count, "has_api_key": bool(get_api_key())})
        elif path == "/api/reindex":
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
        path = urlparse(self.path).path
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        if path == "/api/ask":
            question = body.get("question", "").strip()
            if not question:
                return self.send_json({"error": "No question provided"}, 400)
            if not INDEX:
                return self.send_json({"error": "No documents indexed. Click Re-index PDFs."}, 400)
            relevant = search(question, INDEX, CHUNK_TFS, IDF)
            if not relevant:
                return self.send_json({"error": "No relevant content found."})
            result = ask_llm(question, relevant)
            result["chunks_used"] = len(relevant)
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

def main():
    global INDEX, CHUNK_TFS, IDF
    os.makedirs(PDF_DIR, exist_ok=True)
    INDEX = load_index()
    if INDEX:
        CHUNK_TFS, IDF = build_tfidf(INDEX)
    else:
        print(f"No index found. Add PDFs to '{PDF_DIR}/' and click Re-index.")
    print(f"\n✅ Server running → open http://localhost:{PORT}\n")
    HTTPServer(("", PORT), Handler).serve_forever()

if __name__ == "__main__":
    main()
