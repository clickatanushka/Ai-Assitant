"""FastAPI entrypoint — the whole backend.

One function rather than a directory of handlers: module-level clients stay warm
between invocations, and the shared `audit` package imports normally.
"""

import hmac
import os

from fastapi import FastAPI, File, Header, HTTPException, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from audit import config, ingest, retrieve, store

app = FastAPI(title="Audit Assistant")

PUBLIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public")


def require_admin(password: str | None) -> None:
    """Uploading and deleting are gated; asking is not."""
    if not config.ADMIN_PASSWORD:
        raise HTTPException(500, "ADMIN_PASSWORD is not configured on the server")
    if not password or not hmac.compare_digest(password, config.ADMIN_PASSWORD):
        raise HTTPException(401, "Incorrect password")


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)


class IngestRequest(BaseModel):
    blob_url: str
    filename: str


@app.post("/api/ask")
def api_ask(body: AskRequest):
    question = body.question.strip()
    if not question:
        raise HTTPException(400, "No question provided")
    try:
        return retrieve.ask(question)
    except RuntimeError as e:          # missing configuration
        raise HTTPException(500, str(e)) from e


@app.get("/api/documents")
def api_documents():
    return {"documents": store.list_documents()}


@app.get("/api/documents/{doc_id}")
def api_document(doc_id: str):
    """Full transcription of one document — backs the in-app source viewer."""
    doc = store.get_document(doc_id)
    if not doc:
        raise HTTPException(404, "No such document")
    return doc


@app.post("/api/ingest")
def api_ingest(body: IngestRequest,
               x_admin_password: str | None = Header(default=None)):
    require_admin(x_admin_password)
    try:
        pdf = store.fetch_blob(body.blob_url)
        doc = ingest.ingest_pdf(pdf, filename=body.filename, blob_url=body.blob_url)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "ok": True,
        "doc_id": doc["doc_id"],
        "file": doc["file"],
        "pages": len(doc["pages"]),
        "chunks": doc["n_chunks"],
    }


@app.post("/api/ingest/upload")
async def api_ingest_upload(file: UploadFile = File(...),
                            x_admin_password: str | None = Header(default=None)):
    """Direct browser upload. Vercel caps a function request body at 4.5 MB, so
    anything larger has to go through Blob and /api/ingest instead."""
    require_admin(x_admin_password)
    if not (file.filename or "").lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files can be indexed")
    pdf = await file.read()
    if not pdf:
        raise HTTPException(400, "Empty file")
    try:
        doc = ingest.ingest_pdf(pdf, filename=os.path.basename(file.filename))
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    return {
        "ok": True,
        "doc_id": doc["doc_id"],
        "file": doc["file"],
        "pages": len(doc["pages"]),
        "chunks": doc["n_chunks"],
    }


@app.delete("/api/documents/{doc_id}")
def api_delete(doc_id: str, x_admin_password: str | None = Header(default=None)):
    require_admin(x_admin_password)
    existed = store.delete_document(doc_id)
    if not existed:
        raise HTTPException(404, "No such document")
    return {"ok": True}


@app.get("/api/status")
def api_status():
    configured = {
        "gemini": bool(config.GEMINI_API_KEY),
        "vector": bool(config.UPSTASH_VECTOR_URL and config.UPSTASH_VECTOR_TOKEN),
        "admin_password": bool(config.ADMIN_PASSWORD),
    }
    documents = store.document_count() if configured["vector"] else None
    return {"configured": configured, "documents": documents, **store.index_stats()}


@app.get("/")
def root():
    """Redirect rather than serve the file.

    Vercel serves `public/` as static assets and does not put it in the function
    bundle, so reading index.html from disk here 500s in production while working
    fine locally. A redirect is correct in both.
    """
    return RedirectResponse("/index.html")


@app.exception_handler(HTTPException)
def http_error(_request, exc: HTTPException):
    return JSONResponse({"error": exc.detail}, status_code=exc.status_code)


@app.exception_handler(RequestValidationError)
def validation_error(_request, exc: RequestValidationError):
    """Same {"error": ...} shape as every other failure, so the frontend has one
    thing to read instead of two."""
    first = (exc.errors() or [{}])[0]
    field = ".".join(str(p) for p in first.get("loc", ()) if p != "body")
    message = first.get("msg", "Invalid request")
    return JSONResponse({"error": f"{field}: {message}" if field else message},
                        status_code=422)


if os.path.isdir(PUBLIC_DIR):
    app.mount("/", StaticFiles(directory=PUBLIC_DIR, html=True), name="public")
