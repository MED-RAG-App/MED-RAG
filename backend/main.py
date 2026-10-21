import logging
import os
import re
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

load_dotenv()

from pdf_processor import process_pdf  # noqa: E402
from rag_pipeline import (  # noqa: E402
    COLLECTION,
    ask_question,
    delete_document,
    ensure_collection,
    index_pages,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("medrag.api")


# ============================================================
# APP
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    ensure_collection()
    yield


app = FastAPI(
    title="MED-RAG Secure Multilingual API",
    version="2.1.0",
    lifespan=lifespan,
)

# ============================================================
# CORS
# ============================================================

ALLOWED_ORIGINS = [
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]

ALLOWED_ORIGINS += [
    o.strip() for o in os.getenv("CORS_ORIGINS", "").split(",") if o.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)

# ============================================================
# SECURITY CONFIG
# ============================================================

MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE_MB", "15")) * 1024 * 1024
SAVE_UPLOADS = os.getenv("SAVE_UPLOADS", "true").lower() == "true"

UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads")).resolve()
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
UUID_RE = re.compile(r"^[0-9a-fA-F-]{36}$")

ALLOWED_CONTENT_TYPES = {"application/pdf", "application/x-pdf", "application/octet-stream"}


# ============================================================
# MODELS
# ============================================================

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    patient_id: str = Field(min_length=1, max_length=64)
    # Optional: restrict the search to specific documents of THIS patient.
    document_ids: List[str] = Field(default_factory=list, max_length=50)


# ============================================================
# VALIDATION HELPERS
# ============================================================

def validate_patient_id(patient_id: str) -> str:
    patient_id = (patient_id or "").strip()

    if not PATIENT_RE.fullmatch(patient_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid patient_id. Use only letters, numbers, underscore or hyphen (max 64).",
        )

    return patient_id


def clean_original_name(filename: str) -> str:
    # Treat both / and \ as separators so "..\\..\\x.pdf" is neutralised on every OS.
    name = Path((filename or "").replace("\\", "/")).name

    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Invalid filename.")

    if Path(name).suffix.lower() != ".pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    return name


def safe_storage_name(original: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]", "_", Path(original).stem)[:80] or "document"
    return f"{stem}_{uuid.uuid4().hex[:12]}.pdf"


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():
    return {"status": "ok", "collection": COLLECTION}


# ============================================================
# PDF UPLOAD
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    file: UploadFile = File(...),
    patient_id: str = Form(...),
):
    patient_id = validate_patient_id(patient_id)
    original_filename = clean_original_name(file.filename)

    if file.content_type and file.content_type.lower() not in ALLOWED_CONTENT_TYPES:
        await file.close()
        raise HTTPException(status_code=400, detail="Invalid content type. Upload a PDF.")

    try:
        pdf_bytes = await file.read(MAX_FILE_SIZE + 1)
    finally:
        await file.close()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")

    if len(pdf_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"PDF exceeds the {MAX_FILE_SIZE // (1024 * 1024)} MB limit.",
        )

    if not pdf_bytes.startswith(b"%PDF-"):
        raise HTTPException(
            status_code=400,
            detail="The uploaded file does not appear to be a valid PDF.",
        )

    # Parse (CPU-bound -> thread pool so the server stays responsive)
    try:
        pages = await run_in_threadpool(process_pdf, pdf_bytes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("PDF processing failed")
        raise HTTPException(status_code=400, detail="Could not process the PDF.") from exc

    document_id = str(uuid.uuid4())

    # Index (embedding + Qdrant)
    try:
        indexed = await run_in_threadpool(
            index_pages,
            patient_id=patient_id,
            document_id=document_id,
            filename=original_filename,
            pages=pages,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        log.exception("Indexing failed")
        try:
            await run_in_threadpool(delete_document, document_id)
        except Exception:
            log.exception("Rollback failed")
        raise HTTPException(status_code=500, detail="Could not index the document.") from exc

    # Optional local copy, written only AFTER success, inside UPLOAD_DIR only.
    if SAVE_UPLOADS:
        try:
            destination = (UPLOAD_DIR / safe_storage_name(original_filename)).resolve()

            if destination.parent != UPLOAD_DIR:
                raise ValueError("Path escape blocked.")

            destination.write_bytes(pdf_bytes)
        except Exception:
            log.exception("Could not save a local copy (indexing already succeeded)")

    return {
        "message": "PDF uploaded and indexed successfully.",
        "patient_id": patient_id,
        "document_id": document_id,
        "filename": original_filename,
        "pages": len(pages),
        "chunks": indexed["chunks"],
    }


# ============================================================
# ASK
# ============================================================

@app.post("/ask")
def ask(request: AskRequest):
    patient_id = validate_patient_id(request.patient_id)
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    document_ids = []

    for doc_id in request.document_ids:
        if not UUID_RE.fullmatch(doc_id or ""):
            raise HTTPException(status_code=400, detail="Invalid document_id.")
        document_ids.append(doc_id)

    try:
        return ask_question(
            question=question,
            patient_id=patient_id,
            document_ids=document_ids,
        )
    except HTTPException:
        raise
    except Exception as exc:
        log.exception("ask failed")
        raise HTTPException(
            status_code=502,
            detail="The answer service is temporarily unavailable. Please try again.",
        ) from exc