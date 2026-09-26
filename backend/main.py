from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid

from pdf_processor import (
    extract_text_from_pdf_bytes,
    extract_text_from_image_bytes,
)

from rag_pipeline import (
    create_document_chunks,
    generate_embeddings,
    store_embeddings,
    generate_answer,
)


app = FastAPI(
    title="MED-RAG API",
    description="AI Medical Document Assistant with OCR and RAG",
    version="4.0.0",
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    return {
        "message": "MED-RAG backend is running",
        "status": "success",
        "features": [
            "PDF",
            "Images",
            "OCR",
            "RAG",
            "Source Pages",
            "Document Isolation",
        ],
    }


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload-pdf")
async def upload_document(file: UploadFile = File(...)):

    filename = file.filename or ""

    allowed = [
        ".pdf",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
    ]

    if not any(filename.lower().endswith(ext) for ext in allowed):
        raise HTTPException(
            status_code=400,
            detail="Only PDF, PNG, JPG, JPEG and WEBP files are supported.",
        )

    try:
        file_content = await file.read()

        # -------------------------------------------------
        # EXTRACT TEXT
        # -------------------------------------------------

        if filename.lower().endswith(".pdf"):
            extraction = extract_text_from_pdf_bytes(file_content)
        else:
            extraction = extract_text_from_image_bytes(file_content)

        extracted_text = extraction["text"].strip()

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No readable text was found in the document.",
            )

        # -------------------------------------------------
        # CREATE UNIQUE DOCUMENT ID
        # -------------------------------------------------

        document_id = str(uuid.uuid4())

        # -------------------------------------------------
        # CREATE PAGE-AWARE CHUNKS
        # -------------------------------------------------

        chunks = create_document_chunks(
            extraction["pages"]
        )

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No usable document content was found.",
            )

        texts = [
            item["text"]
            for item in chunks
        ]

        # -------------------------------------------------
        # EMBEDDINGS
        # -------------------------------------------------

        embeddings = generate_embeddings(texts)

        # -------------------------------------------------
        # STORE IN QDRANT
        # -------------------------------------------------

        stored_count = store_embeddings(
            chunks,
            embeddings,
            filename,
            document_id,
        )

        return {
            "document_id": document_id,
            "filename": filename,
            "pages": extraction["total_pages"],
            "characters": len(extracted_text),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "stored_in_qdrant": stored_count,
            "ocr_used": extraction["ocr_used"],
            "ocr_pages": extraction["ocr_pages"],
            "average_ocr_confidence": extraction[
                "average_ocr_confidence"
            ],
            "message": "Document processed successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {str(error)}",
        )


class QuestionRequest(BaseModel):

    question: str

    documents: list[str] = []

    document_ids: list[str] = []


@app.post("/ask")
def ask_question(request: QuestionRequest):

    question = request.question.strip()

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    return generate_answer(
        question,
        limit=5,
        filenames=request.documents,
        document_ids=request.document_ids,
    )