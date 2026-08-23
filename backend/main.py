from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pypdf import PdfReader
from io import BytesIO
from pydantic import BaseModel

from rag_pipeline import (
    create_chunks,
    generate_embeddings,
    store_embeddings,
    generate_answer,
)


app = FastAPI(
    title="MED-RAG API",
    description="Backend API for medical document question answering",
    version="1.0.0",
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
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
    }


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    file_content = await file.read()

    try:
        # 1. Extract text from PDF
        reader = PdfReader(BytesIO(file_content))

        pages_text = []

        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)

        extracted_text = "\n\n".join(pages_text).strip()

        if not extracted_text:
            raise HTTPException(
                status_code=400,
                detail="No extractable text found in the PDF.",
            )

        # 2. Create chunks
        chunks = create_chunks(extracted_text)

        # 3. Generate embeddings
        embeddings = generate_embeddings(chunks)

        # 4. Store in Qdrant
        stored_count = store_embeddings(
            chunks,
            embeddings,
            file.filename,
        )

        return {
            "filename": file.filename,
            "pages": len(reader.pages),
            "characters": len(extracted_text),
            "chunks": len(chunks),
            "embeddings": len(embeddings),
            "stored_in_qdrant": stored_count,
            "message": "PDF processed and stored successfully.",
        }

    except HTTPException:
        raise

    except Exception as exc:
        print("UPLOAD ERROR:", repr(exc))

        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(exc)}",
        )
class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
def ask_question(request: QuestionRequest):
    result = generate_answer(request.question, limit=10)

    return result