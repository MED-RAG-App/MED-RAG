from fastapi import FastAPI, File, UploadFile, HTTPException
from pypdf import PdfReader
from io import BytesIO

from rag_pipeline import create_chunks


app = FastAPI(
    title="MED-RAG API",
    description="Backend API for medical document question answering",
    version="1.0.0",
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

        chunks = create_chunks(extracted_text)

        return {
            "filename": file.filename,
            "pages": len(reader.pages),
            "characters": len(extracted_text),
            "chunks": len(chunks),
            "message": "PDF processed successfully.",
        }

    except HTTPException:
        raise

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=f"PDF processing failed: {str(error)}",
        )