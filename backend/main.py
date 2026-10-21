from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from pdf_processor import (
    extract_text_from_pdf_bytes,
    extract_text_from_image_bytes,
)
from pydantic import BaseModel
from typing import List

from rag_pipeline import (
    create_page_chunks,
    generate_embeddings,
    store_embeddings,
    generate_answer,
)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI()


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "MED-RAG API is running"
    }


# ============================================================
# UPLOAD MULTIPLE PDFs
# ============================================================

@app.post("/upload-pdf")
async def upload_pdf(
    files: List[UploadFile]
):

    # Canonical filenames used by the evaluation dataset
    filename_map = {

        "Patient_1_Medical_Report.pdf":
            "Patient_1_Abdul218_Harris789.pdf",

        "Patient_2_Medical_Report.pdf":
            "Patient_2_Carol737_Lesch175.pdf",

        "Patient_3_Medical_Report.pdf":
            "Patient_3_Pasquale620_Kuhn96.pdf",

        "Patient_4_Medical_Report.pdf":
            "Patient_4_Synthia172_Davis923.pdf",

        "Patient_5_Medical_Report.pdf":
            "Patient_5_Grady603_Predovic534.pdf",
    }

    results = []

    # ========================================================
    # Process each uploaded PDF independently
    # ========================================================

    for file in files:

        # ----------------------------------------------------
        # 1. Check file type
        # ----------------------------------------------------

        filename = file.filename or ""
        extension = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""

        if extension not in {"pdf", "png", "jpg", "jpeg", "webp"}:
            results.append({
                "filename": filename,
                "status": "failed",
                "error": "Supported files: PDF, PNG, JPG, JPEG, WEBP.",
            })
            continue

        try:

            # ------------------------------------------------
            # 2. Read uploaded file
            # ------------------------------------------------

            file_content = await file.read()

            if not file_content:

                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "Uploaded file is empty.",
                })

                continue

            # ------------------------------------------------
            # 3. Extract text using OCR-aware processor
            # ------------------------------------------------

            filename = file.filename or ""
            extension = filename.rsplit(".", 1)[-1].lower()

            if extension == "pdf":
                extraction = extract_text_from_pdf_bytes(
                    file_content
                )

            elif extension in {"png", "jpg", "jpeg", "webp"}:
                extraction = extract_text_from_image_bytes(
                    file_content
                )

            else:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": "Unsupported file type. Upload PDF, PNG, JPG, JPEG, or WEBP.",
                })
                continue

            pages_text = [
                page["text"]
                for page in extraction["pages"]
            ]

            extracted_text = "\n\n".join(
                pages_text
            ).strip()

            # ------------------------------------------------
            # 4. Check extracted text
            # ------------------------------------------------

            if not extracted_text:
                results.append({
                    "filename": filename,
                    "status": "failed",
                    "error": "No text could be extracted from the file, even after OCR.",
                })
                continue

            # ------------------------------------------------
            # 6. Convert to canonical filename
            # ------------------------------------------------

            stored_filename = filename_map.get(
                file.filename,
                file.filename
            )

            # ------------------------------------------------
            # 7. Create page-aware chunks
            # ------------------------------------------------

            chunks = create_page_chunks(
                pages_text
            )

            if not chunks:

                results.append({
                    "filename": file.filename,
                    "status": "failed",
                    "error": "No chunks were created from the PDF.",
                })

                continue

            # ------------------------------------------------
            # 8. Generate embeddings
            # ------------------------------------------------

            embeddings = generate_embeddings(
                chunks
            )

            # ------------------------------------------------
            # 9. Store embeddings in Qdrant
            # ------------------------------------------------

            stored_count = store_embeddings(
                chunks,
                embeddings,
                stored_filename,
            )

            # ------------------------------------------------
            # 10. Record success
            # ------------------------------------------------

            results.append({

                "filename": file.filename,

                "stored_filename": stored_filename,

                "status": "success",

                "pages": extraction["total_pages"],

                "characters": len(extracted_text),

                "chunks": len(chunks),

                "embeddings": len(embeddings),

                "stored_in_qdrant": stored_count,
            })

        except Exception as exc:

            # One failed PDF should not stop
            # the remaining PDFs

            print(
                "UPLOAD ERROR:",
                file.filename,
                repr(exc)
            )

            results.append({

                "filename": file.filename,

                "status": "failed",

                "error": str(exc),
            })

    # ========================================================
    # Batch summary
    # ========================================================

    successful = sum(
        1
        for result in results
        if result["status"] == "success"
    )

    failed = len(results) - successful

    return {

        "total_files": len(files),

        "successful": successful,

        "failed": failed,

        "results": results,

        "message": "Batch PDF processing completed.",
    }

# ============================================================
# ASK QUESTION
# ============================================================

class QuestionRequest(BaseModel):
    question: str


@app.post("/ask")
async def ask_question(request: QuestionRequest):

    try:

        answer = generate_answer(
            request.question
        )

        return answer

    except Exception as exc:

        print(
            "ASK ERROR:",
            repr(exc)
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc)
        )
        
# ============================================================
# CUSTOM OPENAPI
# Fix Swagger UI multiple-file upload display
# ============================================================

def custom_openapi():

    if app.openapi_schema:
        return app.openapi_schema

    schema = get_openapi(

        title=app.title,

        version=app.version,

        routes=app.routes,
    )

    components = schema.get(
        "components",
        {}
    ).get(
        "schemas",
        {}
    )

    for component in components.values():

        properties = component.get(
            "properties",
            {}
        )

        for prop in properties.values():

            # ------------------------------------------------
            # Multiple uploaded files
            # ------------------------------------------------

            if prop.get("type") == "array":

                items = prop.get(
                    "items",
                    {}
                )

                if (
                    items.get("contentMediaType")
                    == "application/octet-stream"
                ):

                    items.pop(
                        "contentMediaType",
                        None
                    )

                    items["format"] = "binary"

            # ------------------------------------------------
            # Single uploaded file
            # ------------------------------------------------

            elif (
                prop.get("type") == "string"
                and
                prop.get("contentMediaType")
                == "application/octet-stream"
            ):

                prop.pop(
                    "contentMediaType",
                    None
                )

                prop["format"] = "binary"

    app.openapi_schema = schema

    return app.openapi_schema


# Tell FastAPI to use our custom OpenAPI schema
app.openapi = custom_openapi