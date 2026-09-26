from typing import List
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# =========================================================
# ENVIRONMENT
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")


QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError(
        "Qdrant environment variables are missing."
    )


if not GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing."
    )


# =========================================================
# CLIENTS
# =========================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


COLLECTION_NAME = "medical_documents"


# =========================================================
# EMBEDDING MODEL
# =========================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# =========================================================
# TEXT CHUNKING
# =========================================================

def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200,
) -> List[str]:

    if not text or not text.strip():
        return []

    if chunk_size <= 0:
        raise ValueError(
            "chunk_size must be greater than 0."
        )

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    text = text.strip()

    chunks = []

    start = 0
    step = chunk_size - overlap

    while start < len(text):

        chunk = text[
            start:start + chunk_size
        ].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# =========================================================
# PAGE-AWARE CHUNKING
# =========================================================

def create_document_chunks(pages):

    result = []

    for page in pages:

        page_number = page.get(
            "page",
            1
        )

        page_text = (
            page.get("text") or ""
        ).strip()

        if not page_text:
            continue

        page_chunks = create_chunks(
            page_text
        )

        for chunk in page_chunks:

            result.append({
                "text": chunk,
                "page": page_number,
            })

    return result


# =========================================================
# EMBEDDINGS
# =========================================================

def generate_embeddings(
    chunks: List[str]
) -> List[List[float]]:

    if not chunks:
        return []

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True,
    )

    return embeddings.tolist()


# =========================================================
# STORE DOCUMENT
# =========================================================

def store_embeddings(
    chunks,
    embeddings,
    filename,
    document_id=None,
):

    if not chunks or not embeddings:
        return 0

    points = []

    # If no document ID is supplied by main.py,
    # create one automatically.
    if not document_id:
        document_id = str(uuid.uuid4())

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        points.append(
            models.PointStruct(

                id=str(uuid.uuid4()),

                vector=embedding,

                payload={
                    "text": chunk["text"],

                    "filename": filename,

                    "document_id": document_id,

                    "page": chunk["page"],
                },
            )
        )

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


# =========================================================
# SEARCH DOCUMENTS
# =========================================================

def search_documents(
    query,
    limit=5,
    filenames=None,
    document_ids=None,
):

    if not query or not query.strip():
        return []

    # -----------------------------------------------------
    # Convert question into embedding
    # -----------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True,
    ).tolist()


    # -----------------------------------------------------
    # Retrieve more candidates
    # -----------------------------------------------------

    candidate_count = max(
        limit * 10,
        20
    )


    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,

        query=query_embedding,

        limit=candidate_count,

        with_payload=True,
    ).points


    # -----------------------------------------------------
    # Allowed filenames
    # -----------------------------------------------------

    allowed_names = {
        name.strip()
        for name in (filenames or [])
        if name and name.strip()
    }


    # -----------------------------------------------------
    # Allowed document IDs
    # -----------------------------------------------------

    allowed_ids = {
        value.strip()
        for value in (document_ids or [])
        if value and value.strip()
    }


    filtered = []


    # =====================================================
    # FILTER RESULTS
    # =====================================================

    for result in results:

        payload = result.payload or {}


        filename = payload.get(
            "filename",
            ""
        )


        document_id = payload.get(
            "document_id",
            ""
        )


        # -------------------------------------------------
        # Document ID is the strongest filter.
        # -------------------------------------------------

        if allowed_ids:

            if document_id not in allowed_ids:
                continue


        # -------------------------------------------------
        # If no document ID was supplied,
        # fall back to filename.
        # -------------------------------------------------

        elif allowed_names:

            if filename not in allowed_names:
                continue


        filtered.append({

            "score": result.score,

            "text": payload.get(
                "text",
                ""
            ),

            "filename": filename,

            "document_id": document_id,

            "page": payload.get(
                "page",
                1
            ),
        })


        if len(filtered) >= limit:
            break


    return filtered


# =========================================================
# GENERATE ANSWER
# =========================================================

def generate_answer(
    query,
    limit=5,
    filenames=None,
    document_ids=None,
):

    # -----------------------------------------------------
    # Retrieve relevant document chunks
    # -----------------------------------------------------

    results = search_documents(

        query,

        limit=limit,

        filenames=filenames,

        document_ids=document_ids,
    )


    # -----------------------------------------------------
    # Nothing found
    # -----------------------------------------------------

    if not results:

        return {

            "answer": (
                "The information is not available "
                "in the uploaded document."
            ),

            "sources": [],
        }


    # =====================================================
    # BUILD CONTEXT
    # =====================================================

    context_parts = []


    for index, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(

            f"[Source {index}]\n"

            f"Document: "
            f"{result['filename']}\n"

            f"Page: "
            f"{result['page']}\n"

            f"Content:\n"
            f"{result['text']}"
        )


    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # MED-RAG PROMPT
    # =====================================================

    prompt = f"""
You are MED-RAG, a medical document
question-answering assistant.

Your job is to answer the user's question
ONLY using the uploaded document context.

IMPORTANT RULES:

1. Never invent information.

2. Never use outside medical knowledge
to fill missing details.

3. If the answer is not clearly present
in the document, say:

"The information is not available
in the uploaded document."

4. Read OCR text carefully.

5. Preserve numbers exactly as written.

6. Do not silently correct OCR text.

7. Do not diagnose the patient.

8. Do not prescribe medicines.

9. Give a short and simple answer.

10. Make the answer easy for a normal
person to understand.

11. If a value is present in the document,
quote that value clearly.

12. Do not mention unrelated documents.

13. Use ONLY the document context provided.

14. Pay special attention to patient
information such as:

- Age
- Gender
- Symptoms
- Diagnosis
- Date
- Vital signs
- Blood pressure
- Temperature
- Heart rate
- Oxygen saturation
- Medications
- Allergies
- Lab values

15. OCR MAY contain symbols such as:

>
<
≥
≤

Do NOT remove or change these symbols.

16. If the document says:

Age > 45 years

answer:

Age: > 45 years.

17. If the document says:

Age: 45 years

answer:

Age: 45 years.

18. Do NOT convert:

Age > 45 years

into:

Age 45 years.

19. Do not guess an exact value
from a greater-than or less-than statement.

20. When answering a question about a
specific patient value, prefer the exact
source text containing that value.

21. Mention the page when useful.

DOCUMENT CONTEXT:

{context}


USER QUESTION:

{query}
"""


    # =====================================================
    # CALL GEMINI
    # =====================================================

    try:

        response = gemini_client.models.generate_content(

            model="gemini-3.6-flash",

            contents=prompt,
        )


    except Exception as e:

        error_text = str(e)


        # -------------------------------------------------
        # Gemini quota exceeded
        # -------------------------------------------------

        if (
            "429" in error_text
            or "RESOURCE_EXHAUSTED" in error_text
        ):

            return {

                "answer": (
                    "The document was processed successfully, "
                    "but the AI response limit has been reached. "
                    "Please try again later."
                ),

                "sources": [

                    {
                        "filename": result["filename"],

                        "document_id": result.get(
                            "document_id",
                            ""
                        ),

                        "page": result["page"],

                        "score": result["score"],

                        "text": result["text"],
                    }

                    for result in results
                ],

                "quota_error": True,
            }


        # -------------------------------------------------
        # Other errors
        # -------------------------------------------------

        return {

            "answer": (
                "The document was processed, "
                "but I could not generate the answer "
                "right now."
            ),

            "sources": [

                {
                    "filename": result["filename"],

                    "document_id": result.get(
                        "document_id",
                        ""
                    ),

                    "page": result["page"],

                    "score": result["score"],

                    "text": result["text"],
                }

                for result in results
            ],

            "error": error_text,
        }


    # =====================================================
    # READ GEMINI RESPONSE
    # =====================================================

    answer = (
        response.text
        or "No answer was generated."
    )


    # =====================================================
    # RETURN ANSWER + SOURCES
    # =====================================================

    return {

        "answer": answer,

        "sources": [

            {

                "filename": result[
                    "filename"
                ],

                "document_id": result.get(
                    "document_id",
                    ""
                ),

                "page": result[
                    "page"
                ],

                "score": result[
                    "score"
                ],

                "text": result[
                    "text"
                ],
            }

            for result in results
        ],
    }