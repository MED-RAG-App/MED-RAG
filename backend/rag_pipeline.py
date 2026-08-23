from typing import List
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# ============================================================
# Load environment variables
# ============================================================

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


# ============================================================
# Connect to Qdrant
# ============================================================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)


# ============================================================
# Connect to Gemini
# ============================================================

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)


# ============================================================
# Configuration
# ============================================================

COLLECTION_NAME = "medical_documents"


# ============================================================
# Create Qdrant payload index
# ============================================================

try:
    qdrant_client.create_payload_index(
        collection_name=COLLECTION_NAME,
        field_name="filename",
        field_schema=models.PayloadSchemaType.KEYWORD,
    )
except Exception:
    # Index may already exist
    pass


# ============================================================
# Load embedding model
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# Create text chunks
# ============================================================

def create_chunks(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 200
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

        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += step

    return chunks


# ============================================================
# Generate embeddings
# ============================================================

def generate_embeddings(
    chunks: List[str]
) -> List[List[float]]:

    if not chunks:
        return []

    embeddings = embedding_model.encode(
        chunks,
        convert_to_numpy=True
    )

    return embeddings.tolist()


# ============================================================
# Store embeddings in Qdrant
# ============================================================

def store_embeddings(
    chunks: List[str],
    embeddings: List[List[float]],
    filename: str
):

    if not chunks or not embeddings:
        return 0

    # --------------------------------------------------------
    # Delete existing chunks belonging to this PDF
    # --------------------------------------------------------

    qdrant_client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="filename",
                        match=models.MatchValue(
                            value=filename
                        ),
                    )
                ]
            )
        ),
        wait=True,
    )

    # --------------------------------------------------------
    # Create Qdrant points
    # --------------------------------------------------------

    points = []

    for chunk, embedding in zip(
        chunks,
        embeddings
    ):

        points.append(
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload={
                    "text": chunk,
                    "filename": filename,
                },
            )
        )

    # --------------------------------------------------------
    # Upload points
    # --------------------------------------------------------

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
        wait=True,
    )

    return len(points)


# ============================================================
# Search documents
# ============================================================

def search_documents(
    query: str,
    limit: int = 10
):

    if not query or not query.strip():
        return []

    query_lower = query.lower()

    # --------------------------------------------------------
    # Convert question into embedding
    # --------------------------------------------------------

    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    # --------------------------------------------------------
    # Retrieve more candidates than requested
    # --------------------------------------------------------

    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=20,
        with_payload=True,
    ).points

    # --------------------------------------------------------
    # Field-aware keyword matching
    # --------------------------------------------------------

    keywords = []

    if "age" in query_lower:

        keywords = [
            "age/gender"
        ]

    elif "mcv" in query_lower:

        keywords = [
            "mcv"
        ]

    elif "mch" in query_lower:

        keywords = [
            "mch"
        ]

    elif (
        "hemoglobin" in query_lower
        or "haemoglobin" in query_lower
    ):

        keywords = [
            "hemoglobin",
            "haemoglobin"
        ]

    elif "rbc" in query_lower:

        keywords = [
            "rbc count"
        ]

    elif (
        "blood pressure" in query_lower
        or query_lower.strip() == "bp"
    ):

        keywords = [
            "systolicbp",
            "diastolicbp"
        ]

    # --------------------------------------------------------
    # Move keyword-matching chunks to the front
    # --------------------------------------------------------

    if keywords:

        matching = []
        others = []

        for result in results:

            text = result.payload.get(
                "text",
                ""
            ).lower()

            if any(
                keyword in text
                for keyword in keywords
            ):
                matching.append(result)

            else:
                others.append(result)

        results = matching + others

    # --------------------------------------------------------
    # Convert Qdrant results to dictionaries
    # --------------------------------------------------------

    return [
        {
            "score": result.score,
            "text": result.payload.get(
                "text",
                ""
            ),
            "filename": result.payload.get(
                "filename",
                ""
            ),
        }
        for result in results[:limit]
    ]


# ============================================================
# Generate answer using Gemini
# ============================================================

def generate_answer(
    query: str,
    limit: int = 10
):

    # --------------------------------------------------------
    # Retrieve relevant document chunks
    # --------------------------------------------------------

    results = search_documents(
        query,
        limit
    )

    # --------------------------------------------------------
    # No relevant information found
    # --------------------------------------------------------

    if not results:

        return {
            "answer": (
                "The information is not available "
                "in the uploaded document."
            ),
            "sources": []
        }

    # --------------------------------------------------------
    # Build document context
    # --------------------------------------------------------

    context_parts = []

    for index, result in enumerate(
        results,
        start=1
    ):

        context_parts.append(
            f"[Source {index}]\n"
            f"Document: {result['filename']}\n"
            f"Content:\n{result['text']}"
        )

    context = "\n\n".join(
        context_parts
    )

    # --------------------------------------------------------
    # Gemini prompt
    # --------------------------------------------------------

    prompt = f"""
You are MED-RAG, a medical document question-answering assistant.

Answer the user's question using ONLY the information provided
in the document context.

STRICT RULES:

1. Do not invent information.
2. Do not assume information that is not explicitly present.
3. Do not use outside medical knowledge.
4. If the answer cannot be found in the provided context,
   reply exactly:

"The information is not available in the uploaded document."

5. Do not diagnose the patient.
6. Do not recommend or prescribe medicines.
7. Give only the answer to the user's question.
8. Keep the answer very short: maximum 1-2 sentences.
9. For simple factual questions, provide the value and unit.
10. Do not repeat the question.
11. Do not summarize the document.
12. Do not mention the source document unless the user
    specifically asks for the source.
13. Do not add unnecessary explanations.
14. Do not use headings such as "Answer:".

Examples:

Question: What is the patient's hemoglobin level?
Answer: Hemoglobin: 12 g/dL

Question: What is the patient's RBC count?
Answer: RBC count: 4.07 Million/cu.mm

Question: What is the patient's MCV?
Answer: MCV: 88.4 fL

Document Context:
{context}

User Question:
{query}
"""

    # --------------------------------------------------------
    # Generate answer
    # --------------------------------------------------------

    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    answer = (
        response.text
        if response.text
        else "No answer was generated."
    )

    # --------------------------------------------------------
    # Prepare sources
    # --------------------------------------------------------

    sources = []

    seen_filenames = set()

    for result in results:

        filename = result["filename"]

        if filename not in seen_filenames:

            seen_filenames.add(filename)

            sources.append(
                {
                    "filename": filename,
                    "text": result["text"],
                }
            )

    # --------------------------------------------------------
    # Return final response
    # --------------------------------------------------------

    return {
        "answer": answer,
        "sources": sources
    }