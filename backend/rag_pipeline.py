from typing import List
import os
import uuid
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient, models


# ==============================
# Load environment variables
# ==============================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Qdrant environment variables are missing.")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing.")


# ==============================
# Connect to services
# ==============================

qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
    timeout=60,
)

gemini_client = genai.Client(
    api_key=GEMINI_API_KEY
)

COLLECTION_NAME = "medical_documents"


# ==============================
# Load embedding model
# ==============================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ==============================
# Create text chunks
# ==============================

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


# ==============================
# Generate embeddings
# ==============================

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


# ==============================
# Store embeddings in Qdrant
# ==============================

def store_embeddings(
    chunks: List[str],
    embeddings: List[List[float]],
    filename: str
):

    if not chunks or not embeddings:
        return 0

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

    qdrant_client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    return len(points)


# ==============================
# Search documents
# ==============================

def search_documents(
    query: str,
    limit: int = 5
):

    if not query or not query.strip():
        return []

    # Convert query into embedding
    query_embedding = embedding_model.encode(
        query,
        convert_to_numpy=True
    ).tolist()

    # Search Qdrant
    results = qdrant_client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=limit,
        with_payload=True,
    ).points

    return [
        {
            "score": result.score,
            "text": result.payload.get("text", ""),
            "filename": result.payload.get("filename", ""),
        }
        for result in results
    ]


# ==============================
# Generate answer using Gemini
# ==============================

def generate_answer(
    query: str,
    limit: int = 5
):

    # Retrieve relevant chunks
    results = search_documents(
        query,
        limit
    )

    if not results:
        return {
            "answer": (
                "I could not find relevant information "
                "in the uploaded documents."
            ),
            "sources": []
        }

    # Build context
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

    context = "\n\n".join(context_parts)

    # Prompt Gemini
    prompt = f"""
You are MED-RAG, a medical document
question-answering assistant.

Answer the user's question using ONLY
the information provided in the document
context below.

Rules:

1. Do not invent information.
2. Do not use outside medical knowledge.
3. If the answer cannot be found in the
   context, say:
   "The information is not available
   in the uploaded document."
4. Do not diagnose the patient.
5. Do not recommend or prescribe medicines.
6. Mention the source document when possible.
7. Keep the answer clear and concise.

Document Context:
{context}

User Question:
{query}
"""

    # Generate answer
    response = gemini_client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt,
    )

    answer = response.text or "No answer was generated."

    return {
        "answer": answer,
        "sources": [
            {
                "filename": result["filename"],
                "score": result["score"],
                "text": result["text"],
            }
            for result in results
        ],
    }