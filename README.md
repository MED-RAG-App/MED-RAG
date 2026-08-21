# MED-RAG

## Medical Document Question Answering using Retrieval-Augmented Generation

MED-RAG is a medical document question-answering system that uses
Retrieval-Augmented Generation (RAG) to answer questions from uploaded
medical documents.

The system processes PDF documents, extracts their text, divides the text
into smaller chunks, converts the chunks into numerical embeddings, and
stores them in Qdrant Cloud for semantic search.

When a user asks a question, the system retrieves the most relevant
document chunks and provides them as context to Google Gemini. Gemini then
generates an answer using the retrieved document content.

---

# Backend

The backend is responsible for:

- PDF upload and text extraction
- Text chunking
- Embedding generation
- Vector storage using Qdrant Cloud
- Semantic document retrieval
- Answer generation using Google Gemini
- REST API development using FastAPI
- Returning answers and source information to the frontend

---

# Backend Architecture

The MED-RAG backend follows a Retrieval-Augmented Generation (RAG)
architecture.

```text
                    FRONTEND
                       |
                       | REST API
                       v
                +---------------+
                |    FastAPI    |
                |   Backend     |
                +-------+-------+
                        |
          +-------------+-------------+
          |                           |
          | PDF Upload                | User Question
          v                           v
   +-------------+            +------------------+
   |    PyPDF    |            | Sentence         |
   | Text Extract|            | Transformer      |
   +------+------+            | MiniLM-L6-v2     |
          |                   +--------+---------+
          v                            |
   +-------------+                     |
   | Text        |                     |
   | Chunking    |                     |
   +------+------+                     |
          |                             |
          v                             v
   +-------------+              Query Embedding
   | Embedding   |                     |
   | Generation  |                     |
   +------+------+                     |
          |                             |
          +-------------+---------------+
                        |
                        v
                +---------------+
                | Qdrant Cloud  |
                | Vector DB     |
                +-------+-------+
                        |
                  Similarity Search
                        |
                        v
                Relevant Chunks
                        |
                        v
                +---------------+
                | Google Gemini |
                | 3.6 Flash     |
                +-------+-------+
                        |
                        v
                Grounded Answer
                        |
                        v
                    FRONTEND
RAG Pipeline
The backend works in two main stages.
1. Document Ingestion
When a PDF is uploaded:
PDF
 ↓
PyPDF Text Extraction
 ↓
Text Chunking
 ↓
Embedding Generation
 ↓
Qdrant Storage
