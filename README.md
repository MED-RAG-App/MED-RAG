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
Process
FastAPI receives the uploaded PDF.
PyPDF extracts text from each page.
The extracted text is divided into overlapping chunks.
all-MiniLM-L6-v2 converts each chunk into a 384-dimensional vector.
The text chunks and their vectors are stored in Qdrant Cloud.
2. Question Answering
When the user asks a question:
User Question
      ↓
Question Embedding
      ↓
Qdrant Similarity Search
      ↓
Relevant Document Chunks
      ↓
Gemini
      ↓
Generated Answer
      ↓
Sources
Process
FastAPI receives the user's question.
The question is converted into an embedding using
all-MiniLM-L6-v2.
Qdrant performs a similarity search against stored document embeddings.
The most relevant document chunks are retrieved.
The retrieved chunks are provided to Google Gemini as context.
Gemini generates an answer based only on the retrieved document context.
The backend returns the answer and source information to the frontend.
Technologies Used
Backend
Technology	Purpose
Python 3.11	Backend programming language
FastAPI	REST API framework
Uvicorn	ASGI server
Pydantic	Request validation
Document Processing
Technology	Purpose
PyPDF	Extract text from PDF documents
RAG and Embeddings
Technology	Purpose
Sentence Transformers	Generate text embeddings
all-MiniLM-L6-v2	Embedding model
Qdrant Cloud	Vector database and similarity search
Generative AI
Technology	Purpose
Google Gemini API	Answer generation
Gemini 3.6 Flash	Generative model
google-genai	Gemini Python SDK
Development
Technology	Purpose
python-dotenv	Load environment variables
Git	Version control
GitHub	Code collaboration
Python virtual environment	Dependency isolation
Project Structure
MED-RAG/
│
├── backend/
│   ├── main.py
│   ├── rag_pipeline.py
│   ├── pdf_processor.py
│   ├── .env
│   └── venv/
│
├── .gitignore
├── README.md
└── .git/
Important Files
backend/main.py
Contains the FastAPI application and API endpoints.
Responsible for:
Receiving PDF uploads
Extracting PDF text
Creating chunks
Generating embeddings
Storing data in Qdrant
Receiving user questions
Returning generated answers
backend/rag_pipeline.py
Contains the main RAG logic.
Responsible for:
Text chunking
Embedding generation
Qdrant storage
Similarity search
Gemini answer generation
backend/pdf_processor.py
Contains PDF-related processing functionality used by the backend.
API Endpoints
The backend exposes REST APIs through FastAPI.
1. Root Endpoint
GET /
Used to verify that the backend is running.
Example response:
{
  "message": "MED-RAG backend is running",
  "status": "success"
}
2. Health Check
GET /health
Used to check whether the backend application is healthy.
Example response:
{
  "status": "healthy"
}
3. Upload PDF
POST /upload-pdf
Uploads and processes a medical PDF.
Processing
PDF
 ↓
Text Extraction
 ↓
Chunking
 ↓
Embedding Generation
 ↓
Qdrant Storage
Example Response
{
  "filename": "Sample-Smart-Report-Clinics.pdf",
  "pages": 5,
  "characters": 12000,
  "chunks": 15,
  "embeddings": 15,
  "stored_in_qdrant": 15,
  "message": "PDF processed and stored successfully."
}
The actual numbers depend on the uploaded document.
4. Ask Question
POST /ask
Used by the frontend to send a question to the RAG system.
Request
{
  "question": "What information is mentioned in the report?"
}
Response Structure
{
  "answer": "Answer generated from the uploaded document.",
  "sources": [
    {
      "filename": "Sample-Smart-Report-Clinics.pdf",
      "score": 0.3974,
      "text": "Relevant document content..."
    }
  ]
}
Frontend Integration
The frontend communicates with the backend using HTTP REST API requests.
Base URL
When running locally:
http://127.0.0.1:8000
API Summary
Action	Method	Endpoint
Check backend	GET	/
Health check	GET	/health
Upload PDF	POST	/upload-pdf
Ask question	POST	/ask
Frontend Workflow
User
 |
 | Upload PDF
 v
Frontend
 |
 | POST /upload-pdf
 v
FastAPI Backend
 |
 v
Qdrant
 |
 | Document stored
 v
Frontend
 |
 | POST /ask
 v
FastAPI Backend
 |
 v
Qdrant Retrieval
 |
 v
Gemini
 |
 v
Answer + Sources
 |
 v
Frontend
 |
 v
User
Testing the Backend
FastAPI provides an interactive Swagger UI for testing the API.
Start the backend:
python -m uvicorn main:app
The backend runs at:
http://127.0.0.1:8000
Open:
http://127.0.0.1:8000/docs
Test PDF Upload
Open /docs.
Find POST /upload-pdf.
Click Try it out.
Select a PDF file.
Click Execute.
Check the response.
Verify that chunks and embeddings were generated.
Verify that vectors were stored in Qdrant.
Test Question Answering
Open /docs.
Find POST /ask.
Click Try it out.
Enter:
{
  "question": "What information is mentioned in the report?"
}
Click Execute.
Check the generated answer.
Check the returned source information.
Backend Setup
1. Clone the Repository
git clone https://github.com/MED-RAG-App/MED-RAG.git
cd MED-RAG/backend
2. Create Virtual Environment
python3 -m venv venv
Activate it:
source venv/bin/activate
3. Install Dependencies
pip install fastapi uvicorn pypdf python-dotenv sentence-transformers qdrant-client google-genai
4. Configure Environment Variables
Create:
backend/.env
Add:
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
GEMINI_API_KEY=your_gemini_api_key
Security
API keys must never be committed to GitHub.
The .env file is excluded using .gitignore.
Problems Faced and Solutions
During backend development, several configuration and integration issues
were encountered.
1. Gemini Model Availability
The initial Gemini model used for testing was no longer available to new
users.
Solution
The Gemini model configuration was updated to:
model="gemini-3.6-flash"
2. Python Environment Mismatch
Initially, uvicorn was being executed from a different Python installation
than the project's virtual environment.
This caused errors such as:
ModuleNotFoundError: No module named 'fastapi'
Solution
The backend was run using the virtual environment's Python:
python -m uvicorn main:app
This ensures that the packages installed inside venv are used.
3. Missing Python Dependencies
Some packages were missing from the virtual environment, including:
FastAPI
PyPDF
Uvicorn
Google GenAI SDK
Solution
The required dependencies were installed inside the virtual environment.
4. Port 8000 Already in Use
The backend initially produced:
ERROR: [Errno 48] error while attempting to bind on address
('127.0.0.1', 8000): address already in use
Cause
Another process was already using port 8000.
Solution
The existing server process was identified/stopped before starting the
backend again.
5. Gemini API Warning
The Google GenAI SDK displayed a warning related to Automatic Function
Calling (AFC).
The API request still completed successfully.
This warning does not prevent the current MED-RAG generation workflow from
working.
Current Backend Status
The following backend components have been implemented and tested:
 FastAPI application
 PDF upload API
 PDF text extraction
 Text chunking
 Embedding generation
 Qdrant Cloud vector storage
 Semantic document retrieval
 Gemini integration
 Question-answering API
 Source information returned with answers
 Swagger API testing
 GitHub version control
Current Limitations
The current implementation has some limitations:
PDF text extraction currently depends on extractable PDF text.
Scanned/image-only PDFs require OCR support.
Retrieval quality depends on the quality of extracted text and embeddings.
The current implementation uses a fixed chunk size and overlap.
Multiple uploaded documents are stored in the same Qdrant collection.
Source citations currently return retrieved chunks rather than exact page
numbers.
Multilingual support and OCR enhancements are planned for future versions.
Future Improvements
Planned improvements include:
OCR support for scanned medical documents
Multilingual queries
Support for English, Hindi, and Kannada
Improved document chunking
Page-level source citations
Better retrieval and ranking
Improved frontend integration
Authentication and access control
Production deployment
Security Notes
The project uses external services such as Qdrant Cloud and Google Gemini.
The following information must remain private:
QDRANT_API_KEY
GEMINI_API_KEY
These values should only be stored in the local .env file or secure
deployment environment.
Never place API keys directly inside Python source code or commit them to
GitHub.
Developer
Backend Development:
Nisha
Responsibilities:
Backend API development
RAG pipeline implementation
PDF processing
Embedding generation
Qdrant integration
Gemini integration
Frontend API integration support