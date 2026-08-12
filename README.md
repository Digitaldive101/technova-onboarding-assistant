# AI Employee Onboarding Assistant

A local, privacy-conscious Retrieval-Augmented Generation (RAG) application that helps employees find answers in approved company policy documents. It uses FastAPI, Streamlit, LangChain, ChromaDB, sentence-transformer embeddings, and a local Ollama model.

The assistant is intentionally **not** a generic PDF chatbot: the knowledge base is fixed and managed only by the backend. Employees cannot upload documents.

## Features

- Grounded answers from company PDFs only
- Exact fallback when retrieved documents do not contain the answer
- Local Ollama inference with `qwen3.6` at temperature `0`
- Persistent ChromaDB vector database with `all-MiniLM-L6-v2` embeddings
- Source citations showing document name and page number
- FastAPI API with typed request and response models
- Modern Streamlit chat dashboard with suggested questions, timestamps, history, and clear chat
- Error handling and logging for missing PDFs, corrupt files, missing indexes, unavailable ChromaDB, and offline Ollama

## Architecture

```text
                    Fixed company PDFs (data/)
                               |
                               v
                    backend/ingest.py
             PyPDFLoader + RecursiveCharacterTextSplitter
                       chunks: 1000 / overlap: 200
                               |
                               v
       all-MiniLM-L6-v2 embeddings -> Persistent ChromaDB (chroma_db/)
                               |
                               v
Employee -> Streamlit UI -> FastAPI POST /chat -> Top 5 relevant chunks
                                                    |
                                                    v
                                  Strict RAG prompt -> Ollama qwen3.6
                                                    |
                                                    v
                              Answer + document/page citations -> Streamlit
```

## Project structure

```text
employee_onboarding_ai/
├── backend/
│   ├── app.py                 # FastAPI app and health endpoint
│   ├── chat.py                # Local Ollama client
│   ├── config.py              # Typed environment settings
│   ├── embeddings.py          # Sentence-transformer factory
│   ├── ingest.py              # Fixed-PDF ingestion command
│   ├── pdf_loader.py          # Safe company-PDF loading
│   ├── prompt.py              # Grounded-only RAG prompt
│   ├── rag.py                 # Retrieval and generation orchestration
│   ├── routes.py              # API schemas and endpoints
│   ├── utils.py               # Knowledge-base status helpers
│   └── vector_store.py        # Persistent ChromaDB access
├── frontend/
│   └── streamlit_app.py       # Employee-facing dashboard
├── data/                      # Backend-managed policy PDFs (not committed)
├── chroma_db/                 # Generated local vector index (not committed)
├── .env.example
├── .gitignore
├── requirements.txt
└── README.md
```

## Requirements

- Python 3.12
- [Ollama](https://ollama.com/)
- The local `qwen3.6` Ollama model
- Company policy PDFs placed by an administrator in `data/`

## Installation

1. Clone the repository and enter the project directory.

   ```powershell
   cd employee_onboarding_ai
   ```

2. Create and activate a virtual environment.

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

3. Install Python dependencies.

   ```powershell
   pip install -r requirements.txt
   ```

4. Create the local environment file.

   ```powershell
   Copy-Item .env.example .env
   ```

## Install and prepare Ollama

1. Install Ollama from [ollama.com/download](https://ollama.com/download).
2. Start the Ollama service if it is not already running.
3. Pull the required model.

   ```powershell
   ollama pull qwen3.6
   ```

4. Optionally verify it.

   ```powershell
   ollama list
   ```

> If your local Ollama registry uses a different available Qwen tag, update `OLLAMA_MODEL` in `.env` to that tag before starting the application.

## Add fixed company documents

Create the `data` directory if needed, then place only administrator-approved PDFs inside it. Typical files are:

```text
data/
├── Employee_Handbook.pdf
├── IT_Policy.pdf
├── Code_of_Conduct.pdf
├── Payroll_and_Benefits.pdf
├── Travel_Policy.pdf
├── Health_Insurance.pdf
├── VPN_Guide.pdf
└── Work_From_Home.pdf
```

The Streamlit interface intentionally has no upload control. PDFs are ignored by Git to avoid committing company-confidential material.

## Company logo

To show your logo in the Streamlit sidebar, place a PNG file at:

```text
assets/company_logo.png
```

Restart Streamlit after adding or changing the file. The logo is kept local and
is excluded from Git by default.

To use a separate design in the white top header, add:

```text
assets/header_logo.png
```

## Build the vector database

Run this whenever an administrator updates the PDFs:

```powershell
python backend/ingest.py
```

The command reads every PDF in `data/`, skips and logs unreadable files, splits text into 1000-character chunks with 200-character overlap, embeds it, and replaces the local collection in `chroma_db/`.

## Run the application

Start FastAPI in one terminal:

```powershell
uvicorn backend.app:app --host 0.0.0.0 --port 8000 --reload
```

The health endpoint is available at `http://localhost:8000/` and returns:

```json
{"status":"running"}
```

Start Streamlit in a second terminal:

```powershell
streamlit run frontend/streamlit_app.py
```

Open the URL printed by Streamlit, normally `http://localhost:8501`.

If FastAPI runs elsewhere, set the frontend endpoint before starting Streamlit:

```powershell
$env:API_BASE_URL = "http://your-host:8000"
streamlit run frontend/streamlit_app.py
```

## API

### `POST /chat`

Request:

```json
{"question":"How do I connect to the VPN?"}
```

An optional, bounded `history` list can be sent for follow-up questions. It is
used only to resolve conversational references; it is never treated as an
information source.

Response:

```json
{
  "answer": "Follow the VPN setup steps provided by IT.",
  "sources": [
    {"document": "VPN_Guide.pdf", "page": 2}
  ]
}
```

If the index does not contain sufficient relevant material, the answer is exactly:

```text
I couldn't find that information in the company documents.
```

## Grounding and safety behavior

For each question, the backend embeds the question, retrieves up to five relevant chunks, and applies a relevance threshold. When no relevant chunk exists, it returns the fallback without asking the LLM. When context exists, the system prompt forbids the model from using prior knowledge, guessing, or inventing facts. Every supported answer returns citations from its retrieved pages.

The chat history is kept only in the browser session for a convenient conversation experience; company documents remain backend-managed.

### `GET /knowledge-base/status`

Returns fixed-document and local-index readiness metadata used by the dashboard,
without exposing document content.

## Configuration

Copy `.env.example` to `.env` and adjust values as needed. Important settings include:

| Variable | Default | Purpose |
| --- | --- | --- |
| `OLLAMA_MODEL` | `qwen3.6` | Local Ollama model tag |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Local embedding model |
| `RETRIEVAL_K` | `5` | Maximum retrieved chunks |
| `SIMILARITY_SCORE_THRESHOLD` | `0.10` | Minimum relevance score |
| `CHROMA_DIRECTORY` | `chroma_db` | Persistent vector store location |

## Screenshots

Add dashboard screenshots here before publishing the project.

```text
[Screenshot placeholder: Streamlit onboarding chat dashboard]
[Screenshot placeholder: Answer with document/page citations]
```

## Future improvements

- Add administrator-authenticated index status and re-index controls
- Add automated unit, integration, and retrieval-quality tests
- Add hybrid search and reranking for larger policy collections
- Add containerization and internal deployment configuration
- Add document versioning and audit metadata
