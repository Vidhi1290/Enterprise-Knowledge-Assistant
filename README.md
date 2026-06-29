# 🧠 Enterprise Knowledge Assistant

> A production-oriented Retrieval Augmented Generation (RAG) system designed to help employees query internal knowledge bases using natural language. The solution runs locally with Ollama, requires no API keys, and is built to be accurate, grounded, and practical for enterprise use.

**Assignment:** AI Engineer — Enterprise Knowledge Assistant  
**Built by:** Vidhi Waghela

---

## 🎯 Project Overview

Enterprises often store critical knowledge across HR policies, technical guides, compliance documents, customer FAQs, and product documentation. Searching these files manually is slow and inefficient. This project addresses that problem by building an AI-powered assistant that can:

- ingest internal documents,
- extract and index semantic knowledge,
- retrieve the most relevant context,
- generate grounded answers,
- and cite the source documents used for each response.

The system is designed as a practical RAG application with strong engineering discipline: local-first deployment, retrieval quality control, prompt grounding, source attribution, and a simple user interface.

---

## 🧩 Business Problem

Employees regularly ask questions such as:

- “What is the leave policy?”
- “How many sick leaves are allowed?”
- “What is the refund procedure?”
- “What does the SLA guarantee?”

Manually searching through large internal repositories is time-consuming and error-prone. The Enterprise Knowledge Assistant provides a faster and more reliable method for retrieving information from organization documents.

---

## ✅ Functional Objectives

This solution delivers the core capabilities required for the assignment:

1. Document ingestion and preprocessing
2. Knowledge extraction and indexing
3. Semantic retrieval
4. Context-aware answer generation
5. Source citation and traceability
6. User interaction through a web app, API, and CLI

---

## 🏗️ System Architecture

```text
┌──────────────────────────────────────────────────────────────────────┐
│                     ENTERPRISE KNOWLEDGE ASSISTANT                 │
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌───────────────┐   ┌───────────────┐   ┌──────────────────────┐   │
│  │  Streamlit UI │   │  FastAPI API  │   │        CLI           │   │
│  │  Chat / Upload│   │  /ask /ingest │   │ ingest / ask / eval │   │
│  └──────┬────────┘   └──────┬────────┘   └──────────┬───────────┘   │
│         └──────────────────┴─────────────────────────────┘          │
│                                │                                       │
│                     ┌──────────▼──────────┐                            │
│                     │   RAG Pipeline      │                            │
│                     │  retrieve + generate │                            │
│                     └──────────┬──────────┘                            │
│                                │                                       │
│        ┌───────────────────────┼───────────────────────┐             │
│        │                       │                       │             │
│  ┌─────▼──────┐        ┌──────▼────────┐       ┌──────▼────────┐     │
│  │ Hybrid Search│       │  FAISS Store   │       │ Ollama LLM    │     │
│  │ Semantic + BM25│     │  Vector Index  │       │ llama3.2:3b   │     │
│  └─────────────┘        └────────────────┘       └──────────────┘     │
│                                                                      │
│  ┌────────────────────────────────────────────────────────────────┐  │
│  │ Document Ingestion Pipeline: Files → Extract → Chunk → Embed │  │
│  │ Supported formats: PDF, TXT, MD, DOCX, CSV                    │  │
│  └────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Data Flow

### Query Time

```text
User Question
   │
   ▼
Query embedding via Ollama
   │
   ▼
Semantic retrieval from FAISS index
   │
   ▼
BM25 keyword scoring on retrieved candidates
   │
   ▼
RRF fusion to combine semantic and keyword signals
   │
   ▼
Context assembly with source metadata
   │
   ▼
Grounded answer generation with local LLM
   │
   ▼
Answer + citations + confidence score
```

### Ingestion Time

```text
Document files
   │
   ▼
Text extraction
   │
   ▼
Chunking
   │
   ▼
Embedding generation
   │
   ▼
FAISS indexing + metadata persistence
```

---

## 🛠️ Technology Stack

| Component | Technology | Why it was chosen |
|---|---|---|
| LLM | Ollama + llama3.2:3b | Fully local, privacy-preserving, fast enough for enterprise demos |
| Embeddings | Ollama + nomic-embed-text | Retrieval-optimized embedding model that runs locally |
| Vector Store | FAISS | Lightweight, dependency-light, efficient exact similarity search |
| API Layer | FastAPI | High-performance async API with automatic documentation |
| Web UI | Streamlit | Fast to build, intuitive for demos and interactive testing |
| CLI | Python | Simple command-line workflow for ingest, ask, eval, and stats |
| Document Parsing | pdfplumber / python-docx / text readers | Supports common enterprise document formats |

---

## 🧠 Design Decisions

### 1. Local-first architecture
The system is designed to run 100% locally using Ollama. This is critical for enterprise environments where internal policy and compliance documents cannot be sent to third-party APIs.

### 2. Grounded generation
The model is instructed to answer only from the provided retrieved context. This significantly reduces hallucinations and makes the answers more trustworthy.

### 3. Hybrid retrieval strategy
The system combines:

- semantic search using embeddings,
- BM25 keyword scoring,
- and Reciprocal Rank Fusion (RRF)

This improves recall for both conceptual and exact-match questions.

### 4. FAISS over ChromaDB
FAISS was chosen because it avoids the sqlite dependency issue on macOS and offers efficient, exact nearest-neighbor retrieval for this use case.

### 5. Chunking strategy
Documents are split into smaller chunks with overlap to preserve context across boundaries while keeping retrieval precise.

### 6. Confidence scoring and source citation
Each response includes a confidence score and source references so users can verify the answer and understand where it came from.

---

## ✨ Key Features

- Local document ingestion from multiple file types
- Semantic search over embedding vectors
- Hybrid retrieval using semantic + keyword signals
- Source-grounded answer generation
- Confidence scoring for retrieval quality
- Web UI through Streamlit
- REST API for integration
- CLI for batch workflows and evaluation
- Optional query rewriting for improved retrieval

---

## 🚀 Setup Instructions

### Prerequisites

- Python 3.9+
- Ollama installed and running

### Install Ollama models

```bash
ollama pull llama3.2:3b
ollama pull nomic-embed-text
```

### Create a virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Start Ollama

```bash
ollama serve
```

---

## ▶️ How to Run the Application

### Option 1: Streamlit UI (Recommended for demos)

```bash
streamlit run streamlit_app.py
```

Then open the local browser URL shown by Streamlit.

### Option 2: FastAPI server

```bash
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000
```

API docs will be available at:

```text
http://localhost:8000/docs
```

### Option 3: CLI

```bash
python cli.py ingest ./data/sample_docs
python cli.py ask "What is the leave policy?"
python cli.py interactive
python cli.py stats
python cli.py eval
```

---

## 🔌 API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

Example response:

```json
{
  "answer": "Customers are eligible for a full refund within 30 days of purchase. (Source 1)",
  "sources": [
    {
      "document": "Customer_Policy.md",
      "page": 1,
      "relevance_score": 0.847
    }
  ],
  "confidence": 0.847,
  "confidence_label": "high",
  "chunks_used": 3,
  "question": "What is the refund policy?"
}
```

### POST /ingest

```bash
curl -X POST http://localhost:8000/ingest -F "files=@HR_Policy.pdf"
```

### GET /stats

```bash
curl http://localhost:8000/stats
```

### GET /health

```bash
curl http://localhost:8000/health
```

---

## 📊 Evaluation Approach

The project includes an evaluation workflow for testing answer quality and retrieval behavior.

### Evaluation dimensions

- accuracy of retrieved information,
- relevance of generated answers,
- citation quality,
- and hallucination prevention.

### Improvements applied

- smaller retrieval context to reduce noise,
- lower temperature for more deterministic answers,
- local embedding cache for repeated queries,
- and strict grounding instructions for safer enterprise use.

---

## ⚠️ Known Limitations

1. Sequential embedding can be slower for very large document sets.
2. FAISS does not provide native deletion semantics like a full vector database.
3. OCR for scanned PDFs is not included.
4. Table-heavy PDFs may lose structural fidelity during extraction.
5. The system is optimized for English-language documents.

---

## 🔭 Future Improvements

- add re-ranking for higher retrieval precision,
- support Docker deployment,
- introduce authentication and RBAC,
- improve multi-document reasoning,
- add user feedback collection,
- and integrate a more advanced evaluation framework such as RAGAS.

---

## 📁 Project Structure

```text
enterprise-knowledge-assistant/
├── app/
│   ├── api.py
│   ├── ingestion.py
│   ├── rag.py
│   └── vector_store.py
├── data/
│   ├── sample_docs/
│   └── chroma_db/
├── tests/
├── cli.py
├── config.py
├── streamlit_app.py
├── requirements.txt
└── README.md
```

---

## 🏁 Summary

This project demonstrates a strong end-to-end RAG implementation for an enterprise knowledge assistant. It combines modern AI retrieval techniques with practical software engineering principles to create a system that is local, grounded, explainable, and suitable for real-world internal knowledge search scenarios.
