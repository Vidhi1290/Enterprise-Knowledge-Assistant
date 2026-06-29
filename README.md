# 🧠 Enterprise Knowledge Assistant

> A production-ready RAG system that runs **100% locally** using Ollama — no OpenAI/Anthropic API keys needed.

Built with: **Python · FastAPI · ChromaDB · Streamlit · Ollama (llama3.2:3b + nomic-embed-text)**

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    ENTERPRISE KNOWLEDGE ASSISTANT                │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│   ┌──────────┐   ┌──────────┐   ┌───────────────────────────┐  │
│   │Streamlit │   │FastAPI   │   │         CLI               │  │
│   │   UI     │   │REST API  │   │  (ingest/ask/interactive) │  │
│   └────┬─────┘   └────┬─────┘   └───────────┬───────────────┘  │
│        │              │                      │                  │
│        └──────────────┴──────────────────────┘                  │
│                           │                                      │
│                    ┌──────▼──────┐                               │
│                    │ RAG Pipeline │                               │
│                    └──────┬──────┘                               │
│                           │                                      │
│         ┌─────────────────┼─────────────────┐                   │
│         │                 │                 │                    │
│  ┌──────▼──────┐  ┌───────▼──────┐  ┌──────▼──────┐           │
│  │Query Rewrite│  │Hybrid Search │  │ LLM Answer  │           │
│  │(Ollama LLM) │  │Semantic+BM25 │  │ Generation  │           │
│  └─────────────┘  └───────┬──────┘  └─────────────┘           │
│                           │                                      │
│              ┌────────────┴────────────┐                        │
│              │                         │                        │
│     ┌────────▼────────┐    ┌──────────▼─────────┐             │
│     │   ChromaDB      │    │  Ollama Embeddings  │             │
│     │ (Vector Store)  │    │  (nomic-embed-text) │             │
│     └─────────────────┘    └────────────────────┘             │
│                                                                  │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                INGESTION PIPELINE                         │   │
│  │  Files → Load → Extract → Chunk → Embed → Store         │   │
│  │  PDF | TXT | MD | DOCX | CSV                            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Question
     │
     ▼
[1] Query Rewriting (LLM expands vague queries for better recall)
     │
     ▼
[2] Semantic Search (nomic-embed-text → ChromaDB cosine similarity)
     │
     ▼
[3] BM25 Keyword Scoring (on semantic candidates)
     │
     ▼
[4] RRF Fusion (70% semantic + 30% keyword)
     │
     ▼
[5] Context Assembly (top-K chunks with source metadata)
     │
     ▼
[6] LLM Generation (grounded answer — llama3.2:3b or gpt-oss:20b)
     │
     ▼
Answer + Sources + Confidence Score
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- [Ollama](https://ollama.ai) installed and running
- Required models pulled:

```bash
ollama pull llama3.2:3b         # LLM for answer generation (fast)
ollama pull nomic-embed-text    # Embeddings (required)

# Optional — for higher quality answers:
ollama pull gpt-oss:20b
ollama pull phi3
```

### Installation

```bash
# 1. Clone / navigate to project
cd enterprise-knowledge-assistant

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
ollama serve
```

### Option A: Streamlit UI (Recommended)

```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
# Upload docs via sidebar → ask questions
```

### Option B: FastAPI REST Server

```bash
# Start API server
python -m app.api
# or:
uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

# API docs at http://localhost:8000/docs
```

### Option C: CLI

```bash
# Ingest documents
python cli.py ingest ./data/sample_docs

# Ask a question
python cli.py ask "What is the employee leave policy?"

# Ask with a better model
python cli.py ask "Summarize the refund policy" --model gpt-oss:20b

# Interactive mode (conversation with memory)
python cli.py interactive

# Check stats
python cli.py stats

# Run evaluation
python cli.py eval
```

---

## 🔌 API Reference

### POST /ask

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "use_query_rewriting": true
  }'
```

Response:
```json
{
  "answer": "Customers are eligible for a full refund within 30 days of purchase. (Source: Customer_Policy.md, Page 1)",
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
curl -X POST http://localhost:8000/ingest \
  -F "files=@HR_Policy.pdf" \
  -F "files=@Technical_Guide.pdf"
```

### GET /stats

```bash
curl http://localhost:8000/stats
```

### GET /health

```bash
curl http://localhost:8000/health
```

### Streaming (SSE)

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Explain the leave policy in detail", "stream": true}'
```

### Multi-turn Conversation

```json
{
  "question": "What about paternity leave?",
  "conversation_history": [
    {"role": "user", "content": "What is the maternity leave policy?"},
    {"role": "assistant", "content": "Maternity leave is 26 weeks..."}
  ]
}
```

---

## ⚙️ Configuration

Edit `config.py` or use environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_MODEL` | `llama3.2:3b` | LLM for answer generation |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `CHUNK_SIZE` | `512` | Chars per chunk |
| `CHUNK_OVERLAP` | `50` | Overlap between chunks |
| `TOP_K` | `5` | Chunks retrieved per query |
| `MIN_SIMILARITY` | `0.3` | Min cosine similarity threshold |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint |

```bash
# Example: use gpt-oss:20b for better quality
LLM_MODEL=gpt-oss:20b python -m app.api
```

---

## 🧠 Design Decisions

### Why nomic-embed-text for Embeddings?
- Optimized for retrieval tasks (high MRL benchmark scores)
- 768-dim vectors — good balance of quality vs storage
- Runs locally via Ollama — no API cost or data privacy concerns
- Alternative: `mxbai-embed-large` for higher quality at 1024-dim

### Why Hybrid Search (Semantic + BM25)?
Pure semantic search misses exact keyword matches (e.g. "Section 4.2", "API v3", "SOC 2"). BM25 catches these. RRF fusion (70/30 semantic/keyword) combines both without needing separate keyword index.

### Chunking Strategy: Recursive Character Splitter
- Respects natural boundaries: paragraphs → sentences → words
- 512 chars with 50-char overlap preserves cross-boundary context
- Alternative: semantic chunking (split at topic change) — better quality but slower

### Why ChromaDB over FAISS?
- ChromaDB is persistent without extra setup — FAISS requires manual serialization
- Built-in metadata filtering for source citations
- Cosine similarity is better than L2 for normalized text embeddings

### Why Low Temperature (0.1)?
Factual QA requires deterministic, accurate answers. Higher temp increases creativity but also hallucination risk — unacceptable for enterprise use.

### Confidence Scoring
Based on avg cosine similarity of retrieved chunks. Thresholds:
- ≥ 0.7: high — answer well-supported
- ≥ 0.5: medium — answer likely correct
- ≥ 0.3: low — answer may be incomplete
- < 0.3: filtered out (below `MIN_SIMILARITY`)

---

## 📊 Evaluation

```bash
python cli.py eval
```

Test suite covers:
- **Easy**: Direct factual questions ("How many leaves?")
- **Medium**: Multi-step ("What's the refund process?")
- **Hard**: Cross-document reasoning ("What certifications + what's the SLA?")
- **Edge**: Questions with no answer (tests hallucination prevention)

Metrics tracked:
- Keyword hit rate (expected terms in answer)
- Source citation rate
- Average retrieval confidence
- Pass/fail per test case

Results saved to `tests/eval_results.json`.

---

## ⚠️ Known Limitations

1. **No table extraction from PDFs** — pdfplumber reads table text but loses structure. Workaround: convert tables to markdown before ingestion.
2. **Sequential embedding** — Ollama doesn't support batch embedding API; ingestion of 100+ docs is slow. Fix: run multiple workers or use sentence-transformers directly.
3. **Single-collection** — all documents in one ChromaDB collection; no per-department access control. Fix: multiple collections + auth middleware.
4. **BM25 simplified** — IDF computed over query, not corpus. True BM25 needs corpus-level term statistics.
5. **No cross-encoder re-ranker** — would improve retrieval quality significantly; deferred due to local compute constraints.

---

## 🔭 Future Improvements

- [ ] Cross-encoder re-ranking (ms-marco-MiniLM via sentence-transformers)
- [ ] PDF table extraction with Camelot/pdfplumber structured output
- [ ] Batch embedding with sentence-transformers for faster ingestion
- [ ] Per-user conversation history with SQLite
- [ ] Role-based access control (RBAC) for document collections
- [ ] Evaluation with RAGAS framework (faithfulness, context precision, answer relevancy)
- [ ] Docker Compose for one-command deployment
- [ ] Async ingestion pipeline with job queue

---

## 📁 Project Structure

```
enterprise-knowledge-assistant/
├── app/
│   ├── __init__.py
│   ├── ingestion.py    # Document processing pipeline
│   ├── rag.py          # Retrieval + generation pipeline
│   └── api.py          # FastAPI REST endpoints
├── data/
│   ├── sample_docs/    # Drop your documents here
│   └── chroma_db/      # Auto-created vector store
├── tests/
│   ├── test_questions.json  # Evaluation test suite
│   └── eval_results.json    # Auto-generated eval output
├── config.py           # All configuration in one place
├── streamlit_app.py    # Web UI
├── cli.py              # Command-line interface
├── requirements.txt
└── README.md
```

---

## 🙏 Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| LLM | Ollama (llama3.2:3b / gpt-oss:20b) | Local, no API cost |
| Embeddings | nomic-embed-text via Ollama | Local, retrieval-optimized |
| Vector Store | ChromaDB | Persistent, no server needed |
| API | FastAPI | Async, auto-docs, production-grade |
| UI | Streamlit | Fast to build, good for demos |
| PDF parsing | pdfplumber | Page-level extraction with metadata |
