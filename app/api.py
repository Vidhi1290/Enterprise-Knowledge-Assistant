"""
FastAPI REST API for Enterprise Knowledge Assistant

Endpoints:
  POST /ask          — Ask a question (with streaming support)
  POST /ingest       — Upload and ingest a document
  GET  /stats        — Collection statistics
  GET  /health       — Health check
  DELETE /reset      — Clear knowledge base (admin)
"""

import os
import sys
import json
import logging
import tempfile
import shutil
from pathlib import Path
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field

import config
from app.ingestion import DocumentIngestionPipeline
from app.rag import RAGPipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── App Setup ──────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise Knowledge Assistant API",
    description="Local RAG system using Ollama — no external API keys required",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy-load pipelines (initialized on first request)
_ingestion_pipeline: Optional[DocumentIngestionPipeline] = None
_rag_pipeline: Optional[RAGPipeline] = None


def get_ingestion() -> DocumentIngestionPipeline:
    global _ingestion_pipeline
    if _ingestion_pipeline is None:
        _ingestion_pipeline = DocumentIngestionPipeline(
            chroma_dir=str(config.CHROMA_DIR),
            collection_name=config.COLLECTION_NAME,
            embed_model=config.EMBED_MODEL,
            ollama_url=config.OLLAMA_BASE_URL,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    return _ingestion_pipeline


def get_rag() -> RAGPipeline:
    global _rag_pipeline
    if _rag_pipeline is None:
        _rag_pipeline = RAGPipeline(
            chroma_dir=str(config.CHROMA_DIR),
            collection_name=config.COLLECTION_NAME,
            embed_model=config.EMBED_MODEL,
            llm_model=config.LLM_MODEL,
            ollama_url=config.OLLAMA_BASE_URL,
            top_k=config.TOP_K,
            min_similarity=config.MIN_SIMILARITY,
        )
    return _rag_pipeline


# ── Request/Response Models ────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str = Field(..., description="Natural language question", min_length=3)
    conversation_history: Optional[List[dict]] = Field(
        default=None, description="Previous turns: [{role: user|assistant, content: str}]"
    )
    use_query_rewriting: bool = Field(default=True, description="Expand query before retrieval")
    stream: bool = Field(default=False, description="Stream tokens in real-time")


class SourceCitation(BaseModel):
    document: str
    page: int
    relevance_score: float


class QuestionResponse(BaseModel):
    answer: str
    sources: List[SourceCitation]
    confidence: float
    confidence_label: str
    chunks_used: int
    question: str


class IngestResponse(BaseModel):
    status: str
    results: List[dict]
    total_files: int
    successful: int


class StatsResponse(BaseModel):
    total_chunks: int
    collection: str
    llm_model: str
    embed_model: str
    chunk_size: int
    top_k: int


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check — verify Ollama connection."""
    import requests as req
    try:
        resp = req.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=5)
        models = [m["name"] for m in resp.json().get("models", [])]
        return {
            "status": "healthy",
            "ollama": "connected",
            "available_models": models,
            "configured_llm": config.LLM_MODEL,
            "configured_embed": config.EMBED_MODEL,
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e), "hint": "Make sure Ollama is running: `ollama serve`"},
        )


@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a question against the knowledge base.

    Returns grounded answer with source citations and confidence score.
    Set stream=true for token-by-token streaming response.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        rag = get_rag()

        if request.stream:
            result = rag.ask(
                question=request.question,
                conversation_history=request.conversation_history,
                use_query_rewriting=request.use_query_rewriting,
                stream=True,
            )

            async def token_generator():
                # Stream metadata first
                meta = {
                    "type": "meta",
                    "sources": result["sources"],
                    "confidence": result["confidence"],
                    "confidence_label": result["confidence_label"],
                    "chunks_used": result["chunks_used"],
                }
                yield f"data: {json.dumps(meta)}\n\n"

                # Stream answer tokens
                for token in result["answer_stream"]:
                    yield f"data: {json.dumps({'type': 'token', 'content': token})}\n\n"

                yield f"data: {json.dumps({'type': 'done'})}\n\n"

            return StreamingResponse(token_generator(), media_type="text/event-stream")

        result = rag.ask(
            question=request.question,
            conversation_history=request.conversation_history,
            use_query_rewriting=request.use_query_rewriting,
            stream=False,
        )
        return result

    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Cannot connect to Ollama. Make sure it's running: `ollama serve`",
        )
    except Exception as e:
        logger.error(f"Error answering question: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(files: List[UploadFile] = File(...)):
    """
    Upload and ingest documents into the knowledge base.
    Supports: PDF, TXT, MD, DOCX, CSV
    """
    pipeline = get_ingestion()
    results = []
    tmp_dir = Path(tempfile.mkdtemp())

    try:
        for upload in files:
            suffix = Path(upload.filename).suffix.lower()
            if suffix not in config.SUPPORTED_EXTENSIONS:
                results.append({
                    "status": "skipped",
                    "file": upload.filename,
                    "reason": f"Unsupported type {suffix}. Supported: {config.SUPPORTED_EXTENSIONS}",
                })
                continue

            tmp_path = tmp_dir / upload.filename
            with open(tmp_path, "wb") as f:
                content = await upload.read()
                f.write(content)

            result = pipeline.ingest_file(tmp_path)
            results.append(result)

    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    successful = len([r for r in results if r.get("status") == "success"])
    return {
        "status": "complete",
        "results": results,
        "total_files": len(files),
        "successful": successful,
    }


@app.post("/ingest/directory")
async def ingest_directory(background_tasks: BackgroundTasks, path: str = str(config.DOCS_DIR)):
    """Ingest all documents from a server-side directory (background task)."""
    if not Path(path).exists():
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")

    def run_ingestion():
        pipeline = get_ingestion()
        pipeline.ingest_directory(path)

    background_tasks.add_task(run_ingestion)
    return {"status": "started", "message": f"Ingesting documents from {path} in background"}


@app.get("/stats", response_model=StatsResponse)
async def get_stats():
    """Return knowledge base statistics."""
    try:
        rag = get_rag()
        stats = rag.get_collection_stats()
        return {
            **stats,
            "llm_model": config.LLM_MODEL,
            "embed_model": config.EMBED_MODEL,
            "chunk_size": config.CHUNK_SIZE,
            "top_k": config.TOP_K,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/reset")
async def reset_knowledge_base(confirm: bool = False):
    """⚠️ Delete all ingested documents. Requires confirm=true."""
    global _ingestion_pipeline, _rag_pipeline

    if not confirm:
        raise HTTPException(
            status_code=400,
            detail="Pass ?confirm=true to reset the knowledge base",
        )

    from app.vector_store import FAISSVectorStore
    store = FAISSVectorStore(str(config.CHROMA_DIR), config.COLLECTION_NAME)
    try:
        store.reset()
        _ingestion_pipeline = None
        _rag_pipeline = None
        return {"status": "reset", "message": "Knowledge base cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Startup: Auto-ingest sample docs ──────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Auto-ingest sample docs if they exist and DB is empty."""
    try:
        config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

        from app.vector_store import FAISSVectorStore
        store = FAISSVectorStore(str(config.CHROMA_DIR), config.COLLECTION_NAME)
        collection = client.get_or_create_collection(config.COLLECTION_NAME)

        if collection.count() == 0 and any(config.DOCS_DIR.rglob("*.*")):
            logger.info("📚 Auto-ingesting sample documents on startup...")
            pipeline = get_ingestion()
            pipeline.ingest_directory(str(config.DOCS_DIR))
        else:
            logger.info(f"✓ Knowledge base has {store.count()} chunks ready")
    except Exception as e:
        logger.warning(f"Startup ingestion skipped: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.api:app", host=config.API_HOST, port=config.API_PORT, reload=True)
