"""
Configuration for Enterprise Knowledge Assistant
All settings in one place — easy to modify without touching app code.
"""

import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = DATA_DIR / "sample_docs"
CHROMA_DIR = DATA_DIR / "chroma_db"

# ── Ollama Models (fully local, no API keys needed) ────────────────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# LLM for answer generation — use llama3.2:3b for speed, gpt-oss:20b for quality
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# Embedding model — nomic-embed-text is fast and good for semantic search
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")

# ── Chunking Strategy ──────────────────────────────────────────────────────────
# 512 tokens with 50-token overlap balances context completeness vs retrieval precision
# Smaller chunks (256) = more precise retrieval but may lose context
# Larger chunks (1024) = more context but noisier retrieval
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# ── Retrieval Settings ─────────────────────────────────────────────────────────
# Number of chunks to retrieve per query
TOP_K = int(os.getenv("TOP_K", "3"))

# Minimum similarity score (cosine) to include a chunk — prevents hallucination
# from low-quality retrievals
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.3"))

# ── ChromaDB ──────────────────────────────────────────────────────────────────
COLLECTION_NAME = "enterprise_knowledge"

# ── API Settings ──────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

# ── Supported File Types ───────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".docx", ".csv"}

# ── Prompt Template ────────────────────────────────────────────────────────────
# Strict grounding — model must only answer from provided context
RAG_SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Your job is to answer employee questions accurately based ONLY on the provided document excerpts.

Rules:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. If the context does not contain enough information to answer, say: "I don't have enough information in the knowledge base to answer this question."
3. Be concise and direct.
4. Always mention which document(s) your answer comes from.
5. If the question is ambiguous, ask for clarification.

Context from knowledge base:
{context}

Question: {question}

Answer (cite sources inline):"""

# Confidence thresholds for API response
CONFIDENCE_HIGH = 0.7
CONFIDENCE_MED = 0.5
