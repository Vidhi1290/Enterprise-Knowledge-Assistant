"""
Document Ingestion Pipeline
Handles: loading → text extraction → chunking → embedding → storing in FAISS

No ChromaDB, no sqlite. Pure FAISS + JSON.
"""

import os
import re
import hashlib
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ── Text Splitter ──────────────────────────────────────────────────────────────

class RecursiveTextSplitter:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = ["\n\n", "\n", ". ", "! ", "? ", " ", ""]

    def split_text(self, text: str) -> List[str]:
        chunks = []
        text = text.strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        separator = ""
        for sep in self.separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        current_chunk = []
        current_len = 0

        for split in splits:
            split_len = len(split)
            if current_len + split_len + len(separator) > self.chunk_size:
                if current_chunk:
                    chunk_text = separator.join(current_chunk).strip()
                    if chunk_text:
                        chunks.append(chunk_text)
                    overlap_len = 0
                    overlap_splits = []
                    for s in reversed(current_chunk):
                        overlap_len += len(s) + len(separator)
                        if overlap_len >= self.chunk_overlap:
                            break
                        overlap_splits.insert(0, s)
                    current_chunk = overlap_splits
                    current_len = sum(len(s) for s in current_chunk)
            current_chunk.append(split)
            current_len += split_len + len(separator)

        if current_chunk:
            chunk_text = separator.join(current_chunk).strip()
            if chunk_text:
                chunks.append(chunk_text)
        return chunks


# ── File Loaders ───────────────────────────────────────────────────────────────

def load_txt(filepath: Path) -> List[Dict]:
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        return [{"text": f.read(), "page": 1}]

def load_pdf(filepath: Path) -> List[Dict]:
    try:
        import pdfplumber
        pages = []
        with pdfplumber.open(filepath) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                text = page.extract_text() or ""
                if text.strip():
                    pages.append({"text": text, "page": i})
        return pages
    except Exception as e:
        logger.error(f"PDF load error: {e}")
        return []

def load_docx(filepath: Path) -> List[Dict]:
    try:
        import docx
        doc = docx.Document(filepath)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        return [{"text": text, "page": 1}]
    except Exception as e:
        logger.error(f"DOCX load error: {e}")
        return []

def load_csv(filepath: Path) -> List[Dict]:
    import csv
    rows = []
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(" | ".join(f"{k}: {v}" for k, v in row.items()))
    return [{"text": "\n".join(rows), "page": 1}]

LOADERS = {".txt": load_txt, ".md": load_txt, ".pdf": load_pdf, ".docx": load_docx, ".csv": load_csv}


# ── Ollama Embedder ────────────────────────────────────────────────────────────

class OllamaEmbedder:
    def __init__(self, model: str = "nomic-embed-text", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._test_connection()

    def _test_connection(self):
        import requests
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if any(self.model in m for m in models):
                logger.info(f"✓ Ollama connected. Using embed model: {self.model}")
            else:
                logger.warning(f"Model '{self.model}' not found. Available: {models}")
        except Exception as e:
            logger.error(f"Cannot connect to Ollama: {e}")
            raise

    def embed(self, text: str) -> List[float]:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        embeddings = []
        for i, text in enumerate(texts):
            if i % 10 == 0 and i > 0:
                logger.info(f"  Embedding {i}/{len(texts)}...")
            embeddings.append(self.embed(text))
        return embeddings


# ── Ingestion Pipeline ─────────────────────────────────────────────────────────

class DocumentIngestionPipeline:
    def __init__(
        self,
        store_dir: str = "./data/chroma_db",
        collection_name: str = "enterprise_knowledge",
        embed_model: str = "nomic-embed-text",
        ollama_url: str = "http://localhost:11434",
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        # Legacy params kept for API compat
        chroma_dir: str = None,
        **kwargs,
    ):
        from app.vector_store import FAISSVectorStore
        actual_dir = chroma_dir or store_dir
        self.embedder = OllamaEmbedder(model=embed_model, base_url=ollama_url)
        self.splitter = RecursiveTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.store = FAISSVectorStore(store_dir=actual_dir, collection_name=collection_name)

    def _file_hash(self, filepath: Path) -> str:
        h = hashlib.md5()
        with open(filepath, "rb") as f:
            h.update(f.read())
        return h.hexdigest()

    def ingest_file(self, filepath: Path) -> Dict[str, Any]:
        filepath = Path(filepath)
        ext = filepath.suffix.lower()

        if ext not in LOADERS:
            return {"status": "skipped", "reason": f"Unsupported: {ext}", "file": filepath.name}

        file_hash = self._file_hash(filepath)
        if self.store.already_ingested(file_hash):
            logger.info(f"  Skipping (already ingested): {filepath.name}")
            return {"status": "skipped", "reason": "Already ingested", "file": filepath.name}

        logger.info(f"  Loading: {filepath.name}")
        pages = LOADERS[ext](filepath)
        if not pages:
            return {"status": "error", "reason": "No text extracted", "file": filepath.name}

        all_chunks, all_ids, all_meta = [], [], []
        chunk_idx = 0
        for page_data in pages:
            for chunk in self.splitter.split_text(page_data["text"]):
                if len(chunk.strip()) < 30:
                    continue
                all_chunks.append(chunk)
                all_ids.append(f"{file_hash}_{chunk_idx}")
                all_meta.append({
                    "source": filepath.name,
                    "page": page_data["page"],
                    "chunk_index": chunk_idx,
                    "file_hash": file_hash,
                })
                chunk_idx += 1

        if not all_chunks:
            return {"status": "error", "reason": "No valid chunks", "file": filepath.name}

        logger.info(f"  Generating {len(all_chunks)} embeddings for {filepath.name}...")
        embeddings = self.embedder.embed_batch(all_chunks)

        # Add in batches of 100
        for i in range(0, len(all_chunks), 100):
            self.store.add(
                ids=all_ids[i:i+100],
                documents=all_chunks[i:i+100],
                embeddings=embeddings[i:i+100],
                metadatas=all_meta[i:i+100],
            )

        logger.info(f"  ✓ Ingested {len(all_chunks)} chunks from {filepath.name}")
        return {"status": "success", "file": filepath.name, "chunks": len(all_chunks), "pages": len(pages)}

    def ingest_directory(self, docs_dir: str) -> List[Dict]:
        docs_dir = Path(docs_dir)
        files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in LOADERS and f.is_file()]
        if not files:
            logger.warning(f"No supported files in {docs_dir}")
            return []
        logger.info(f"\n📚 Ingesting {len(files)} files from {docs_dir}...")
        results = [self.ingest_file(f) for f in files]
        success = [r for r in results if r["status"] == "success"]
        logger.info(f"\n✅ Done: {len(success)}/{len(files)} files")
        return results

    def get_stats(self) -> Dict:
        return self.store.get_stats()
