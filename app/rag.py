"""
RAG Pipeline — FAISS backend (no ChromaDB, no sqlite)
Speed optimizations: embed cache, streaming, keep_alive, smaller prompts
"""

import re
import json
import time
import logging
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url
        self._warm_up()

    def _warm_up(self):
        import requests
        try:
            requests.post(
                f"{self.base_url}/api/generate",
                json={"model": self.model, "keep_alive": -1, "prompt": ""},
                timeout=30,
            )
            logger.info(f"✓ LLM warm: {self.model}")
        except Exception:
            pass

    def generate(self, prompt: str, system: str = "", temperature: float = 0.1) -> str:
        import requests
        resp = requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model, "prompt": prompt, "system": system,
                "stream": False, "keep_alive": -1,
                "options": {"temperature": temperature, "num_predict": 512, "num_ctx": 3072},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"].strip()

    def generate_stream(self, prompt: str, system: str = "", temperature: float = 0.1):
        import requests
        with requests.post(
            f"{self.base_url}/api/generate",
            json={
                "model": self.model, "prompt": prompt, "system": system,
                "stream": True, "keep_alive": -1,
                "options": {"temperature": temperature, "num_predict": 512, "num_ctx": 3072},
            },
            stream=True, timeout=120,
        ) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")
                    if chunk.get("done"):
                        break


class CachedEmbedder:
    """LRU embed cache — repeated queries cost 0ms instead of 200-800ms."""
    def __init__(self, embedder):
        self._embedder = embedder
        self._cache: Dict[str, List[float]] = {}
        self._hits = self._misses = 0

    def embed(self, text: str) -> List[float]:
        key = text.lower().strip()
        if key in self._cache:
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        emb = self._embedder.embed(text)
        if len(self._cache) > 256:
            del self._cache[next(iter(self._cache))]
        self._cache[key] = emb
        return emb

    def embed_batch(self, texts):
        return self._embedder.embed_batch(texts)

    @property
    def cache_stats(self):
        total = self._hits + self._misses
        return {"hits": self._hits, "misses": self._misses,
                "hit_rate": f"{self._hits/total:.0%}" if total else "0%"}


def bm25_score(query: str, document: str) -> float:
    query_terms = re.findall(r'\w+', query.lower())
    doc_terms = re.findall(r'\w+', document.lower())
    if not doc_terms:
        return 0.0
    term_freq = defaultdict(int)
    for t in doc_terms:
        term_freq[t] += 1
    k1, b, avg_len = 1.5, 0.75, 150
    score = 0.0
    for term in query_terms:
        tf = term_freq.get(term, 0)
        score += (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * len(doc_terms) / avg_len))
    return score


def rrf_fusion(semantic: List[Dict], keyword: List[Dict], k=60, sw=0.7, kw=0.3) -> List[Dict]:
    scores = defaultdict(float)
    doc_map = {}
    for rank, doc in enumerate(semantic):
        scores[doc["id"]] += sw / (k + rank + 1)
        doc_map[doc["id"]] = doc
    for rank, doc in enumerate(keyword):
        scores[doc["id"]] += kw / (k + rank + 1)
        doc_map.setdefault(doc["id"], doc)
    return [doc_map[i] for i, _ in sorted(scores.items(), key=lambda x: -x[1])]


class RAGPipeline:
    def __init__(
        self,
        store_dir: str = "./data/chroma_db",
        collection_name: str = "enterprise_knowledge",
        embed_model: str = "nomic-embed-text",
        llm_model: str = "llama3.2:3b",
        ollama_url: str = "http://localhost:11434",
        top_k: int = 3,
        min_similarity: float = 0.3,
        # legacy compat params
        chroma_dir: str = None,
        **kwargs,
    ):
        from app.ingestion import OllamaEmbedder
        from app.vector_store import FAISSVectorStore

        actual_dir = chroma_dir or store_dir
        raw_embedder = OllamaEmbedder(model=embed_model, base_url=ollama_url)
        self.embedder = CachedEmbedder(raw_embedder)
        self.llm = OllamaLLM(model=llm_model, base_url=ollama_url)
        self.store = FAISSVectorStore(store_dir=actual_dir, collection_name=collection_name)
        self.top_k = top_k
        self.min_similarity = min_similarity
        logger.info(f"✓ RAG ready | LLM: {llm_model} | top_k: {top_k}")

    def retrieve(self, query: str) -> Tuple[List[Dict], float]:
        if self.store.count() == 0:
            return [], 0.0

        t0 = time.time()
        query_emb = self.embedder.embed(query)
        raw = self.store.query(query_emb, n_results=self.top_k * 2)

        # Filter by min similarity
        semantic = [r for r in raw if r["score"] >= self.min_similarity]
        if not semantic:
            return [], 0.0

        # BM25 on candidates + RRF
        keyword = sorted(semantic, key=lambda d: bm25_score(query, d["text"]), reverse=True)
        fused = rrf_fusion(semantic, keyword)[:self.top_k]

        avg_conf = sum(d["score"] for d in fused) / len(fused)
        logger.info(f"  Retrieval: {(time.time()-t0)*1000:.0f}ms | {len(fused)} chunks | conf={avg_conf:.2f}")
        return fused, avg_conf

    def rewrite_query(self, query: str) -> str:
        prompt = f"Rewrite this question for enterprise document search. Return ONLY the rewritten query (<20 words).\nQuestion: {query}\nRewritten:"
        try:
            r = self.llm.generate(prompt, temperature=0.1)
            return r if 5 < len(r) < 200 else query
        except Exception:
            return query

    def _build_prompt(self, question: str, chunks: List[Dict], history: Optional[List[Dict]]) -> Tuple[str, str]:
        context = "\n\n".join(
            f"[{i}] {c['metadata'].get('source','?')} p.{c['metadata'].get('page','?')}\n{c['text']}"
            for i, c in enumerate(chunks, 1)
        )
        history_text = ""
        if history:
            for turn in history[-2:]:
                role = "User" if turn["role"] == "user" else "Assistant"
                content = turn["content"][:200] + "..." if len(turn["content"]) > 200 else turn["content"]
                history_text += f"{role}: {content}\n"
            history_text += "\n"

        system = "You are an Enterprise Knowledge Assistant. Answer ONLY from the provided context. Be concise. Cite sources as (Source N)."
        prompt = f"{history_text}Context:\n{context}\n\nQuestion: {question}\n\nAnswer (cite sources, be brief):"
        return prompt, system

    def _format_sources(self, chunks: List[Dict]) -> List[Dict]:
        seen, sources = set(), []
        for c in chunks:
            meta = c["metadata"]
            key = (meta.get("source", ""), meta.get("page", 0))
            if key not in seen:
                seen.add(key)
                sources.append({"document": meta.get("source", "Unknown"),
                                 "page": meta.get("page", 1),
                                 "relevance_score": round(c["score"], 3)})
        return sources

    def _confidence_label(self, score: float) -> str:
        if score >= 0.7: return "high"
        if score >= 0.5: return "medium"
        if score >= 0.3: return "low"
        return "very_low"

    def ask(self, question: str, conversation_history=None,
            use_query_rewriting: bool = False, stream: bool = False) -> Dict[str, Any]:
        t0 = time.time()
        search_q = self.rewrite_query(question) if use_query_rewriting else question
        chunks, confidence = self.retrieve(search_q)

        if not chunks:
            return {"answer": "I don't have enough information in the knowledge base to answer this question.",
                    "sources": [], "confidence": 0.0, "confidence_label": "none",
                    "chunks_used": 0, "question": question, "latency_ms": int((time.time()-t0)*1000)}

        prompt, system = self._build_prompt(question, chunks, conversation_history)
        sources = self._format_sources(chunks)

        if stream:
            return {"answer_stream": self.llm.generate_stream(prompt, system=system),
                    "sources": sources, "confidence": round(confidence, 3),
                    "confidence_label": self._confidence_label(confidence),
                    "chunks_used": len(chunks), "question": question}

        answer = self.llm.generate(prompt, system=system)
        return {"answer": answer, "sources": sources, "confidence": round(confidence, 3),
                "confidence_label": self._confidence_label(confidence),
                "chunks_used": len(chunks), "question": question,
                "latency_ms": int((time.time()-t0)*1000)}

    def get_collection_stats(self) -> Dict:
        stats = self.store.get_stats()
        stats["embed_cache"] = self.embedder.cache_stats
        return stats
