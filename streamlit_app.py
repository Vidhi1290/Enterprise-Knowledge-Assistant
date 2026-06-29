"""
Streamlit Chat Interface — Enterprise Knowledge Assistant

Features:
- Chat-style conversation with memory
- Real-time source citations with confidence badges
- Document upload directly from browser
- Knowledge base stats sidebar
- Conversation history with export
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise Knowledge Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
        padding: 20px 30px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #e94560;
    }
    .main-header h1 { color: #e94560; margin: 0; font-size: 1.8rem; }
    .main-header p { color: #a8b2d8; margin: 5px 0 0 0; font-size: 0.9rem; }

    .chat-message-user {
        background: #1e3a5f;
        border-left: 3px solid #4fc3f7;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .chat-message-assistant {
        background: #1a2744;
        border-left: 3px solid #e94560;
        padding: 12px 16px;
        border-radius: 8px;
        margin: 8px 0;
    }
    .source-badge {
        display: inline-block;
        background: #0f3460;
        color: #4fc3f7;
        border: 1px solid #4fc3f7;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 3px;
    }
    .confidence-high { color: #4caf50; font-weight: bold; }
    .confidence-medium { color: #ff9800; font-weight: bold; }
    .confidence-low { color: #f44336; font-weight: bold; }
    .metric-card {
        background: #1a2744;
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 12px;
        text-align: center;
    }
    .stChatMessage { background: transparent !important; }

    /* Dark theme tweaks */
    .stTextInput input { background: #1e3a5f !important; color: white !important; }
    .stButton button {
        background: #e94560 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
    }
</style>
""", unsafe_allow_html=True)


# ── Initialize Session State ───────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "rag" not in st.session_state:
    st.session_state.rag = None
if "ingestion" not in st.session_state:
    st.session_state.ingestion = None


@st.cache_resource
def load_pipelines():
    """Load RAG and ingestion pipelines (cached across sessions)."""
    import config
    from app.ingestion import DocumentIngestionPipeline
    from app.rag import RAGPipeline

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    config.DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ingestion = DocumentIngestionPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    rag = RAGPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        top_k=config.TOP_K,
        min_similarity=config.MIN_SIMILARITY,
    )

    # Auto-ingest sample docs if DB is empty
    import config as cfg
    from app.vector_store import FAISSVectorStore
    store = FAISSVectorStore(str(cfg.CHROMA_DIR), cfg.COLLECTION_NAME)
    if store.count() == 0 and any(cfg.DOCS_DIR.rglob("*.*")):
        ingestion.ingest_directory(str(cfg.DOCS_DIR))

    return rag, ingestion


def confidence_badge(label: str) -> str:
    colors = {"high": "🟢", "medium": "🟡", "low": "🔴", "very_low": "⚫", "none": "⚫"}
    return colors.get(label, "⚪")


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 Enterprise Knowledge Assistant</h1>
    <p>Powered by Ollama (local) • ChromaDB • Hybrid RAG • No API keys needed</p>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    import config

    llm_choice = st.selectbox(
        "LLM Model",
        ["llama3.2:3b", "phi3:latest", "gpt-oss:20b"],
        index=0,
        help="llama3.2:3b = fast | gpt-oss:20b = smarter but slower",
    )
    config.LLM_MODEL = llm_choice

    top_k = st.slider("Chunks to retrieve (Top-K)", 3, 10, config.TOP_K)
    config.TOP_K = top_k

    query_rewrite = st.checkbox("Query rewriting (slower)", value=False, help="LLM rewrites your query before retrieval — adds ~3-10s but helps vague queries")

    st.markdown("---")
    st.markdown("## 📤 Upload Documents")

    uploaded_files = st.file_uploader(
        "Upload knowledge base documents",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "docx", "csv"],
        help="Supported: PDF, TXT, MD, DOCX, CSV",
    )

    if uploaded_files and st.button("📥 Ingest Documents", use_container_width=True):
        try:
            rag, ingestion = load_pipelines()
            import tempfile, shutil
            tmp_dir = Path(tempfile.mkdtemp())
            progress = st.progress(0)
            status = st.empty()

            for i, f in enumerate(uploaded_files):
                status.text(f"Processing {f.name}...")
                tmp_path = tmp_dir / f.name
                tmp_path.write_bytes(f.read())
                result = ingestion.ingest_file(tmp_path)
                progress.progress((i + 1) / len(uploaded_files))

                if result["status"] == "success":
                    st.success(f"✓ {f.name} ({result['chunks']} chunks)")
                elif result["status"] == "skipped":
                    st.info(f"⏭ {f.name} (already ingested)")
                else:
                    st.error(f"✗ {f.name}: {result.get('reason', 'error')}")

            shutil.rmtree(tmp_dir, ignore_errors=True)
            status.empty()
            st.cache_resource.clear()

        except Exception as e:
            st.error(f"Ingestion failed: {e}")

    st.markdown("---")
    st.markdown("## 📊 Knowledge Base Stats")

    try:
        rag, ingestion = load_pipelines()
        stats = rag.get_collection_stats()

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Chunks", stats["total_chunks"])
        with col2:
            st.metric("LLM", config.LLM_MODEL.split(":")[0])

        cache = stats.get("embed_cache", {})
        st.caption(f"Embed: {config.EMBED_MODEL} | Top-K: {config.TOP_K} | Cache: {cache.get('hit_rate','0%')}")

    except Exception as e:
        st.error(f"Could not connect to Ollama: {e}")
        st.info("Start Ollama: `ollama serve`")

    st.markdown("---")

    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    if st.button("💾 Export Chat", use_container_width=True):
        chat_export = json.dumps(st.session_state.messages, indent=2)
        st.download_button(
            "⬇️ Download JSON",
            chat_export,
            "chat_history.json",
            "application/json",
        )


# ── Chat Interface ─────────────────────────────────────────────────────────────

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💼" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])

        # Show sources for assistant messages
        if msg["role"] == "assistant" and msg.get("sources"):
            with st.expander(
                f"{confidence_badge(msg.get('confidence_label', 'medium'))} "
                f"Sources ({len(msg['sources'])}) | Confidence: {msg.get('confidence_label', 'N/A').upper()}"
            ):
                for src in msg["sources"]:
                    st.markdown(
                        f"📄 **{src['document']}** — Page {src['page']} "
                        f"(relevance: {src['relevance_score']:.2f})"
                    )

# Chat input
if question := st.chat_input("Ask a question about your documents..."):
    # Add user message
    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user", avatar="🧑‍💼"):
        st.markdown(question)

    # Generate response — streaming mode for fast perceived latency
    with st.chat_message("assistant", avatar="🤖"):
        try:
            import time
            rag, _ = load_pipelines()

            history = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages[:-1]
                if m["role"] in ("user", "assistant")
            ][-4:]

            t_start = time.time()

            # First: retrieve (fast — embed cache + ChromaDB)
            with st.spinner("🔍 Searching..."):
                search_q = rag.rewrite_query(question) if query_rewrite else question
                chunks, confidence = rag.retrieve(search_q)

            if not chunks:
                answer = "I don't have enough information in the knowledge base to answer this question."
                sources = []
                conf_label = "none"
                st.warning(answer)
            else:
                sources = rag._format_sources(chunks)
                conf_label = rag._confidence_label(confidence)
                prompt, system = rag._build_prompt(question, chunks, history if history else None)

                # Stream answer token by token — user sees output immediately
                answer = st.write_stream(rag.llm.generate_stream(prompt, system=system))

            latency = int((time.time() - t_start) * 1000)

            if sources:
                with st.expander(
                    f"{confidence_badge(conf_label)} Sources ({len(sources)}) | "
                    f"Confidence: {conf_label.upper()} ({confidence:.0%}) | ⚡ {latency}ms"
                ):
                    for src in sources:
                        st.markdown(
                            f"📄 **{src['document']}** — Page {src['page']} "
                            f"(relevance: {src['relevance_score']:.2f})"
                        )
            elif chunks:
                st.caption(f"⚡ {latency}ms")

            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "confidence": confidence if chunks else 0.0,
                "confidence_label": conf_label,
            })

        except Exception as e:
            err_msg = f"❌ Error: {str(e)}\n\nMake sure Ollama is running: `ollama serve`"
            st.error(err_msg)
            st.session_state.messages.append({"role": "assistant", "content": err_msg})

# Welcome message if no chat history
if not st.session_state.messages:
    st.markdown("""
    <div style="text-align: center; padding: 40px; color: #a8b2d8;">
        <h3>👋 Welcome!</h3>
        <p>Upload your documents in the sidebar, then start asking questions.</p>
        <br>
        <b>Example questions:</b><br>
        <i>"What is the employee leave policy?"</i><br>
        <i>"What are the refund terms?"</i><br>
        <i>"Summarize the compliance guidelines"</i>
    </div>
    """, unsafe_allow_html=True)
