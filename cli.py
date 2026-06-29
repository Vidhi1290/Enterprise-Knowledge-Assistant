"""
CLI Tool — Enterprise Knowledge Assistant

Usage:
  python cli.py ingest ./data/sample_docs
  python cli.py ask "What is the leave policy?"
  python cli.py ask "What is the refund policy?" --model gpt-oss:20b
  python cli.py stats
  python cli.py interactive
"""

import sys
import json
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)  # suppress info for clean CLI output

import config


def cmd_ingest(args):
    """Ingest documents from a directory or single file."""
    from app.ingestion import DocumentIngestionPipeline

    pipeline = DocumentIngestionPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    target = Path(args.path)

    if target.is_file():
        print(f"📄 Ingesting file: {target}")
        result = pipeline.ingest_file(target)
        print(json.dumps(result, indent=2))
    elif target.is_dir():
        print(f"📚 Ingesting directory: {target}")
        results = pipeline.ingest_directory(str(target))
        successful = [r for r in results if r.get("status") == "success"]
        print(f"\n✅ Done: {len(successful)}/{len(results)} files ingested")
        for r in results:
            status_icon = {"success": "✓", "skipped": "⏭", "error": "✗"}.get(r["status"], "?")
            print(f"  {status_icon} {r['file']} — {r.get('chunks', '')} chunks | {r.get('reason', '')}")
    else:
        print(f"❌ Path not found: {target}")
        sys.exit(1)


def cmd_ask(args):
    """Ask a single question."""
    from app.rag import RAGPipeline

    if args.model:
        config.LLM_MODEL = args.model

    rag = RAGPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        top_k=config.TOP_K,
        min_similarity=config.MIN_SIMILARITY,
    )

    print(f"\n🔍 Question: {args.question}\n")
    print("─" * 60)

    result = rag.ask(
        question=args.question,
        use_query_rewriting=not args.no_rewrite,
    )

    print(f"\n📝 Answer:\n{result['answer']}\n")
    print("─" * 60)

    if result["sources"]:
        print(f"\n📚 Sources (Confidence: {result['confidence_label'].upper()} {result['confidence']:.0%}):")
        for src in result["sources"]:
            print(f"  • {src['document']} — Page {src['page']} (relevance: {src['relevance_score']:.2f})")
    else:
        print("\n⚠️  No relevant sources found.")

    if args.json:
        print("\n" + "─" * 60)
        print("JSON Output:")
        print(json.dumps(result, indent=2))


def cmd_stats(args):
    """Show knowledge base statistics."""
    from app.rag import RAGPipeline

    rag = RAGPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
    )
    stats = rag.get_collection_stats()
    print("\n📊 Knowledge Base Stats")
    print("─" * 40)
    print(f"  Total chunks: {stats['total_chunks']}")
    print(f"  Collection:   {stats['collection']}")
    print(f"  LLM model:    {config.LLM_MODEL}")
    print(f"  Embed model:  {config.EMBED_MODEL}")
    print(f"  Chunk size:   {config.CHUNK_SIZE} chars")
    print(f"  Top-K:        {config.TOP_K}")


def cmd_interactive(args):
    """Start an interactive Q&A session in the terminal."""
    from app.rag import RAGPipeline

    if args.model:
        config.LLM_MODEL = args.model

    print("\n🧠 Enterprise Knowledge Assistant — Interactive Mode")
    print("   Type 'quit' or 'exit' to stop | 'clear' to reset history\n")

    rag = RAGPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        top_k=config.TOP_K,
        min_similarity=config.MIN_SIMILARITY,
    )

    stats = rag.get_collection_stats()
    print(f"✓ Knowledge base: {stats['total_chunks']} chunks | LLM: {config.LLM_MODEL}\n")

    history = []

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in ("quit", "exit"):
            print("Goodbye!")
            break
        if question.lower() == "clear":
            history = []
            print("✓ Conversation history cleared\n")
            continue

        result = rag.ask(
            question=question,
            conversation_history=history[-8:] if history else None,
            use_query_rewriting=True,
        )

        print(f"\nAssistant: {result['answer']}\n")

        if result["sources"]:
            conf_icon = {"high": "🟢", "medium": "🟡", "low": "🔴"}.get(result["confidence_label"], "⚫")
            print(f"{conf_icon} Sources ({result['confidence_label']}): ", end="")
            src_parts = [f"{s['document']} p.{s['page']}" for s in result["sources"]]
            print(", ".join(src_parts))
        print()

        history.append({"role": "user", "content": question})
        history.append({"role": "assistant", "content": result["answer"]})


def cmd_eval(args):
    """Run evaluation on test questions."""
    from app.rag import RAGPipeline

    # Load test questions
    test_file = Path(args.test_file) if args.test_file else Path("tests/test_questions.json")
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        print("Create tests/test_questions.json with format:")
        print('[{"question": "...", "expected_keywords": ["...", "..."]}]')
        sys.exit(1)

    with open(test_file) as f:
        test_cases = json.load(f)

    rag = RAGPipeline(
        chroma_dir=str(config.CHROMA_DIR),
        collection_name=config.COLLECTION_NAME,
        embed_model=config.EMBED_MODEL,
        llm_model=config.LLM_MODEL,
        ollama_url=config.OLLAMA_BASE_URL,
        top_k=config.TOP_K,
        min_similarity=config.MIN_SIMILARITY,
    )

    print(f"\n🧪 Running {len(test_cases)} test cases...\n")
    results = []

    for i, tc in enumerate(test_cases, 1):
        question = tc["question"]
        expected_keywords = tc.get("expected_keywords", [])

        print(f"[{i}/{len(test_cases)}] {question[:60]}...")
        result = rag.ask(question=question, use_query_rewriting=True)

        # Check if answer contains expected keywords (basic evaluation)
        answer_lower = result["answer"].lower()
        keyword_hits = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
        keyword_score = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

        has_sources = len(result["sources"]) > 0
        not_hallucinating = "don't have" not in answer_lower or keyword_score > 0

        test_result = {
            "question": question,
            "answer": result["answer"][:200] + "...",
            "confidence": result["confidence"],
            "confidence_label": result["confidence_label"],
            "sources": result["sources"],
            "keyword_score": round(keyword_score, 2),
            "has_sources": has_sources,
            "pass": keyword_score >= 0.5 and has_sources,
        }
        results.append(test_result)

        status = "✅ PASS" if test_result["pass"] else "❌ FAIL"
        print(f"  {status} | Confidence: {result['confidence']:.2f} | Keywords: {keyword_hits}/{len(expected_keywords)}")

    # Summary
    passed = sum(1 for r in results if r["pass"])
    avg_confidence = sum(r["confidence"] for r in results) / len(results)

    print("\n" + "=" * 60)
    print(f"📊 EVALUATION SUMMARY")
    print(f"  Pass rate:        {passed}/{len(results)} ({passed/len(results):.0%})")
    print(f"  Avg confidence:   {avg_confidence:.2f}")
    print(f"  Source coverage:  {sum(1 for r in results if r['has_sources'])}/{len(results)}")

    # Save results
    output = Path("tests/eval_results.json")
    output.parent.mkdir(exist_ok=True)
    with open(output, "w") as f:
        json.dump({"summary": {"passed": passed, "total": len(results), "avg_confidence": avg_confidence}, "results": results}, f, indent=2)
    print(f"\n  Results saved → {output}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Enterprise Knowledge Assistant CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python cli.py ingest ./data/sample_docs
  python cli.py ask "What is the leave policy?"
  python cli.py ask "Refund terms?" --model gpt-oss:20b --json
  python cli.py interactive
  python cli.py stats
  python cli.py eval --test-file tests/test_questions.json
        """,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest
    p_ingest = subparsers.add_parser("ingest", help="Ingest documents")
    p_ingest.add_argument("path", help="File or directory to ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    # ask
    p_ask = subparsers.add_parser("ask", help="Ask a question")
    p_ask.add_argument("question", help="Your question")
    p_ask.add_argument("--model", help="Override LLM model")
    p_ask.add_argument("--no-rewrite", action="store_true", help="Disable query rewriting")
    p_ask.add_argument("--json", action="store_true", help="Also output raw JSON")
    p_ask.set_defaults(func=cmd_ask)

    # stats
    p_stats = subparsers.add_parser("stats", help="Show knowledge base stats")
    p_stats.set_defaults(func=cmd_stats)

    # interactive
    p_inter = subparsers.add_parser("interactive", help="Interactive Q&A session")
    p_inter.add_argument("--model", help="Override LLM model")
    p_inter.set_defaults(func=cmd_interactive)

    # eval
    p_eval = subparsers.add_parser("eval", help="Run evaluation test suite")
    p_eval.add_argument("--test-file", help="Path to test_questions.json")
    p_eval.set_defaults(func=cmd_eval)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
