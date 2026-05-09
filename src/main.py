import sys
from src.db.vector_store import upload_documents, collection_has_documents
from src.graph.workflow import run_agent


# ==============================================================
# Startup: Upload documents once
# ==============================================================

def startup():
    force_reindex = "--reindex" in sys.argv

    if force_reindex:
        print("🔄 Force re-index requested...")

    print("🚀 Starting company knowledge agent...")
    upload_documents(data_path="data/manuals", force=force_reindex)
    print("✅ Knowledge base ready.\n")


# ==============================================================
# CLI Mode — for testing the agent
# ==============================================================

def run_cli():
    """
    Interactive CLI for testing the agent.
    Type your question, get an answer.
    Type 'exit' to quit.
    """
    print("=" * 55)
    print("  🤖 Company Knowledge Agent — CLI Mode")
    print("  Supports Arabic & English questions")
    print("  Type 'exit' to quit")
    print("=" * 55)

    while True:
        try:
            question = input("\n📝 Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n👋 Goodbye!")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit", "خروج"):
            print("👋 Goodbye!")
            break

        print("\n⏳ Thinking...\n")

        result = run_agent(question)

        print(f"{'─' * 50}")
        print(f"📌 Route     : {result['route']}")
        print(f"🤖 LLM Used  : {result['llm_used']}")
        print(f"🔒 Sensitive : {result['is_sensitive']}")
        print(f"{'─' * 50}")
        print(f"💬 Answer:\n\n{result['answer']}")
        print(f"{'─' * 50}")


# ==============================================================
# Entry Point
# ==============================================================

if __name__ == "__main__":
    startup()   
    run_cli()   