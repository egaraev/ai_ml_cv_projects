"""
cli.py — Interactive CLI to query your RAG pipeline.

Usage:
    python cli.py
    python cli.py --collection my_docs
"""

import argparse
from generator import answer
import config


BANNER = """
╔══════════════════════════════════════════╗
║           RAG  Query  CLI                ║
║  Type your question and press Enter.     ║
║  Type 'exit' or Ctrl+C to quit.          ║
╚══════════════════════════════════════════╝
"""


def main():
    parser = argparse.ArgumentParser(description="Query your RAG pipeline interactively")
    parser.add_argument(
        "--collection",
        default=config.COLLECTION_NAME,
        help=f"Qdrant collection to query (default: {config.COLLECTION_NAME})",
    )
    args = parser.parse_args()

    print(BANNER)
    print(f"Collection : {args.collection}")
    print(f"Model      : {config.OLLAMA_MODEL}")
    print()

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            print("Bye!")
            break

        print("\nThinking...\n")
        try:
            result = answer(question=question, collection_name=args.collection)
            print(f"Answer: {result}\n")
            print("-" * 50)
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
