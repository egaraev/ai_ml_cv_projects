"""
ingest.py — Load a .txt file into Qdrant.

Usage:
    python ingest.py data.txt                        # wipe collection and re-ingest
    python ingest.py data.txt --append               # add only new chunks, skip duplicates
    python ingest.py data.txt --collection my_docs   # use a custom collection name
    python ingest.py data.txt --append --collection my_docs
"""

import sys
import argparse
from pathlib import Path
from embedder import ingest
import config


def load_txt(path: Path) -> list[dict]:
    """Split txt file into paragraphs, each treated as a document."""
    text = path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    return [
        {"text": p, "metadata": {"source": str(path.name)}}
        for p in paragraphs
    ]


def main():
    parser = argparse.ArgumentParser(description="Ingest a .txt file into Qdrant")
    parser.add_argument("file", help="Path to the .txt file to ingest")
    parser.add_argument(
        "--collection",
        default=config.COLLECTION_NAME,
        help=f"Qdrant collection name (default: {config.COLLECTION_NAME})",
    )
    parser.add_argument(
        "--append",
        action="store_true",
        help="Append to existing collection, skipping duplicate chunks (default: wipe and recreate)",
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"Error: file not found: {path}")
        sys.exit(1)
    if path.suffix.lower() != ".txt":
        print(f"Error: only .txt files are supported, got: {path.suffix}")
        sys.exit(1)

    mode = "append (dedup)" if args.append else "recreate"
    print(f"Loading '{path.name}' [{mode}]...")
    docs = load_txt(path)
    print(f"Found {len(docs)} paragraphs.")
    ingest(text_array=docs, collection_name=args.collection, append=args.append)


if __name__ == "__main__":
    main()

