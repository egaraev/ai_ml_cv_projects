"""
embedder.py — Handles chunking, embedding, and ingestion into Qdrant.
"""

import hashlib
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, ScrollRequest

import config


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
    )


def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )
    return splitter.split_text(text)


def chunk_hash(text: str) -> str:
    """MD5 hash of a chunk — used to detect duplicates."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_existing_hashes(client: QdrantClient, collection_name: str) -> set[str]:
    """Fetch all chunk hashes already stored in the collection."""
    hashes = set()
    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=None,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for point in results:
            h = point.payload.get("chunk_hash")
            if h:
                hashes.add(h)
        if offset is None:
            break
    return hashes


def ingest(text_array: list[dict], collection_name: str, append: bool = False):
    """
    text_array : list of {"text": "...", "metadata": {...}}
    append     : if True, keep existing data and skip duplicate chunks
                 if False (default), wipe and recreate the collection
    """
    embeddings = get_embeddings()
    client = QdrantClient(url=config.QDRANT_URL)

    # Build all chunks with hashes
    all_docs = []
    for item in text_array:
        for chunk in chunk_text(item["text"]):
            h = chunk_hash(chunk)
            metadata = {**item["metadata"], "chunk_hash": h}
            all_docs.append(Document(page_content=chunk, metadata=metadata))

    if append:
        # Ensure collection exists (create if not)
        existing_collections = [c.name for c in client.get_collections().collections]
        if collection_name not in existing_collections:
            vector_size = len(embeddings.embed_query("test"))
            client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            existing_hashes = set()
        else:
            existing_hashes = get_existing_hashes(client, collection_name)

        # Filter out chunks already in Qdrant
        new_docs = [d for d in all_docs if d.metadata["chunk_hash"] not in existing_hashes]
        skipped = len(all_docs) - len(new_docs)

        if skipped:
            print(f"⏭  Skipped {skipped} duplicate chunk(s)")
        if not new_docs:
            print("✓ Nothing new to add — all chunks already exist in the collection")
            return

        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        vector_store.add_documents(new_docs)
        print(f"✓ Appended {len(new_docs)} new chunk(s) into '{collection_name}'")

    else:
        # Wipe and recreate
        vector_size = len(embeddings.embed_query("test"))
        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=embeddings,
        )
        vector_store.add_documents(all_docs)
        print(f"✓ Ingested {len(all_docs)} chunks into '{collection_name}'")
