"""
retriever.py — Connects to Qdrant and retrieves relevant document chunks.
"""

from langchain_qdrant import QdrantVectorStore
from embedder import get_embeddings
import config


def get_retriever(collection_name: str):
    embeddings = get_embeddings()
    vector_store = QdrantVectorStore.from_existing_collection(
        collection_name=collection_name,
        url=config.QDRANT_URL,
        embedding=embeddings,
    )
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": config.TOP_K},
    )



def retrieve(question: str, collection_name: str) -> list[str]:
    """Returns a list of relevant text chunks for a given question."""
    retriever = get_retriever(collection_name)
    docs = retriever.invoke(question)
    return [doc.page_content for doc in docs]
