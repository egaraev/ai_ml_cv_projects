"""
main.py — Entry point. Run ingestion and query the RAG pipeline.
"""

import config
from embedder import ingest
from generator import answer

# ── Sample documents (replace with your own data) ──────────
texts = [
    {"text": "India boasts a diverse landscape with mountain ranges including the Himalayas, Karakoram, Western Ghats, Eastern Ghats, and Aravalli.", "metadata": {"source": "wikipedia.com"}},
    {"text": "The Himalayas are grouped into three distinct ranges: the Greater, Middle, and Outer Himalaya Range, all in northern India.", "metadata": {"source": "study.com"}},
    {"text": "The Siachen Glacier in the Himalayas is one of the biggest glaciers outside the Arctic zones.", "metadata": {"source": "mapsofindia.com"}},
]


# ── Ingest ─────────────────────────────────────────────────
ingest(text_array=texts, collection_name=config.COLLECTION_NAME)

# ── Query ──────────────────────────────────────────────────
question = "What are the mountain ranges in India?"
result = answer(question=question, collection_name=config.COLLECTION_NAME)
print("\nAnswer:", result)
