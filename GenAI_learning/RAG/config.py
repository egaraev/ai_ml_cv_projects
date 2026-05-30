QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "my_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL    = "qwen2.5-coder:7b"  # change to your model from `ollama list`
CHUNK_SIZE      = 250
CHUNK_OVERLAP   = 50
TOP_K           = 5
TEMPERATURE     = 0     # 0 = deterministic, 1 = creative/random
