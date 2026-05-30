from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


class RAG:
    def __init__(self, qdrant_url, embedding_model, llama_model):
        self.qdrant_url = qdrant_url
        self.embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model,
            model_kwargs={'device': 'cpu'}
        )
        self.llm = OllamaLLM(model=llama_model)

    def split_text_into_chunks(self, text):
        splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50)
        return splitter.split_text(text)

    def ingest_vector(self, text_array, collection_name):
        doc_array = []
        for item in text_array:
            chunks = self.split_text_into_chunks(item["text"])
            doc_array.extend([
                Document(page_content=chunk, metadata=item["metadata"])
                for chunk in chunks
            ])

        # Create collection explicitly (drop if exists)
        client = QdrantClient(url=self.qdrant_url)
        sample_vector = self.embeddings.embed_query("test")
        vector_size = len(sample_vector)

        client.recreate_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )

        # Add documents
        vector_store = QdrantVectorStore(
            client=client,
            collection_name=collection_name,
            embedding=self.embeddings,
        )
        vector_store.add_documents(doc_array)
        print("✓ Documents ingested successfully")

    def answer_question(self, question, collection_name):
        vector_store = QdrantVectorStore.from_existing_collection(
            collection_name=collection_name,
            url=self.qdrant_url,
            embedding=self.embeddings
        )
        retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        prompt = PromptTemplate.from_template("""
1. Use the following context to answer the question at the end.
2. If you don't know the answer, say "Cannot answer the question!" — don't make up an answer.
3. Keep the answer crisp and limited to 3-4 sentences.

                                              
Context: {context}

Question: {question}

Helpful Answer:""")

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
            | StrOutputParser()
        )

        result = chain.invoke(question)
        return {"answer": result}


# ── Configuration ──────────────────────────────────────────
QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "my_collection"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL    = "llama3:8b-instruct-q4_K_M"   # ← change to your model from `ollama list`

# ── Init ───────────────────────────────────────────────────
rag = RAG(qdrant_url=QDRANT_URL, embedding_model=EMBEDDING_MODEL, llama_model=OLLAMA_MODEL)

# ── Sample documents ───────────────────────────────────────
texts = [
    {"text": "India boasts a diverse landscape with mountain ranges including the Himalayas, Karakoram, Western Ghats, Eastern Ghats, and Aravalli.", "metadata": {"source": "wikipedia.com"}},
    {"text": "The Himalayas are grouped into three distinct ranges: the Greater, Middle, and Outer Himalaya Range, all in northern India.", "metadata": {"source": "study.com"}},
    {"text": "The Siachen Glacier in the Himalayas is one of the biggest glaciers outside the Arctic zones.", "metadata": {"source": "mapsofindia.com"}},
]

rag.ingest_vector(text_array=texts, collection_name=COLLECTION_NAME)

# ── Query ──────────────────────────────────────────────────
answer = rag.answer_question("What are the mountain ranges in India?", COLLECTION_NAME)
print("\nAnswer:", answer["answer"])
