"""
generator.py — Builds the prompt and generates an answer using the LLM.
"""

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from retriever import get_retriever
import config


def get_llm():
    return OllamaLLM(model=config.OLLAMA_MODEL, temperature=config.TEMPERATURE)


PROMPT_TEMPLATE = PromptTemplate.from_template("""You are a helpful assistant. Answer the question using ONLY the information provided in the context below.
List ALL relevant items from the context — do not omit any.
You may reason and connect facts within the context, but do NOT add any information from outside it.
If the context does not contain enough information to answer, say "I don't have that information in my database."

Context: {context}

Question: {question}

Answer:""")



def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def answer(question: str, collection_name: str) -> str:
    retriever = get_retriever(collection_name)
    llm = get_llm()
    
    # DEBUG — remove after testing
    docs = retriever.invoke(question)
    print("\n--- Retrieved chunks ---")
    for i, d in enumerate(docs):
        print(f"[{i+1}] {d.page_content}")
    print("------------------------\n")

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | PROMPT_TEMPLATE
        | llm
        | StrOutputParser()
    )

    return chain.invoke(question)
