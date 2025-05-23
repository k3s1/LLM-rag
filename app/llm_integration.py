# app/llm_integration.py
from typing import List
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from app.config import settings

# Define the RAG prompt template [cite: 8]
RAG_PROMPT_TEMPLATE = """
You are an AI assistant for question-answering over documents.
Use the following retrieved context to answer the question concisely and accurately.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Context:
{context}

Question: {question}

Answer:
"""

def get_llm():
    """Initializes and returns the appropriate LLM based on configuration."""
    if settings.LLM_PROVIDER == "openai":
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY not set in .env")
        # [cite: 2]
        return ChatOpenAI(api_key=settings.OPENAI_API_KEY, model="gpt-3.5-turbo", temperature=0.7)
    elif settings.LLM_PROVIDER == "gemini":
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in .env")
        # [cite: 2]
        return ChatGoogleGenerativeAI(google_api_key=settings.GOOGLE_API_KEY, model="gemini-pro", temperature=0.7)
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")

def generate_response_with_llm(query: str, retrieved_docs: List[Document]) -> str:
    """
    Generates a contextual response using the LLM and retrieved documents.
    [cite: 8]
    """
    llm = get_llm()
    prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

    # Format the context from retrieved documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])

    # Create the RAG chain [cite: 8]
    rag_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    # Invoke the chain [cite: 8]
    response = rag_chain.invoke({"context": context, "question": query})
    return response