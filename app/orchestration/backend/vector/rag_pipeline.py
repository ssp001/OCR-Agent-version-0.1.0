from typing import List
from vectordb_mcp import search_vectors


def build_prompt(context_chunks: List[str], query: str) -> str:
    """
    Build structured RAG prompt
    """

    context = "\n\n".join(context_chunks)

    prompt = f"""
You are a helpful AI assistant.
Answer the question ONLY using the provided context.
If the answer is not in the context, say:
"I could not find the answer in the provided documents."

---------------------
Context:
{context}
---------------------

Question:
{query}

Answer clearly and professionally.
"""

    return prompt.strip()


def rag_pipeline(query: str, chat_id: str) -> str:
    """
    1. Retrieve relevant documents
    2. Build prompt with context
    3. Return final prompt for LLM
    """

    # Step 1: Retrieve similar documents
    retrieved_docs = search_vectors(query=query, chat_id=chat_id, limit=5)

    # Step 2: If no docs found
    if not retrieved_docs:
        return query  # fallback to normal LLM response

    # Step 3: Build structured prompt
    prompt = build_prompt(retrieved_docs, query)

    return prompt
