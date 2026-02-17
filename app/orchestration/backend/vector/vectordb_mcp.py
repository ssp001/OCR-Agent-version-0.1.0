import logging
from dotenv import load_dotenv
from mcp_use.server import MCPServer
from sentence_transformers import SentenceTransformer
import psycopg2
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["MCP_USE_ANONYMIZED_TELEMETRY"] = "false"

logging.basicConfig(level=logging.CRITICAL)

load_dotenv()

# === MCP SERVER ===
server = MCPServer("vector-db")

# === MODEL ===
model = SentenceTransformer("all-MiniLM-L6-v2")


def embed(text: str):
    return model.encode(text).tolist()


def search_vectors(query: str, chat_id: str, limit: int = 5):
    embedding = embed(query)

    conn = psycopg2.connect(
        host="localhost",
        port=5432,
        database="vectordb",
        user="ssp001",
        password="admin"
    )
    cur = conn.cursor()

    cur.execute("""
        SELECT content
        FROM documents
        WHERE chat_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s
    """, (chat_id, embedding, limit))

    results = cur.fetchall()
    cur.close()
    conn.close()

    return [r[0] for r in results]


def store_document(content: str, chat_id: str):
    embedding = embed(content)

    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        database="vectordb",
        user="ssp001",
        password="admin"
    )
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO documents (content, embedding, chat_id)
        VALUES (%s, %s, %s);
    """, (content, embedding, chat_id))

    conn.commit()
    cur.close()
    conn.close()

    return "Document stored successfully."


# === MCP TOOLS ===

@server.tool()
async def vector_search(query: str, limit: int = 5):
    """Search similar documents inside this chat"""
    return search_vectors(query, limit)


@server.tool()
async def vector_store(content: str):
    """Store document chunk for a specific chat"""
    return store_document(content)


# === RUN MCP SERVER ===
if __name__ == "__main__":
    server.run(transport="stdio")
