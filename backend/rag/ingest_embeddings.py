"""
ingest_embeddings.py  —  Step 2 of RAG pipeline

Uses fastembed (lightweight, no PyTorch required, ~50MB)
to generate embeddings and stores into pgvector.

Model: BAAI/bge-small-en-v1.5 (384 dims, fast, good quality)

Usage:
    cd backend
    python rag/ingest_embeddings.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

from django.db import connection

CHUNKS_PATH = Path(__file__).parent / "waite_chunks.jsonl"
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384


def load_chunks():
    chunks = []
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            chunks.append(json.loads(line.strip()))
    return chunks


def setup_pgvector(cursor):
    print("Setting up pgvector...")
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector;")
    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS rag_chunks (
            id          SERIAL PRIMARY KEY,
            chunk_id    TEXT UNIQUE NOT NULL,
            card_name   TEXT NOT NULL,
            chunk_index INTEGER NOT NULL,
            source      TEXT NOT NULL,
            url         TEXT,
            text        TEXT NOT NULL,
            word_count  INTEGER,
            embedding   vector({EMBEDDING_DIM})
        );
    """)
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS rag_chunks_embedding_idx
        ON rag_chunks
        USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 10);
    """)
    print("  v rag_chunks table ready")


def embed_chunks(chunks):
    from fastembed import TextEmbedding
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    model = TextEmbedding(model_name=EMBEDDING_MODEL)
    texts = [c["text"] for c in chunks]
    print(f"Embedding {len(texts)} chunks...")
    embeddings = list(model.embed(texts))
    print(f"  v {len(embeddings)} embeddings generated")
    return embeddings


def upsert_chunks(cursor, chunks, embeddings):
    print("Upserting into database...")
    for chunk, embedding in zip(chunks, embeddings):
        vector_str = "[" + ",".join(f"{x:.6f}" for x in embedding.tolist()) + "]"
        cursor.execute("""
            INSERT INTO rag_chunks
                (chunk_id, card_name, chunk_index, source, url, text, word_count, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (chunk_id) DO UPDATE SET
                text      = EXCLUDED.text,
                embedding = EXCLUDED.embedding;
        """, (
            chunk["chunk_id"],
            chunk["card_name"],
            chunk["chunk_index"],
            chunk["source"],
            chunk.get("url", ""),
            chunk["text"],
            chunk.get("word_count", 0),
            vector_str,
        ))
    print(f"  v {len(chunks)} chunks upserted")


def main():
    if not CHUNKS_PATH.exists():
        print(f"ERROR: {CHUNKS_PATH} not found. Run prepare_knowledge_base.py first.")
        return

    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks")

    embeddings = embed_chunks(chunks)

    with connection.cursor() as cursor:
        setup_pgvector(cursor)
        upsert_chunks(cursor, chunks, embeddings)

    print("\nDone. Knowledge base ready.")


if __name__ == "__main__":
    main()