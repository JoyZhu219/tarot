"""
retriever.py  —  RAG search using fastembed + pgvector
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

from django.db import connection

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
_model = None


def _get_model():
    global _model
    if _model is None:
        from fastembed import TextEmbedding
        _model = TextEmbedding(model_name=EMBEDDING_MODEL)
    return _model


def _embed_query(text: str) -> list:
    model = _get_model()
    return list(model.embed([text]))[0].tolist()


def retrieve_context(card_name: str, top_k: int = 2) -> list:
    """
    Returns top_k most relevant Waite text chunks for a card.

    Args:
        card_name: e.g. "The Tower", "Five of Swords"
        top_k: number of chunks to return

    Returns:
        List of dicts: [{card_name, text, source, similarity}]
    """
    query = f"Tarot card meaning and symbolism: {card_name}"
    embedding = _embed_query(query)
    vector_str = "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    with connection.cursor() as cursor:
        cursor.execute("""
            SELECT
                card_name,
                text,
                source,
                1 - (embedding <=> %s::vector) AS similarity
            FROM rag_chunks
            WHERE card_name ILIKE %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (vector_str, card_name, vector_str, top_k))
        rows = cursor.fetchall()

    return [
        {
            "card_name": row[0],
            "text": row[1],
            "source": row[2],
            "similarity": round(float(row[3]), 3),
        }
        for row in rows
    ]


def retrieve_context_for_reading(card_objects: list, top_k: int = 2) -> dict:
    """Retrieves Waite context for all cards in a reading."""
    result = {}
    for item in card_objects:
        card_name = item["card"].name
        result[card_name] = retrieve_context(card_name, top_k=top_k)
    return result