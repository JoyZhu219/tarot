"""
prepare_knowledge_base.py

Step 1 of RAG pipeline: download Waite's Pictorial Key to the Tarot
as a single plain-text file from Internet Archive (public domain, 1910),
split into per-card chunks, and save to waite_chunks.jsonl.

Source: https://archive.org/details/A.EWaiteThePictorialKeyToTheTarot
License: Public domain (published 1910)

Output: rag/waite_chunks.jsonl

Chunk strategy:
  - Natural boundary: each card name is a section header in the text
  - chunk_size = 400 words, overlap = 50 words for long sections
  - Most card sections are 150-350 words -> single chunk each
  - Overlap prevents "Divinatory Meanings" from being split off

Usage:
    cd backend
    python rag/prepare_knowledge_base.py
"""

import json
import re
import urllib.request
from pathlib import Path

OUTPUT_PATH = Path(__file__).parent / "waite_chunks.jsonl"
SOURCE = "Waite, A.E. Pictorial Key to the Tarot (1910). Public domain."
ARCHIVE_URL = (
    "https://archive.org/stream/"
    "A.EWaiteThePictorialKeyToTheTarot/"
    "A.%20E%20Waite%20-%20The%20Pictorial%20Key%20to%20the%20Tarot_djvu.txt"
)

# Maps canonical card name -> how it appears as a heading in Waite's text
# (some Major Arcana use Roman numeral prefixes in the original)
CARD_ALIASES = {
    "The Magician":     "I. The Magician",
    "The Hermit":       "IX. The Hermit",
    "Wheel of Fortune": "X. Wheel of Fortune",
    "Death":            "XIII. Death",
}

# All 78 card names — used as section boundary markers
CARD_NAMES = [
    # Major Arcana
    "The Fool", "The Magician", "The High Priestess", "The Empress",
    "The Emperor", "The Hierophant", "The Lovers", "The Chariot",
    "Strength", "The Hermit", "Wheel of Fortune", "Justice",
    "The Hanged Man", "Death", "Temperance", "The Devil",
    "The Tower", "The Star", "The Moon", "The Sun", "Judgement", "The World",
    # Wands
    "King of Wands", "Queen of Wands", "Knight of Wands", "Page of Wands",
    "Ten of Wands", "Nine of Wands", "Eight of Wands", "Seven of Wands",
    "Six of Wands", "Five of Wands", "Four of Wands", "Three of Wands",
    "Two of Wands", "Ace of Wands",
    # Cups
    "King of Cups", "Queen of Cups", "Knight of Cups", "Page of Cups",
    "Ten of Cups", "Nine of Cups", "Eight of Cups", "Seven of Cups",
    "Six of Cups", "Five of Cups", "Four of Cups", "Three of Cups",
    "Two of Cups", "Ace of Cups",
    # Swords
    "King of Swords", "Queen of Swords", "Knight of Swords", "Page of Swords",
    "Ten of Swords", "Nine of Swords", "Eight of Swords", "Seven of Swords",
    "Six of Swords", "Five of Swords", "Four of Swords", "Three of Swords",
    "Two of Swords", "Ace of Swords",
    # Pentacles
    "King of Pentacles", "Queen of Pentacles", "Knight of Pentacles", "Page of Pentacles",
    "Ten of Pentacles", "Nine of Pentacles", "Eight of Pentacles", "Seven of Pentacles",
    "Six of Pentacles", "Five of Pentacles", "Four of Pentacles", "Three of Pentacles",
    "Two of Pentacles", "Ace of Pentacles",
]


def fetch_full_text(url):
    print(f"Downloading full text from Internet Archive...")
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "TarotRAG/1.0 (educational, public domain)"}
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8", errors="replace")


def clean_text(text):
    """Remove OCR artifacts and normalize whitespace."""
    # Remove page headers/footers typical of djvu OCR
    text = re.sub(r'\f', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def split_into_card_sections(text):
    """
    Split the full text into per-card sections using card names as boundaries.
    Returns dict: {card_name: section_text}
    """
    sections = {}

    for i, card_name in enumerate(CARD_NAMES):
        # Use alias if defined (e.g. "I. The Magician" instead of "The Magician")
        search_term = CARD_ALIASES.get(card_name, card_name)
        pattern = re.escape(search_term)
        matches = list(re.finditer(pattern, text, re.IGNORECASE))

        if not matches:
            print(f"  WARNING: '{card_name}' not found in text")
            continue

        # Take the last occurrence (Waite mentions cards in intro too)
        start = matches[-1].start()

        # End = start of next card section, or end of text
        end = len(text)
        for next_card in CARD_NAMES[i + 1:]:
            next_pattern = re.escape(next_card)
            next_matches = list(re.finditer(next_pattern, text[start + 1:], re.IGNORECASE))
            if next_matches:
                end = start + 1 + next_matches[0].start()
                break

        section = text[start:end].strip()
        if len(section) > 50:
            sections[card_name] = section

    return sections


def chunk_text(text, chunk_size=400, overlap=50):
    """
    Split text into word-based chunks.
    chunk_size=400 words (~500 tokens) fits embedding model limits.
    overlap=50 words bridges section boundaries.
    Most card sections are short enough to be a single chunk.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_knowledge_base():
    try:
        raw = fetch_full_text(ARCHIVE_URL)
    except Exception as e:
        print(f"ERROR downloading: {e}")
        print("Try running again or check your internet connection.")
        return

    text = clean_text(raw)
    print(f"Downloaded {len(text):,} characters.")

    sections = split_into_card_sections(text)
    print(f"Found sections for {len(sections)}/78 cards.")

    chunks_written = 0
    with open(OUTPUT_PATH, "w", encoding="utf-8") as out:
        for card_name, section in sections.items():
            chunks = chunk_text(section, chunk_size=400, overlap=50)
            for i, chunk in enumerate(chunks):
                record = {
                    "chunk_id": f"{card_name.lower().replace(' ', '_')}_{i}",
                    "card_name": card_name,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": SOURCE,
                    "url": ARCHIVE_URL,
                    "text": chunk,
                    "word_count": len(chunk.split()),
                }
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                chunks_written += 1

    print(f"\nDone. {chunks_written} chunks written to {OUTPUT_PATH}")
    missing = [c for c in CARD_NAMES if c not in sections]
    if missing:
        print(f"Missing cards ({len(missing)}): {missing}")
    print("\nNext step:")
    print("  Run: python rag/ingest_embeddings.py")


if __name__ == "__main__":
    build_knowledge_base()