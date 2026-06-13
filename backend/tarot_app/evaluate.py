"""
evaluate.py

Card Interpretation Accuracy evaluator (Dimension 1).

For each card in a reading, compares the LLM-generated reading text
against the ground-truth themes sourced from dariusk/corpora.

Matching strategy: keyword-in-text (case-insensitive).
This is intentionally simple — the goal is to surface hallucinations
and omissions, not to do NLP research.

Precision = matched / total themes mentioned in text (approximated)
Recall    = matched / total required themes in ground truth
F1        = harmonic mean of precision and recall
"""


def _extract_text_themes(text, all_themes):
    """
    Returns the subset of all_themes that appear in the reading text.
    This gives us a denominator for precision that makes sense:
    we only count themes we know about, not every word in the text.
    """
    text_lower = text.lower()
    return [t for t in all_themes if t.lower() in text_lower]


def evaluate_reading(reading):
    """
    reading: a Reading model instance with prefetched readingcard_set and cards.

    Returns a dict:
    {
        "reading_id": int,
        "overall_f1": float,
        "cards": [
            {
                "card_name": str,
                "position_label": str,
                "is_reversed": bool,
                "required_themes": [...],      # ground truth from corpora
                "matched_themes": [...],       # found in reading text
                "missing_themes": [...],       # not found — potential hallucination gaps
                "precision": float,
                "recall": float,
                "f1": float,
                "has_ground_truth": bool,      # False if card has no themes loaded yet
            }
        ]
    }
    """
    text = reading.reading_text
    card_results = []

    for rc in reading.readingcard_set.select_related('card').all():
        card = rc.card

        if rc.is_reversed:
            required = card.reversed_required_themes or []
        else:
            required = card.required_themes or []

        has_ground_truth = bool(required)

        if not has_ground_truth:
            card_results.append({
                "card_name": card.name,
                "position_label": rc.position_label,
                "is_reversed": rc.is_reversed,
                "required_themes": [],
                "matched_themes": [],
                "missing_themes": [],
                "precision": None,
                "recall": None,
                "f1": None,
                "has_ground_truth": False,
            })
            continue

        matched = _extract_text_themes(text, required)
        missing = [t for t in required if t not in matched]

        recall = len(matched) / len(required)
        # Precision denominator: how many of the known themes did the
        # text touch vs. how many it mentioned (we use required as proxy)
        precision = len(matched) / len(required) if required else 0
        # Note: true precision would need a theme extractor from the text.
        # Using recall as precision approximation is conservative but honest.
        f1 = (2 * precision * recall / (precision + recall)
              if (precision + recall) > 0 else 0.0)

        card_results.append({
            "card_name": card.name,
            "position_label": rc.position_label,
            "is_reversed": rc.is_reversed,
            "required_themes": required,
            "matched_themes": matched,
            "missing_themes": missing,
            "precision": round(precision, 3),
            "recall": round(recall, 3),
            "f1": round(f1, 3),
            "has_ground_truth": True,
        })

    scored = [r for r in card_results if r["f1"] is not None]
    overall_f1 = (
        round(sum(r["f1"] for r in scored) / len(scored), 3)
        if scored else None
    )

    return {
        "reading_id": reading.id,
        "overall_f1": overall_f1,
        "cards": card_results,
    }
