"""
judge.py  —  Layer 2: LLM-as-Judge

Input:  Reading instance (with cards + reading_text from DB)
Output: VerificationReport saved to DB with:
        - Per-claim verdicts: VERIFIED / UNVERIFIED / HALLUCINATION
        - Overall status: 'ok' or 'needs_review'
        - Precision / Recall / F1 scores

The judge LLM acts as an auditor, not a reader.
It receives structured ground truth (from DB) alongside the generated
reading, and evaluates each claim against what the cards actually mean.
"""

import json
import anthropic
from django.conf import settings
from .models import VerificationReport


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def _build_cards_block_for_judge(card_data: list[dict]) -> str:
    lines = []
    for cd in card_data:
        orientation = "REVERSED" if cd["is_reversed"] else "upright"
        themes_key = "reversed_required_themes" if cd["is_reversed"] else "required_themes"
        themes = cd[themes_key]
        lines += [
            f"### {cd['position_label']}: {cd['card_name']} ({orientation})",
            f"Official keywords: {cd['keywords']}",
            f"Official themes for this orientation: {json.dumps(themes)}",
            "",
        ]
    return "\n".join(lines)


def _build_judge_prompt(reading, card_data: list[dict]) -> str:
    from prompts.prompt_manager import prompt_manager
    return prompt_manager.render(
        "reading_judge",
        user_name=reading.user_name,
        question=reading.question,
        spread_type=reading.spread_type,
        cards_block=_build_cards_block_for_judge(card_data),
        reading_text=reading.reading_text,
    )


# ---------------------------------------------------------------------------
# F1 calculation
# ---------------------------------------------------------------------------

def _compute_metrics(claims: list[dict], all_ground_truth_themes: list[str],
                     covered_themes: list[str]) -> dict:
    verified = sum(1 for c in claims if c["verdict"] == "VERIFIED")
    hallucinations = sum(1 for c in claims if c["verdict"] == "HALLUCINATION")
    total_claims = verified + hallucinations  # UNVERIFIED excluded from precision

    precision = verified / total_claims if total_claims > 0 else 0.0

    total_gt = len(all_ground_truth_themes)
    recall = len(covered_themes) / total_gt if total_gt > 0 else 0.0

    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 else 0.0)

    return {
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1": round(f1, 3),
    }


# ---------------------------------------------------------------------------
# Main judge entry point
# ---------------------------------------------------------------------------

def run_judge(reading) -> VerificationReport:
    """
    Runs LLM-as-Judge on a Reading instance.
    Saves and returns a VerificationReport.
    Called automatically after reading generation.
    """
    # Collect card data + all ground truth themes
    reading_cards = reading.readingcard_set.select_related('card').all()
    card_data = []
    all_gt_themes = []

    for rc in reading_cards:
        card = rc.card
        upright = card.required_themes or []
        reversed_ = card.reversed_required_themes or []
        card_data.append({
            "card_name": card.name,
            "position_label": rc.position_label,
            "is_reversed": rc.is_reversed,
            "keywords": card.keywords,
            "required_themes": upright,
            "reversed_required_themes": reversed_,
        })
        # Ground truth = whichever orientation was drawn
        all_gt_themes += (reversed_ if rc.is_reversed else upright)

    # Build prompt and call LLM
    prompt = _build_judge_prompt(reading, card_data)
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = message.content[0].text.strip()

    # Parse JSON response
    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        # Strip markdown fences if present
        clean = raw.replace("```json", "").replace("```", "").strip()
        result = json.loads(clean)

    claims = result.get("claims", [])
    covered_themes = result.get("covered_themes", [])

    # Compute metrics
    metrics = _compute_metrics(claims, all_gt_themes, covered_themes)

    # Determine status
    has_hallucination = any(c["verdict"] == "HALLUCINATION" for c in claims)
    status = "needs_review" if has_hallucination else "ok"

    # Attach covered/missed to claims payload for storage
    full_payload = {
        "claims": claims,
        "covered_themes": covered_themes,
        "missed_themes": result.get("missed_themes", []),
    }

    # Save or update report
    report, _ = VerificationReport.objects.update_or_create(
        reading=reading,
        defaults={
            "status": status,
            "claims": full_payload,
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
        },
    )

    return report