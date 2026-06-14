"""
verify.py  —  Layer 1 Verification (pure code, no AI)

Checks a generated reading against ground-truth themes from the database.
Uses a synonym dictionary to handle semantic equivalence without ML.

Outputs a VerificationReport with three verdict types per theme:
  MATCH        — theme (or a synonym) found in reading text
  MISMATCH     — theme not found at all (potential hallucination gap)
  NOT_IN_RECORD— card has no ground truth loaded yet (import_themes not run)

For reversed cards, also checks whether the reading incorrectly reads
the card in an upright (positive) direction instead of shadow direction.
"""

# ---------------------------------------------------------------------------
# Synonym dictionary
# Keys are canonical theme words from dariusk/corpora.
# Values are alternative phrasings the LLM might use instead.
# ---------------------------------------------------------------------------
SYNONYMS = {
    # Core emotional / relational
    "compassion":       ["compassion", "warmth", "care for others", "kindness", "caring"],
    "empathy":          ["empathy", "emotional intelligence", "understanding others", "attuned"],
    "insightfulness":   ["insightfulness", "intuition", "insight", "perceptive", "awareness"],
    "spirituality":     ["spirituality", "spirit", "soul", "divine", "higher power", "sacred"],
    "love":             ["love", "affection", "devotion", "romance", "loving"],
    "passion":          ["passion", "desire", "longing", "ardor"],
    "intuition":        ["intuition", "gut feeling", "inner knowing", "instinct"],
    "instinct":         ["instinct", "intuition", "gut", "inner sense"],

    # Power / agency
    "authority":        ["authority", "leadership", "control", "command", "power"],
    "discipline":       ["discipline", "self-control", "restraint", "focus"],
    "victory":          ["victory", "success", "triumph", "winning", "achievement"],
    "advancement":      ["advancement", "progress", "moving forward", "growth"],
    "capability":       ["capability", "ability", "skill", "competence", "talent"],
    "empowerment":      ["empowerment", "strength", "confidence", "agency"],

    # Change / transition
    "ending":           ["ending", "conclusion", "closure", "completion", "finish"],
    "transition":       ["transition", "change", "transformation", "shift", "passage"],
    "upheaval":         ["upheaval", "disruption", "turmoil", "chaos", "shake-up"],
    "demolition":       ["demolition", "destruction", "breakdown", "collapse", "tearing down"],
    "deconstruction":   ["deconstruction", "dismantling", "breaking down", "undoing"],
    "disaster":         ["disaster", "crisis", "catastrophe", "calamity", "shock"],
    "destruction":      ["destruction", "ruin", "collapse", "devastation", "demolition"],
    "revolution":       ["revolution", "radical change", "upheaval", "overhaul"],

    # Inner life
    "solitude":         ["solitude", "alone", "isolation", "withdrawal", "retreat"],
    "reflection":       ["reflection", "contemplation", "introspection", "meditation"],
    "enlightenment":    ["enlightenment", "awakening", "revelation", "clarity", "insight"],
    "sacrifice":        ["sacrifice", "letting go", "surrender", "giving up"],
    "perspective":      ["perspective", "viewpoint", "outlook", "new angle"],
    "suspension":       ["suspension", "pause", "waiting", "stillness", "limbo"],

    # Fortune / material
    "luck":             ["luck", "fortune", "chance", "fate", "serendipity"],
    "cycles":           ["cycles", "patterns", "rhythm", "recurring", "turning point"],
    "karma":            ["karma", "cause and effect", "what goes around", "consequence"],
    "wealth":           ["wealth", "abundance", "prosperity", "riches", "financial"],
    "health":           ["health", "wellbeing", "vitality", "physical", "wellness"],
    "practicality":     ["practicality", "practical", "grounded", "realistic", "sensible"],
    "receiving":        ["receiving", "accepting", "gift", "incoming", "receiving blessings"],

    # Guidance / knowledge
    "guidance":         ["guidance", "direction", "mentorship", "advice", "counsel"],
    "knowledge":        ["knowledge", "wisdom", "understanding", "learning", "insight"],
    "revelation":       ["revelation", "discovery", "uncovering", "realization", "truth"],
    "belief":           ["belief", "faith", "trust", "conviction", "spirituality"],
    "balance":          ["balance", "equilibrium", "harmony", "fairness", "equal"],
    "fairness":         ["fairness", "justice", "equity", "impartiality", "objectivity"],
    "law":              ["law", "rules", "order", "legal", "justice"],
    "objectivity":      ["objectivity", "neutral", "unbiased", "impartial", "fair"],

    # Character / virtue
    "fertility":        ["fertility", "abundance", "creativity", "nurturing", "growth"],
    "productivity":     ["productivity", "output", "results", "fruitful", "effective"],
    "nurturing":        ["nurturing", "caring", "supportive", "motherly", "nourishing"],
    "boldness":         ["boldness", "courage", "bravery", "daring", "confidence"],
    "vitality":         ["vitality", "energy", "vigor", "aliveness", "life force"],
    "experience":       ["experience", "wisdom", "knowledge", "seasoned", "background"],
    "stillness":        ["stillness", "calm", "quiet", "peace", "tranquility"],
    "withdrawal":       ["withdrawal", "retreat", "stepping back", "pulling away", "solitude"],
    "freedom":          ["freedom", "liberation", "independence", "free", "unrestricted"],
    "faith":            ["faith", "trust", "belief", "confidence", "hope"],
    "innocence":        ["innocence", "purity", "naivety", "fresh start", "openness"],

    # Shadow / negative (reversed themes)
    "wallowing":        ["wallowing", "self-pity", "dwelling", "stuck in emotion"],
    "manipulation":     ["manipulation", "deception", "controlling", "scheming"],
    "addiction":        ["addiction", "dependency", "compulsion", "indulgence"],
    "obsession":        ["obsession", "fixation", "preoccupation", "consumed by"],
    "rigidity":         ["rigidity", "inflexibility", "stubborn", "resistant to change"],
}

# Upright positive signal words — if these dominate a reversed-card reading,
# it suggests the LLM ignored the reversed orientation.
UPRIGHT_POSITIVE_SIGNALS = [
    "celebrate", "success", "thrive", "flourish", "abundance",
    "joy", "wonderful", "positive", "bright future", "exciting opportunity",
    "great news", "moving forward with confidence", "fully flowing",
]

REVERSED_SHADOW_SIGNALS = [
    "reversed", "shadow", "blocked", "inward", "turned inward",
    "not flowing", "working against", "challenge", "caution",
    "warning", "difficult", "struggle", "resist", "avoid",
]


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _theme_in_text(theme: str, text_lower: str) -> tuple[bool, str]:
    """
    Returns (matched: bool, matched_via: str).
    Checks the theme itself first, then its synonyms.
    """
    if theme.lower() in text_lower:
        return True, theme

    for synonym in SYNONYMS.get(theme.lower(), []):
        if synonym.lower() in text_lower:
            return True, synonym

    return False, ""


def _check_reversed_orientation(text_lower: str) -> dict:
    """
    Checks whether a reversed card's reading actually addresses
    the shadow/reversed dimension, or defaults to upright positivity.
    """
    positive_hits = [w for w in UPRIGHT_POSITIVE_SIGNALS if w in text_lower]
    shadow_hits = [w for w in REVERSED_SHADOW_SIGNALS if w in text_lower]

    if shadow_hits:
        orientation_verdict = "REVERSED_ACKNOWLEDGED"
    elif positive_hits:
        orientation_verdict = "REVERSED_IGNORED"
    else:
        orientation_verdict = "ORIENTATION_UNCLEAR"

    return {
        "orientation_verdict": orientation_verdict,
        "shadow_signals_found": shadow_hits,
        "positive_signals_found": positive_hits,
    }


# ---------------------------------------------------------------------------
# Main verifier
# ---------------------------------------------------------------------------

def verify_reading(reading) -> dict:
    """
    reading: a Reading model instance.

    Returns a full VerificationReport dict.
    """
    text = reading.reading_text
    text_lower = text.lower()
    card_reports = []

    for rc in reading.readingcard_set.select_related('card').all():
        card = rc.card

        if rc.is_reversed:
            themes = card.reversed_required_themes or []
        else:
            themes = card.required_themes or []

        if not themes:
            card_reports.append({
                "card_name": card.name,
                "position_label": rc.position_label,
                "is_reversed": rc.is_reversed,
                "verdict": "NOT_IN_RECORD",
                "reason": "No ground truth loaded. Run import_themes first.",
                "theme_results": [],
                "orientation_check": None,
                "summary": {"total": 0, "matched": 0, "mismatched": 0},
            })
            continue

        theme_results = []
        for theme in themes:
            matched, matched_via = _theme_in_text(theme, text_lower)
            theme_results.append({
                "theme": theme,
                "verdict": "MATCH" if matched else "MISMATCH",
                "matched_via": matched_via if matched else None,
            })

        matched_count = sum(1 for t in theme_results if t["verdict"] == "MATCH")
        total = len(theme_results)
        recall = matched_count / total if total > 0 else 0.0

        # Overall card verdict
        if recall == 1.0:
            card_verdict = "MATCH"
        elif recall == 0.0:
            card_verdict = "MISMATCH"
        else:
            card_verdict = "PARTIAL_MATCH"

        # Extra check for reversed cards
        orientation_check = None
        if rc.is_reversed:
            orientation_check = _check_reversed_orientation(text_lower)
            # Downgrade to MISMATCH if reversed orientation was completely ignored
            if (card_verdict in ("MATCH", "PARTIAL_MATCH") and
                    orientation_check["orientation_verdict"] == "REVERSED_IGNORED"):
                card_verdict = "MISMATCH"

        card_reports.append({
            "card_name": card.name,
            "position_label": rc.position_label,
            "is_reversed": rc.is_reversed,
            "verdict": card_verdict,
            "theme_results": theme_results,
            "orientation_check": orientation_check,
            "summary": {
                "total": total,
                "matched": matched_count,
                "mismatched": total - matched_count,
                "recall": round(recall, 3),
            },
        })

    # Overall reading verdict
    verdicts = [r["verdict"] for r in card_reports]
    if all(v == "MATCH" for v in verdicts):
        overall = "MATCH"
    elif all(v == "MISMATCH" for v in verdicts):
        overall = "MISMATCH"
    elif any(v == "NOT_IN_RECORD" for v in verdicts):
        overall = "NOT_IN_RECORD"
    else:
        overall = "PARTIAL_MATCH"

    return {
        "reading_id": reading.id,
        "user_name": reading.user_name,
        "question": reading.question,
        "spread_type": reading.spread_type,
        "overall_verdict": overall,
        "cards": card_reports,
    }