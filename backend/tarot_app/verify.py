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

import re

# ---------------------------------------------------------------------------
# Synonym dictionary
# Keys are canonical theme words from dariusk/corpora.
# Values are alternative phrasings the LLM might use instead.
# ---------------------------------------------------------------------------
SYNONYMS = {
    # Core emotional / relational
    "compassion":       ["compassion", "warmth", "care for others", "kindness",
                         "caring", "tenderness", "gentle", "heart"],
    "empathy":          ["empathy", "emotional intelligence", "understanding others",
                         "attuned", "feel for", "emotionally aware", "in tune with"],
    "insightfulness":   ["insightfulness", "intuition", "insight", "perceptive",
                         "awareness", "discerning", "keen sense", "perceive"],
    "spirituality":     ["spirituality", "spirit", "soul", "divine", "higher power",
                         "sacred", "mystical", "transcendent", "universe", "cosmic"],
    "love":             ["love", "affection", "devotion", "romance", "loving",
                         "adoration", "cherish", "beloved"],
    "passion":          ["passion", "desire", "longing", "ardor", "fervor",
                         "enthusiasm", "zeal", "drive"],
    "intuition":        ["intuition", "gut feeling", "inner knowing", "instinct",
                         "inner voice", "sixth sense", "inner wisdom"],
    "instinct":         ["instinct", "intuition", "gut", "inner sense",
                         "natural knowing", "innate sense"],

    # Power / agency
    "authority":        ["authority", "leadership", "control", "command", "power",
                         "in charge", "dominion", "rule"],
    "discipline":       ["discipline", "self-control", "restraint", "focus",
                         "rigor", "structure", "commitment"],
    "victory":          ["victory", "success", "triumph", "winning", "achievement",
                         "accomplishment", "prevail", "overcome"],
    "advancement":      ["advancement", "progress", "moving forward", "growth",
                         "forward momentum", "step ahead", "leveling up"],
    "capability":       ["capability", "ability", "skill", "competence", "talent",
                         "gifted", "skilled", "proficient"],
    "empowerment":      ["empowerment", "strength", "confidence", "agency",
                         "self-assurance", "inner power", "reclaim"],

    # Change / transition
    "ending":           ["ending", "conclusion", "closure", "completion", "finish",
                         "close", "finality", "wrapping up"],
    "transition":       ["transition", "change", "transformation", "shift", "passage",
                         "crossroads", "moving on", "new chapter"],
    "upheaval":         ["upheaval", "disruption", "turmoil", "chaos", "shake-up",
                         "unrest", "instability", "uprooted"],
    "demolition":       ["demolition", "destruction", "breakdown", "collapse",
                         "tearing down", "torn apart", "dismantled"],
    "deconstruction":   ["deconstruction", "dismantling", "breaking down", "undoing",
                         "unraveling", "taking apart"],
    "disaster":         ["disaster", "crisis", "catastrophe", "calamity", "shock",
                         "blow", "devastating", "wreckage"],
    "destruction":      ["destruction", "ruin", "collapse", "devastation",
                         "demolished", "shattered", "torn down"],
    "revolution":       ["revolution", "radical change", "upheaval", "overhaul",
                         "complete shift", "total transformation"],

    # Inner life
    "solitude":         ["solitude", "alone", "isolation", "withdrawal", "retreat",
                         "by yourself", "inner quiet", "stepping away"],
    "reflection":       ["reflection", "contemplation", "introspection", "meditation",
                         "inner look", "self-examination", "pondering"],
    "enlightenment":    ["enlightenment", "awakening", "revelation", "clarity",
                         "insight", "illumination", "realization", "epiphany"],
    "sacrifice":        ["sacrifice", "letting go", "surrender", "giving up",
                         "release", "relinquish", "lay down"],
    "perspective":      ["perspective", "viewpoint", "outlook", "new angle",
                         "point of view", "fresh eyes", "see differently"],
    "suspension":       ["suspension", "pause", "waiting", "stillness", "limbo",
                         "on hold", "in between", "suspended"],

    # Fortune / material
    "luck":             ["luck", "fortune", "chance", "fate", "serendipity",
                         "fortunate", "blessed", "windfall"],
    "cycles":           ["cycles", "patterns", "rhythm", "recurring", "turning point",
                         "wheel", "going around", "repeating"],
    "karma":            ["karma", "cause and effect", "what goes around", "consequence",
                         "reap what you sow", "universal law"],
    "wealth":           ["wealth", "abundance", "prosperity", "riches", "financial",
                         "affluence", "well-off", "materially"],
    "health":           ["health", "wellbeing", "vitality", "physical", "wellness",
                         "body", "healing", "medical"],
    "practicality":     ["practicality", "practical", "grounded", "realistic",
                         "sensible", "down to earth", "no-nonsense", "pragmatic"],
    "receiving":        ["receiving", "accepting", "gift", "incoming",
                         "receiving blessings", "open to receive", "welcomed"],

    # Guidance / knowledge
    "guidance":         ["guidance", "direction", "mentorship", "advice", "counsel",
                         "guide", "steer", "mentor"],
    "knowledge":        ["knowledge", "wisdom", "understanding", "learning", "insight",
                         "know", "grasp", "comprehend"],
    "revelation":       ["revelation", "discovery", "uncovering", "realization",
                         "truth", "unveiled", "coming to light"],
    "belief":           ["belief", "faith", "trust", "conviction", "spirituality",
                         "values", "principles", "creed"],
    "balance":          ["balance", "equilibrium", "harmony", "fairness", "equal",
                         "even", "centered", "stable"],
    "fairness":         ["fairness", "justice", "equity", "impartiality",
                         "objectivity", "fair", "just", "unbiased"],
    "law":              ["law", "rules", "order", "legal", "justice",
                         "regulation", "court", "binding"],
    "objectivity":      ["objectivity", "neutral", "unbiased", "impartial", "fair",
                         "detached", "balanced view"],

    # Character / virtue
    "fertility":        ["fertility", "abundance", "creativity", "nurturing", "growth",
                         "fruitful", "generative", "bloom"],
    "productivity":     ["productivity", "output", "results", "fruitful", "effective",
                         "efficient", "get things done"],
    "nurturing":        ["nurturing", "caring", "supportive", "motherly", "nourishing",
                         "foster", "cultivate", "tend to"],
    "boldness":         ["boldness", "courage", "bravery", "daring", "confidence",
                         "audacious", "fearless", "gutsy"],
    "vitality":         ["vitality", "energy", "vigor", "aliveness", "life force",
                         "vibrant", "alive", "lively"],
    "experience":       ["experience", "wisdom", "knowledge", "seasoned", "background",
                         "veteran", "track record", "history"],
    "stillness":        ["stillness", "calm", "quiet", "peace", "tranquility",
                         "serenity", "hush", "at rest"],
    "withdrawal":       ["withdrawal", "retreat", "stepping back", "pulling away",
                         "solitude", "recede", "pull back"],
    "freedom":          ["freedom", "liberation", "independence", "free",
                         "unrestricted", "unbound", "autonomous"],
    "faith":            ["faith", "trust", "belief", "confidence", "hope",
                         "conviction", "assurance", "rely on"],
    "innocence":        ["innocence", "purity", "naivety", "fresh start", "openness",
                         "untainted", "beginner", "new to"],

    # Shadow / negative (reversed themes)
    "wallowing":        ["wallowing", "self-pity", "dwelling", "stuck in emotion",
                         "ruminating", "can't move on", "mired in"],
    "manipulation":     ["manipulation", "deception", "controlling", "scheming",
                         "underhanded", "pull strings", "covert"],
    "addiction":        ["addiction", "dependency", "compulsion", "indulgence",
                         "can't stop", "hooked", "reliance"],
    "obsession":        ["obsession", "fixation", "preoccupation", "consumed by",
                         "can't let go", "fixated", "all-consuming"],
    "rigidity":         ["rigidity", "inflexibility", "stubborn", "resistant to change",
                         "closed off", "set in ways", "unyielding"],
    "self-doubt":       ["self-doubt", "insecurity", "not good enough", "questioning yourself",
                         "uncertain about yourself", "lack confidence"],
    "avoidance":        ["avoidance", "running away", "escape", "denial",
                         "refusing to face", "turning away", "ignore"],
    "overwhelm":        ["overwhelm", "too much", "flooded", "swamped",
                         "drowned in", "overcome by emotion"],
}

# ---------------------------------------------------------------------------
# Orientation signal words
# IMPORTANT: use word-boundary matching to avoid false positives like
# "joy" matching a person's name, or "positive" matching "repositioned".
# ---------------------------------------------------------------------------

# Signals that suggest LLM treated a reversed card as upright/positive
UPRIGHT_POSITIVE_SIGNALS = [
    "celebrate", "flourish", "wonderful", "bright future",
    "exciting opportunity", "great news", "moving forward with confidence",
    "fully flowing", "everything is working", "things are going well",
    "open your heart", "embrace the abundance",
]

# Signals that confirm LLM acknowledged the reversed dimension
REVERSED_SHADOW_SIGNALS = [
    "reversed", "shadow", "blocked", "turned inward",
    "not flowing", "working against", "inner fog", "self-doubt",
    "inner critic", "holding back", "emotional block",
    "difficult", "struggle", "resist", "avoid", "caution",
    "warning", "challenge ahead", "draining",
]


def _word_boundary_match(phrase: str, text_lower: str) -> bool:
    """
    Match a phrase in text using word boundaries to avoid substring false positives.
    e.g. "joy" should not match inside "enjoy" or a person's name "Joy".
    For multi-word phrases, simple substring is fine.
    """
    phrase_lower = phrase.lower()
    if " " in phrase_lower:
        # multi-word phrase: substring match is specific enough
        return phrase_lower in text_lower
    else:
        # single word: require word boundary
        pattern = r'\b' + re.escape(phrase_lower) + r'\b'
        return bool(re.search(pattern, text_lower))


# ---------------------------------------------------------------------------
# Core matching
# ---------------------------------------------------------------------------

def _theme_in_text(theme: str, text_lower: str) -> tuple[bool, str]:
    """
    Returns (matched: bool, matched_via: str).
    Checks the theme itself first, then its synonyms.
    Long corpora phrases (>3 words) use substring match directly.
    """
    theme_lower = theme.lower()
    words = theme_lower.split()

    if len(words) > 3:
        # Long descriptive phrase from corpora — substring match
        if theme_lower in text_lower:
            return True, theme
    else:
        if _word_boundary_match(theme_lower, text_lower):
            return True, theme

    for synonym in SYNONYMS.get(theme_lower, []):
        if _word_boundary_match(synonym, text_lower):
            return True, synonym

    return False, ""


def _check_reversed_orientation(text_lower: str) -> dict:
    """
    Checks whether a reversed card's reading actually addresses
    the shadow/reversed dimension, or defaults to upright positivity.
    Uses word-boundary matching to avoid false positives on proper nouns.
    """
    positive_hits = [
        w for w in UPRIGHT_POSITIVE_SIGNALS
        if _word_boundary_match(w, text_lower)
    ]
    shadow_hits = [
        w for w in REVERSED_SHADOW_SIGNALS
        if _word_boundary_match(w, text_lower)
    ]

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

    # Strip user_name from text before matching to avoid false positives
    # e.g. if user is named "Joy", it should not trigger positive signal detection
    name = reading.user_name.strip()
    if name:
        import re as _re
        text = _re.sub(_re.escape(name), '', text, flags=_re.IGNORECASE)

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

        if recall == 1.0:
            card_verdict = "MATCH"
        elif recall == 0.0:
            card_verdict = "MISMATCH"
        else:
            card_verdict = "PARTIAL_MATCH"

        orientation_check = None
        if rc.is_reversed:
            orientation_check = _check_reversed_orientation(text_lower)
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