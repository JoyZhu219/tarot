"""
router.py

A minimal Router for the reading-generation pipeline.

assess_complexity() looks at the input (question + spread type + card count)
and decides a Strategy: which prompt version to use, which model, retry
budget, and RAG depth.

So the deciding factor is spread_key membership in
SPREADS_REQUIRING_NARRATIVE_LINKING, not card_count alone. card_count is
still recorded for logging/auditing, but it does not drive the decision
by itself.

Pure Python. No framework dependency — callable from views.py directly.
"""

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Strategy definitions
# ---------------------------------------------------------------------------

@dataclass
class Strategy:
    name: str                  # "simple" | "complex"
    prompt_version: str        # which prompt template to use
    model: str                 # which Claude model to call
    max_retries: int           # retry budget
    rag_top_k: int              # how many RAG chunks per card
    requires_narrative_linking: bool = False  # used by _build_prompt to append instruction
    reasoning: str = ""        # human-readable rationale, filled by assess_complexity


SIMPLE_STRATEGY = Strategy(
    name="simple",
    prompt_version="v4",
    model="claude-sonnet-4-6",
    max_retries=2,
    rag_top_k=2,
    requires_narrative_linking=False,
)

COMPLEX_STRATEGY = Strategy(
    name="complex",
    prompt_version="v4",
    model="claude-sonnet-4-6",
    max_retries=3,       # more cards/positions = more chances for one to fail validation
    rag_top_k=3,         # richer reference material to reduce hallucination risk
    requires_narrative_linking=True,
)


# ---------------------------------------------------------------------------
# Complexity assessment
# ---------------------------------------------------------------------------

# Spreads where positions are meant to relate to each other narratively —
# the reading must connect them, not treat each card in isolation.
# "single" is the only spread that is genuinely simple: one card, no linking.
SPREADS_REQUIRING_NARRATIVE_LINKING = {
    "past_present_future",
    "celtic_cross",
    "relationship",
    "career",
}


def assess_complexity(question: str, card_count: int, spread_key: str = None) -> Strategy:
    """
    Decide a Strategy based on input complexity.

    Args:
        question:   the querent's question text (logging context only)
        card_count: number of cards in the spread (recorded, not the sole driver)
        spread_key: e.g. "single", "past_present_future", "celtic_cross".
                    This is the actual deciding factor.

    Returns:
        Strategy object with prompt_version, model, max_retries, rag_top_k,
        requires_narrative_linking, and a `reasoning` string.

    Rule:
        spread_key in SPREADS_REQUIRING_NARRATIVE_LINKING -> complex
        otherwise (e.g. "single")                          -> simple

    Note: card_count alone is NOT used to decide, because
    Past-Present-Future has only 3 cards but requires the same kind of
    cross-position narrative linking as a 10-card Celtic Cross.
    """
    question_preview = (question or "").strip()[:80]
    requires_linking = spread_key in SPREADS_REQUIRING_NARRATIVE_LINKING

    if not requires_linking:
        strategy = Strategy(**{**SIMPLE_STRATEGY.__dict__})
        strategy.reasoning = (
            f"spread_key='{spread_key}' (card_count={card_count}) is not in "
            f"SPREADS_REQUIRING_NARRATIVE_LINKING -> simple. "
            f"This spread does not require connecting multiple positions into "
            f"one narrative, so default retry budget and RAG depth are sufficient. "
            f"Question preview: '{question_preview}'"
        )
    else:
        strategy = Strategy(**{**COMPLEX_STRATEGY.__dict__})
        strategy.reasoning = (
            f"spread_key='{spread_key}' (card_count={card_count}) IS in "
            f"SPREADS_REQUIRING_NARRATIVE_LINKING -> complex. "
            f"This spread requires positions to relate to each other "
            f"(e.g. past explaining present, challenge informing advice), "
            f"which is harder than independent card-by-card interpretation. "
            f"Retry budget raised to {strategy.max_retries}, rag_top_k raised to "
            f"{strategy.rag_top_k} for richer per-card grounding. "
            f"Note: card_count alone was NOT used — a 3-card Past-Present-Future "
            f"spread is classified complex despite having few cards, because it "
            f"structurally requires the same narrative linking as Celtic Cross. "
            f"Question preview: '{question_preview}'"
        )

    return strategy


# ---------------------------------------------------------------------------
# Decision log builder
# ---------------------------------------------------------------------------

def build_router_decision_log(question: str, card_count: int, spread_key: str,
                              strategy: Strategy, sequence_number: int = 1) -> dict:
    """
    Builds the fields needed to write an AgentDecisionLog entry for this
    routing decision. Caller attaches `reading` and saves.

    Usage:
        strategy = assess_complexity(question, len(card_objects), spread_key)
        log_fields = build_router_decision_log(question, len(card_objects), spread_key, strategy)
        AgentDecisionLog.objects.create(reading=reading, **log_fields)
    """
    return {
        "decision_point": "prompt_build",
        "sequence_number": sequence_number,
        "input_data": {
            "question_preview": (question or "").strip()[:80],
            "card_count": card_count,
            "spread_key": spread_key,
            "spreads_requiring_linking": sorted(SPREADS_REQUIRING_NARRATIVE_LINKING),
        },
        "decision": f"strategy={strategy.name} (prompt={strategy.prompt_version}, "
                    f"model={strategy.model}, max_retries={strategy.max_retries}, "
                    f"rag_top_k={strategy.rag_top_k}, "
                    f"requires_narrative_linking={strategy.requires_narrative_linking})",
        "rationale": strategy.reasoning,
        "output_data": {
            "prompt_version": strategy.prompt_version,
            "model": strategy.model,
            "max_retries": strategy.max_retries,
            "rag_top_k": strategy.rag_top_k,
            "requires_narrative_linking": strategy.requires_narrative_linking,
        },
        "latency_ms": 0,
        "input_tokens": None,
        "output_tokens": None,
        "cost_usd": 0.0,
        "outcome": "success",
    }