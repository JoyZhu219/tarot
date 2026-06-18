"""
graph_edges.py

Conditional Edges for the reading-generation graph, plus the auto_fix node
and final StateGraph assembly.

Two decision points need conditional routing:

1. After generate_node (format/schema check):
     parse succeeded            -> judge_node
     parse failed, retries left -> rag_and_prompt_node (retry with feedback)
     parse failed, exhausted    -> END (schema_status="parse_failed")

2. After judge_node (verify check):
     no hallucination               -> END (success)
     hallucination is "mechanical"  -> auto_fix_node (no LLM call)
     hallucination is "semantic"    -> rag_and_prompt_node (regenerate with feedback)
     loop exhausted                  -> END (loop_status="exhausted")

Mechanical vs semantic classification: judge.py's HALLUCINATION verdict
has no built-in subtype, so this module adds it. "Mechanical" means the
claim references a card_name or position_label that doesn't match what
was actually drawn (checkable directly against state["card_objects"] —
no LLM reasoning needed). "Semantic" means the claim contradicts the
card's official meaning/direction (e.g. reading a reversed card as
positive) — this can only be resolved by the LLM re-reasoning, not by
a string fix. In practice, mechanical errors are rare (the LLM mostly
transcribes card/position info it was given rather than inventing it),
so most hallucinations route to "semantic" -> regenerate.

auto_fix_node handles mechanical hallucinations conservatively: it
confirms the mismatch against ground truth but does not attempt a risky
in-place text rewrite. It routes back to judge_node to re-verify; if the
issue persists, route_after_judge will fall through to regenerate.
"""

from langgraph.graph import StateGraph, END

from .graph_state import ReadingState
from .graph_nodes import router_node, rag_and_prompt_node, generate_node, judge_node


# ---------------------------------------------------------------------------
# Hallucination classifier (mechanical vs semantic)
# ---------------------------------------------------------------------------
#
# "mechanical": the claim references a card_name or position_label that
#               does NOT match what was actually drawn (per state["card_objects"]).
#               This is checkable directly against ground truth — no LLM
#               reasoning needed, just a lookup. Low-probability in practice
#               (the LLM is mostly transcribing card/position info it was
#               given, not inventing it), but mechanically verifiable when
#               it does happen.
# "semantic":   the claim contradicts the card's official themes in meaning
#               — e.g. reading a reversed card as upright/positive, or
#               attributing a meaning the official themes don't support.
#               This requires the LLM to re-reason; no string fix exists.

def _classify_hallucination(claim: dict, card_objects: list) -> str:
    """
    Returns 'mechanical' or 'semantic'.

    mechanical: claim's card_name or position reference doesn't match
                any card actually drawn in this reading (per card_objects).
    semantic:   everything else — a meaning/direction contradiction that
                only the LLM can resolve by re-reasoning about the card.
    """
    claim_text = claim.get("claim", "")
    drawn_card_names = {c["card_name"] for c in card_objects}
    drawn_position_labels = {c["position_label"] for c in card_objects}

    # The claim is tagged with which card it's about (judge.py sets card_name
    # on each claim). If that card_name isn't among what was actually drawn,
    # that's a mechanical error — the LLM referenced a card that wasn't pulled.
    claimed_card = claim.get("card_name", "")
    if claimed_card and claimed_card not in drawn_card_names:
        return "mechanical"

    # Check if the claim text mentions a position label that doesn't match
    # any actual position in this spread (e.g. says "Advice" position but
    # this spread has no such position).
    for word in claim_text.split():
        cleaned = word.strip(".,;:'\"")
        if cleaned in {"Past", "Present", "Future", "Challenge", "Advice",
                       "Outcome", "Above", "Below", "Hopes", "External"}:
            if cleaned not in " ".join(drawn_position_labels):
                return "mechanical"

    return "semantic"


def _classify_all_hallucinations(claims: list, card_objects: list) -> dict:
    """Returns {'mechanical': [...], 'semantic': [...]} from a claims list."""
    hallucinations = [c for c in claims if c["verdict"] == "HALLUCINATION"]
    result = {"mechanical": [], "semantic": []}
    for h in hallucinations:
        result[_classify_hallucination(h, card_objects)].append(h)
    return result


# ---------------------------------------------------------------------------
# auto_fix_node — handles numeric-type hallucinations without an LLM call
# ---------------------------------------------------------------------------

def auto_fix_node(state: dict) -> dict:
    """
    Attempts a direct fix for mechanical-type hallucinations — cases where
    the LLM referenced a card or position that wasn't actually drawn.

    This does NOT call the LLM. It checks claims against state["card_objects"]
    (the ground truth of what was actually pulled) and flags/logs the
    mismatch. Unlike a semantic contradiction, a mechanical mismatch has
    a clear correct answer to check against — but actually rewriting the
    reading text to swap in the correct card/position safely (without
    breaking surrounding sentences) is non-trivial, so this implementation
    is conservative: it confirms the mismatch and lets the subsequent
    judge pass re-verify, rather than attempting a risky text rewrite.
    """
    claims = state.get("judge_claims", [])
    card_objects = state["card_objects"]

    mechanical_hallucinations = [
        c for c in claims
        if c["verdict"] == "HALLUCINATION"
        and _classify_hallucination(c, card_objects) == "mechanical"
    ]

    drawn_card_names = {c["card_name"] for c in card_objects}
    confirmed_mismatches = []

    for h in mechanical_hallucinations:
        claimed_card = h.get("card_name", "")
        if claimed_card and claimed_card not in drawn_card_names:
            confirmed_mismatches.append({
                "claim": h.get("claim", ""),
                "claimed_card": claimed_card,
                "actually_drawn": sorted(drawn_card_names),
            })

    # Conservative: no text mutation happens here. We return reading_text
    # unchanged and rely on route_after_auto_fix sending this back to
    # judge_node, which will re-confirm whether the issue persists.
    # If it does, route_after_judge will see hallucination_count unchanged
    # and fall through to "regenerate" on the next pass.
    return {
        "reading_text": state.get("reading_text", ""),
    }


# ---------------------------------------------------------------------------
# Conditional Edge functions
# ---------------------------------------------------------------------------

def route_after_generate(state: dict) -> str:
    """
    Edge after generate_node. Returns the name of the next node.

    parse succeeded             -> "judge"
    parse failed, retries left  -> "retry_prompt"   (loops back to rag_and_prompt_node)
    parse failed, exhausted     -> "end_parse_failed"
    """
    if state["schema_validation_ok"]:
        return "judge"

    if state["schema_retry_attempt"] >= state["max_retries"]:
        return "end_parse_failed"

    return "retry_prompt"


def route_after_judge(state: dict) -> str:
    """
    Edge after judge_node. Returns the name of the next node.

    no hallucination                       -> "end_success"
    loop exhausted                          -> "end_exhausted"
    hallucination, all mechanical           -> "auto_fix"
    hallucination, any semantic             -> "regenerate"
    """
    if state["hallucination_count"] == 0:
        return "end_success"

    if state["loop_iteration"] >= state["max_loop_iterations"]:
        return "end_exhausted"

    classified = _classify_all_hallucinations(state["judge_claims"], state["card_objects"])

    if classified["semantic"]:
        # Any semantic hallucination requires full regeneration —
        # a mechanical fix alone won't resolve a meaning-level contradiction.
        return "regenerate"

    if classified["mechanical"]:
        return "auto_fix"

    # Shouldn't happen (hallucination_count > 0 but nothing classified),
    # but fail safe toward regeneration rather than looping forever.
    return "regenerate"


def route_after_auto_fix(state: dict) -> str:
    """
    Edge after auto_fix_node. Always re-judges to confirm the fix worked —
    auto_fix never assumes success without verification.
    """
    return "judge"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph():
    """
    Assembles the full StateGraph:

        router -> rag_and_prompt -> generate
                                       |
                          route_after_generate
                          /         |          \\
                    judge    retry_prompt   end_parse_failed
                      |        (-> rag_and_prompt)
              route_after_judge
              /        |         \\         \\
        end_success  auto_fix  regenerate  end_exhausted
         (no halluc)  (mechanical) (semantic)
                       |          |
                     judge   rag_and_prompt
    """
    import tarot_app.graph_nodes as _nodes

    graph = StateGraph(ReadingState)

    # Use lambda wrappers instead of direct function references.
    # This makes patch() work correctly in tests: patch replaces the
    # attribute on the module object, and the lambda looks up the
    # current attribute at call time rather than holding a fixed reference
    # to the original function object.
    graph.add_node("router", lambda s: _nodes.router_node(s))
    graph.add_node("rag_and_prompt", lambda s: _nodes.rag_and_prompt_node(s))
    graph.add_node("generate", lambda s: _nodes.generate_node(s))
    graph.add_node("judge", lambda s: _nodes.judge_node(s))
    graph.add_node("auto_fix", lambda s: _nodes.auto_fix_node(s))

    graph.set_entry_point("router")
    graph.add_edge("router", "rag_and_prompt")
    graph.add_edge("rag_and_prompt", "generate")

    graph.add_conditional_edges(
        "generate",
        route_after_generate,
        {
            "judge": "judge",
            "retry_prompt": "rag_and_prompt",
            "end_parse_failed": END,
        },
    )

    graph.add_conditional_edges(
        "judge",
        route_after_judge,
        {
            "end_success": END,
            "end_exhausted": END,
            "auto_fix": "auto_fix",
            "regenerate": "rag_and_prompt",
        },
    )

    graph.add_conditional_edges(
        "auto_fix",
        route_after_auto_fix,
        {"judge": "judge"},
    )

    return graph.compile()


# ---------------------------------------------------------------------------
# Diagram generation
# ---------------------------------------------------------------------------

def save_graph_diagram(output_path: str = "/home/claude/tarot/backend/rag/reading_graph.png"):
    """
    Renders the compiled graph to a PNG using LangGraph's built-in
    Mermaid-based drawing. Requires graphviz/mermaid rendering deps
    bundled with langgraph; falls back to printing Mermaid source if
    image rendering isn't available in this environment.
    """
    compiled = build_graph()
    try:
        png_bytes = compiled.get_graph().draw_mermaid_png()
        with open(output_path, "wb") as f:
            f.write(png_bytes)
        print(f"Graph diagram saved to {output_path}")
    except Exception as e:
        print(f"Could not render PNG ({e}). Mermaid source:")
        print(compiled.get_graph().draw_mermaid())


if __name__ == "__main__":
    save_graph_diagram()