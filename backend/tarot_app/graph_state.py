"""
graph_state.py

TypedDict definition of the shared State for the LangGraph-based
reading-generation pipeline.

This State is read and written by every Node (router_node, rag_node,
prompt_build_node, generate_node, schema_validate_node, judge_node,
decide_next_node, ...) and is what Edges inspect to decide routing.

Design notes (see prior conversation for full reasoning):
  - schema_status and loop_status are kept SEPARATE because they come
    from two different validation layers (Pydantic schema validation
    vs. LLM-as-Judge content review) and a routing Edge needs to know
    which layer triggered a transition.
  - schema_retry_attempt and loop_iteration are kept SEPARATE counters
    for the same reason — they count two different retry loops.
  - error histories are lists, not single values, because Edges and
    feedback-builders need the full trail, not just the latest error.
  - cost/latency fields are intentionally NOT included here — they
    don't affect routing decisions and are tracked separately in
    LLMCallLog. State should only hold what Edges need to route on.
"""

from typing import TypedDict, Literal, Optional
from typing_extensions import NotRequired


# ---------------------------------------------------------------------------
# Sub-types for clarity
# ---------------------------------------------------------------------------

class CardObject(TypedDict):
    card_id: int
    card_name: str
    position: int
    position_label: str
    is_reversed: bool


class SchemaValidationError(TypedDict):
    field: str
    message: str
    invalid_value: NotRequired[object]


class JudgeClaim(TypedDict):
    card_name: str
    claim: str
    verdict: Literal["VERIFIED", "UNVERIFIED", "HALLUCINATION"]
    reason: str


SchemaStatus = Literal["pending", "ok", "recovered", "parse_failed"]
LoopStatus = Literal["pending", "success", "exhausted", "regression_stopped"]
StrategyName = Literal["simple", "complex"]


# ---------------------------------------------------------------------------
# The State
# ---------------------------------------------------------------------------

class ReadingState(TypedDict):
    # ---- Input layer (set once, unchanged for the whole run) ----
    reading_id: int
    user_name: str
    question: str
    spread_key: str
    spread_label: str
    card_objects: list[CardObject]

    # ---- Router decision output (set once by router_node) ----
    strategy_name: StrategyName
    prompt_version: str
    model: str
    max_retries: int
    max_loop_iterations: int
    rag_top_k: int
    requires_narrative_linking: bool

    # ---- Per-iteration mutable fields ----
    current_prompt: str
    schema_retry_attempt: int
    loop_iteration: int

    # ---- LLM output ----
    raw_llm_output: NotRequired[str]
    parsed_structured: NotRequired[Optional[dict]]

    # ---- Schema validation (Layer 1) ----
    schema_validation_ok: bool
    schema_validation_errors: list[SchemaValidationError]

    # ---- Judge review (Layer 2) ----
    judge_claims: list[JudgeClaim]
    judge_f1: NotRequired[Optional[float]]
    judge_precision: NotRequired[Optional[float]]
    judge_recall: NotRequired[Optional[float]]
    hallucination_count: int

    # ---- History (for feedback-building and Edge decisions) ----
    schema_error_history: list[list[SchemaValidationError]]   # one list per attempt
    hallucination_history: list[list[JudgeClaim]]              # one list per loop_iteration

    # ---- Status flags (two separate layers, see module docstring) ----
    schema_status: SchemaStatus
    loop_status: LoopStatus

    # ---- Final output ----
    reading_text: NotRequired[str]
    best_loop_iteration: NotRequired[int]


# ---------------------------------------------------------------------------
# Factory for the initial state
# ---------------------------------------------------------------------------

def build_initial_state(reading_id: int, user_name: str, question: str,
                        spread_key: str, spread_label: str,
                        card_objects: list[CardObject]) -> ReadingState:
    """
    Constructs the initial ReadingState before router_node runs.
    Strategy fields are placeholder until router_node fills them in.
    """
    return ReadingState(
        reading_id=reading_id,
        user_name=user_name,
        question=question,
        spread_key=spread_key,
        spread_label=spread_label,
        card_objects=card_objects,

        # Strategy fields — filled by router_node, placeholders for now
        strategy_name="simple",
        prompt_version="",
        model="",
        max_retries=0,
        max_loop_iterations=0,
        rag_top_k=0,
        requires_narrative_linking=False,

        current_prompt="",
        schema_retry_attempt=0,
        loop_iteration=0,

        schema_validation_ok=False,
        schema_validation_errors=[],

        judge_claims=[],
        hallucination_count=0,

        schema_error_history=[],
        hallucination_history=[],

        schema_status="pending",
        loop_status="pending",
    )