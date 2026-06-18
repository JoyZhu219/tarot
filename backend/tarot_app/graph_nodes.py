"""
graph_nodes.py

LangGraph node functions. Each node:
  - receives the full ReadingState
  - calls exactly one existing service (router / RAG+prompt / loop / judge)
  - returns ONLY the fields it changed (LangGraph merges this into State)

This file does not change any existing service logic in router.py,
evaluator_generator_loop.py, or judge.py — it wraps them.

Nodes:
  router_node            -> wraps assess_complexity()
  rag_and_prompt_node     -> wraps _fetch_rag_context() + _build_cards_block()
                              + prompt_manager.render() (the prompt-building
                              logic currently inside orchestrator._build_prompt)
  generate_node           -> wraps one generation attempt via _call_llm_with_retry
                              (the LLM-call part of generate_with_retry_loop)
  judge_node              -> wraps run_judge()

Note on generate_with_retry_loop: in the orchestrator version, this
function internally loops AND calls judge. In the graph version, looping
becomes the graph's job (via Edges), so generate_node here does ONE
generation attempt, and judge_node does ONE judge pass. The Edge between
them decides whether to loop back.
"""

import time
import anthropic

from .judge import run_judge


def router_node(state: dict) -> dict:
    """
    Wraps assess_complexity(). Decides strategy based on spread complexity.
    Always the first node in the graph.
    """
    from .router import assess_complexity

    strategy = assess_complexity(
        state["question"],
        card_count=len(state["card_objects"]),
        spread_key=state["spread_key"],
    )

    return {
        "strategy_name": strategy.name,
        "prompt_version": strategy.prompt_version,
        "model": strategy.model,
        "max_retries": strategy.max_retries,
        "max_loop_iterations": 2,  # mirrors orchestrator's hardcoded default
        "rag_top_k": strategy.rag_top_k,
        "requires_narrative_linking": strategy.requires_narrative_linking,
    }


def rag_and_prompt_node(state: dict) -> dict:
    """
    Wraps RAG retrieval + prompt construction.
    Re-uses _fetch_rag_context and _build_cards_block from views.py —
    same logic as orchestrator._retrieve_rag_context + orchestrator._build_prompt,
    just expressed as a node that reads/writes State instead of self.attributes.

    If schema_error_history or hallucination_history already have entries
    (i.e. this is a retry/regeneration pass), their latest entries are
    appended to the prompt as feedback — this is where "bring previous
    errors so the LLM doesn't repeat them" lives.
    """
    from .views import _fetch_rag_context, _build_cards_block
    from prompts.prompt_manager import prompt_manager

    # Reconstruct card_objects with actual Card model instances.
    # State stores plain dicts (JSON-serializable); we need live Card
    # objects to call _fetch_rag_context / _build_cards_block.
    from .models import Card
    card_objects = []
    for c in state["card_objects"]:
        card_objects.append({
            "card": Card.objects.get(id=c["card_id"]),
            "position": c["position"],
            "position_label": c["position_label"],
            "is_reversed": c["is_reversed"],
        })

    rag_context = _fetch_rag_context(card_objects, top_k=state["rag_top_k"])
    cards_block = _build_cards_block(card_objects, rag_context=rag_context)

    if state["requires_narrative_linking"]:
        cards_block += (
            "\n\nNOTE: This spread requires connecting positions into one narrative. "
            "When writing the 'overall' section, explicitly relate at least two "
            "positions to each other (e.g. how 'Past' explains 'Present', or how "
            "'Challenge' informs 'Advice'). Do not treat each card as fully independent."
        )

    prompt = prompt_manager.render(
        "reading_generation",
        user_name=state["user_name"],
        question=state["question"],
        spread_label=state["spread_label"],
        cards_block=cards_block,
    )

    # Append feedback from previous attempts, if any exist.
    # This is the single place feedback is assembled — both schema errors
    # and hallucination history are read from State, not passed as separate
    # function arguments (this is what solves "pain point 3" from earlier).
    feedback_parts = []

    if state.get("schema_error_history"):
        last_schema_errors = state["schema_error_history"][-1]
        if last_schema_errors:
            error_lines = "\n".join(
                f"  - field '{e['field']}': {e['message']}" for e in last_schema_errors
            )
            feedback_parts.append(
                f"Your previous output had JSON structure errors:\n{error_lines}\n"
                f"Fix these and call submit_reading again."
            )

    if state.get("hallucination_history"):
        last_hallucinations = state["hallucination_history"][-1]
        if last_hallucinations:
            hall_lines = "\n".join(
                f"  - \"{h['claim']}\" — {h['reason']}" for h in last_hallucinations
            )
            feedback_parts.append(
                f"Your previous reading had these HALLUCINATIONS (contradicted "
                f"official card meanings):\n{hall_lines}\n"
                f"Do NOT repeat these claims. Rewrite the reading."
            )

    if feedback_parts:
        prompt += "\n\n== PREVIOUS ATTEMPT FEEDBACK ==\n" + "\n\n".join(feedback_parts)

    return {"current_prompt": prompt}


def generate_node(state: dict) -> dict:
    """
    Wraps a single LLM generation + schema validation attempt.
    This is the inner loop body of _call_llm_with_retry, but expressed
    as ONE node call — the graph's Edges handle the retry looping,
    not a Python for-loop inside this function.

    Increments schema_retry_attempt and appends to schema_error_history,
    so router/Edge logic downstream can see the full trail.
    """
    import json as _json
    from django.conf import settings
    from .views import READING_TOOL, _parse_and_validate_llm_output
    from .models import Reading, LLMCallLog

    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)

    start = time.monotonic()
    message = client.messages.create(
        model=state["model"],
        max_tokens=2048,
        tools=[READING_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": state["current_prompt"]}],
    )
    latency_ms = int((time.monotonic() - start) * 1000)

    raw = None
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_reading":
            raw = _json.dumps(block.input)
            break
    if raw is None:
        for block in message.content:
            if hasattr(block, "text"):
                raw = block.text
                break
        raw = raw or "{}"

    parsed = _parse_and_validate_llm_output(raw)
    validation_ok = parsed["validation"]["ok"]
    validation_errors = parsed["validation"].get("errors", [])

    new_attempt = state["schema_retry_attempt"] + 1

    # Log this attempt — same LLMCallLog table the orchestrator version used
    reading = Reading.objects.get(id=state["reading_id"])
    LLMCallLog.objects.create(
        reading=reading,
        prompt_version=state["prompt_version"],
        full_prompt=state["current_prompt"],
        model=state["model"],
        rag_chunks_used=[],  # already logged separately by rag_and_prompt_node's caller if needed
        raw_response=raw,
        attempt_number=new_attempt,
        validation_errors=validation_errors,
        final_status="ok" if (validation_ok and new_attempt == 1) else (
            "recovered" if validation_ok else "parse_failed"
        ),
        latency_ms=latency_ms,
        input_tokens=getattr(message.usage, "input_tokens", None),
        output_tokens=getattr(message.usage, "output_tokens", None),
    )

    return {
        "raw_llm_output": raw,
        "parsed_structured": parsed.get("structured"),
        "reading_text": parsed["reading_text"],
        "schema_validation_ok": validation_ok,
        "schema_validation_errors": validation_errors,
        "schema_retry_attempt": new_attempt,
        "schema_error_history": state["schema_error_history"] + [validation_errors],
        "schema_status": "ok" if (validation_ok and new_attempt == 1) else (
            "recovered" if validation_ok else "parse_failed"
        ),
    }


def judge_node(state: dict) -> dict:
    """
    Wraps run_judge(). Reads the reading's current text (already saved
    by the caller before this node runs) and produces a VerificationReport.

    Increments loop_iteration and appends to hallucination_history.
    """
    from .models import Reading

    reading = Reading.objects.get(id=state["reading_id"])
    # judge reads reading.reading_text and reading.readingcard_set —
    # caller is responsible for having saved reading_text before this node runs.
    report = run_judge(reading)

    claims = report.claims.get("claims", [])
    hallucinations = [c for c in claims if c["verdict"] == "HALLUCINATION"]
    new_loop_iteration = state["loop_iteration"] + 1

    return {
        "judge_claims": claims,
        "judge_f1": report.f1,
        "judge_precision": report.precision,
        "judge_recall": report.recall,
        "hallucination_count": len(hallucinations),
        "loop_iteration": new_loop_iteration,
        "hallucination_history": state["hallucination_history"] + [hallucinations],
    }