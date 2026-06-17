"""
evaluator_generator_loop.py

Evaluator-Generator Loop for the reading-generation pipeline.

Design decisions (see conversation history for full reasoning):

1. Feedback transport: reuses the same conversation-history pattern as
   _call_llm_with_retry (schema validation retries). The judge's
   HALLUCINATION claims + missed_themes are formatted as a tool_result
   message appended to the same conversation, so the LLM sees its own
   prior output and is told specifically what to fix.

2. Cost control: each loop iteration runs a FULL judge pass (not
   per-card), because a single judge call costs ~$0.01-0.02 and the
   simpler design avoids needing to track which specific claims need
   re-checking. The real cost lever is max_loop_iterations, not
   per-call granularity.

3. Stopping condition (all must apply to continue, any one stops it):
   - hallucination_count == 0           -> stop, success
   - loop_iteration >= max_loop_iterations -> stop, exhausted
   - hallucination_count did not improve  -> stop early, regression guard
     (if a new generation produces >= as many hallucinations as the
     previous one, further looping is unlikely to help — stop and
     keep the best iteration seen so far, not necessarily the last one)

Pure Python. No framework dependency.
"""

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoopIterationResult:
    iteration: int
    reading_text: str
    structured: Optional[dict]
    hallucination_count: int
    f1: float
    precision: float
    recall: float
    claims: list
    latency_ms: int
    input_tokens: Optional[int]
    output_tokens: Optional[int]
    cost_usd: float


def _build_feedback_message(judge_result: dict) -> str:
    """
    Formats judge findings into a feedback message for the next generation attempt.
    Only includes HALLUCINATION claims and missed_themes — not the full claim list,
    to keep the feedback focused and not bloat the prompt.
    """
    claims = judge_result.get("claims", {}).get("claims", [])
    hallucinations = [c for c in claims if c["verdict"] == "HALLUCINATION"]
    missed = judge_result.get("claims", {}).get("missed_themes", [])

    lines = ["Your previous reading had the following problems that MUST be fixed:"]

    if hallucinations:
        lines.append("\nHALLUCINATIONS (claims that contradict official card meanings):")
        for h in hallucinations:
            lines.append(f"  - \"{h['claim']}\" — {h['reason']}")
        lines.append("Do NOT repeat these contradicted claims.")

    if missed:
        lines.append(f"\nMISSED official themes you should address: {missed}")

    lines.append(
        "\nRewrite the full reading from scratch, fixing these issues. "
        "Call submit_reading again with the corrected version."
    )
    return "\n".join(lines)


def generate_with_retry_loop(
    initial_prompt: str,
    reading,
    prompt_version: str,
    rag_chunks_used: dict,
    max_loop_iterations: int = 2,
    schema_max_retries: int = 2,
) -> dict:
    """
    Runs the Evaluator-Generator Loop:
        generate -> judge -> if hallucinations found, feed back -> regenerate
        -> repeat up to max_loop_iterations, or stop early if not improving.

    Each iteration's generation step still goes through the existing
    schema-validation retry (_call_llm_with_retry) — that's a separate,
    lower-level loop for JSON structure correctness. This loop operates
    one level up, for content correctness (hallucination).

    Args:
        initial_prompt:    the base prompt (already includes RAG context)
        reading:            Reading model instance (for logging FK)
        prompt_version:    e.g. "v4", passed through to LLMCallLog
        rag_chunks_used:   dict, passed through to LLMCallLog
        max_loop_iterations: max number of REGENERATION attempts after
                              the first one (so total generations = 1 + this)
        schema_max_retries: retry budget for the inner schema-validation loop

    Returns:
        {
            "reading_text": str,           # best iteration's text
            "structured": dict | None,
            "final_status": "success" | "exhausted" | "regression_stopped",
            "iterations": [LoopIterationResult, ...],
            "best_iteration": int,
            "total_cost_usd": float,
        }
    """
    from .views import _call_llm_with_retry
    from .judge import run_judge
    from .models import AgentDecisionLog

    iterations: list[LoopIterationResult] = []
    messages_history = [{"role": "user", "content": initial_prompt}]
    current_prompt = initial_prompt
    sequence_number = 10  # router used 1; leave room for schema-validation logs in between

    best_iteration_index = None
    best_f1 = -1.0

    for loop_i in range(max_loop_iterations + 1):  # +1 because first pass isn't a "retry"
        start_time = time.monotonic()

        parsed = _call_llm_with_retry(
            current_prompt,
            max_retries=schema_max_retries,
            reading=reading,
            prompt_version=prompt_version,
            rag_chunks_used=rag_chunks_used,
        )

        # Temporarily store this generation's text on the reading so judge can read it
        reading.reading_text = parsed["reading_text"]
        reading.save(update_fields=["reading_text"])

        judge_report = run_judge(reading)
        gen_latency_ms = int((time.monotonic() - start_time) * 1000)

        claims = judge_report.claims.get("claims", [])
        hallucination_count = sum(1 for c in claims if c["verdict"] == "HALLUCINATION")

        result = LoopIterationResult(
            iteration=loop_i,
            reading_text=parsed["reading_text"],
            structured=parsed.get("structured"),
            hallucination_count=hallucination_count,
            f1=judge_report.f1 or 0.0,
            precision=judge_report.precision or 0.0,
            recall=judge_report.recall or 0.0,
            claims=claims,
            latency_ms=gen_latency_ms,
            input_tokens=None,   # already logged per-attempt inside _call_llm_with_retry
            output_tokens=None,
            cost_usd=0.0,        # cost is tracked in LLMCallLog; this loop logs hallucination deltas
        )
        iterations.append(result)

        if result.f1 > best_f1:
            best_f1 = result.f1
            best_iteration_index = loop_i

        # Decision log for this loop iteration
        prev_count = iterations[loop_i - 1].hallucination_count if loop_i > 0 else None
        AgentDecisionLog.objects.create(
            reading=reading,
            decision_point="judge_review",
            sequence_number=sequence_number + loop_i,
            input_data={
                "loop_iteration": loop_i,
                "previous_hallucination_count": prev_count,
            },
            decision=(
                f"hallucination_count={hallucination_count}, f1={result.f1:.3f} "
                f"({'first generation' if loop_i == 0 else 'after feedback regeneration'})"
            ),
            rationale=_stop_condition_rationale(
                loop_i, hallucination_count, prev_count, max_loop_iterations
            ),
            output_data={
                "f1": result.f1,
                "precision": result.precision,
                "recall": result.recall,
                "hallucination_count": hallucination_count,
            },
            latency_ms=gen_latency_ms,
            outcome=(
                "success" if hallucination_count == 0
                else ("corrected" if loop_i > 0 else "failed")
            ),
        )

        # --- Stopping conditions ---
        if hallucination_count == 0:
            return _finalize(iterations, best_iteration_index, "success")

        if loop_i >= max_loop_iterations:
            return _finalize(iterations, best_iteration_index, "exhausted")

        if prev_count is not None and hallucination_count >= prev_count:
            # Regression guard: not improving, stop early and use best iteration so far
            return _finalize(iterations, best_iteration_index, "regression_stopped")

        # --- Build feedback and continue loop ---
        feedback = _build_feedback_message({"claims": judge_report.claims})
        current_prompt = initial_prompt + "\n\n== PREVIOUS ATTEMPT FEEDBACK ==\n" + feedback

    # Should not reach here, but safety fallback
    return _finalize(iterations, best_iteration_index, "exhausted")


def _stop_condition_rationale(loop_i, hallucination_count, prev_count, max_loop_iterations) -> str:
    if hallucination_count == 0:
        return f"Stopping: hallucination_count=0, reading is clean after {loop_i} regeneration(s)."
    if loop_i >= max_loop_iterations:
        return (
            f"Stopping: reached max_loop_iterations={max_loop_iterations} with "
            f"hallucination_count={hallucination_count} still > 0. Keeping best F1 iteration."
        )
    if prev_count is not None and hallucination_count >= prev_count:
        return (
            f"Stopping early: hallucination_count={hallucination_count} did not improve "
            f"from previous={prev_count}. Regeneration is not helping; keeping best F1 iteration "
            f"rather than continuing to loop."
        )
    return (
        f"Continuing: hallucination_count={hallucination_count} "
        f"(previous={prev_count}), under max_loop_iterations={max_loop_iterations}. "
        f"Feeding judge findings back into prompt for regeneration."
    )


def _finalize(iterations: list, best_iteration_index: int, final_status: str) -> dict:
    best = iterations[best_iteration_index]
    return {
        "reading_text": best.reading_text,
        "structured": best.structured,
        "final_status": final_status,
        "iterations": iterations,
        "best_iteration": best_iteration_index,
        "best_f1": best.f1,
        "best_hallucination_count": best.hallucination_count,
        "total_generations": len(iterations),
    }