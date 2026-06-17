import anthropic
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Card, Reading, ReadingCard, LLMCallLog

SPREADS = {
    "single": {
        "label": "Single Card",
        "positions": ["Your card"],
    },
    "past_present_future": {
        "label": "Past · Present · Future",
        "positions": ["Past", "Present", "Future"],
    },
    "celtic_cross": {
        "label": "Celtic Cross",
        "positions": [
            "Present", "Challenge", "Past", "Future",
            "Above", "Below", "Advice", "External influences",
            "Hopes & fears", "Outcome",
        ],
    },
    "relationship": {
        "label": "Relationship",
        "positions": ["You", "Them", "The connection", "Challenge", "Potential"],
    },
    "career": {
        "label": "Career Path",
        "positions": ["Current situation", "Obstacle", "Advice", "Likely outcome", "Hidden factor"],
    },
}


class SpreadListView(APIView):
    def get(self, request):
        result = [
            {"key": k, "label": v["label"], "card_count": len(v["positions"])}
            for k, v in SPREADS.items()
        ]
        return Response(result)


class CardListView(APIView):
    def get(self, request):
        cards = Card.objects.all().values("id", "name", "arcana", "suit", "keywords")
        return Response(list(cards))


class GenerateReadingView(APIView):
    def post(self, request):
        data = request.data
        user_name = data.get("user_name")
        question = data.get("question")
        spread_key = data.get("spread_type")
        selected = data.get("selected_cards", [])

        spread = SPREADS[spread_key]
        positions = spread["positions"]

        card_objects = []
        for i, item in enumerate(selected):
            card = Card.objects.get(id=item["card_id"])
            card_objects.append({
                "card": card,
                "position": i,
                "position_label": positions[i],
                "is_reversed": item.get("is_reversed", False),
            })

        prompt = _build_prompt(user_name, question, spread["label"], card_objects)
        rag_chunks_used = _fetch_rag_context(card_objects)

        # Create reading first (placeholder text) so we have an id for logging
        reading = Reading.objects.create(
            user_name=user_name,
            question=question,
            spread_type=spread_key,
            reading_text="",
        )

        from prompts.prompt_manager import prompt_manager
        active_version = prompt_manager.active_version("reading_generation")

        parsed = _call_llm_with_retry(
            prompt,
            max_retries=2,
            reading=reading,
            prompt_version=active_version,
            rag_chunks_used=rag_chunks_used,
        )
        reading_text = parsed["reading_text"]
        reading.reading_text = reading_text
        reading.save(update_fields=["reading_text"])

        for item in card_objects:
            ReadingCard.objects.create(
                reading=reading,
                card=item["card"],
                position=item["position"],
                position_label=item["position_label"],
                is_reversed=item["is_reversed"],
            )

        return Response({
            "reading_id": reading.id,
            "reading_text": reading_text,
            "validation": parsed["validation"],
            "attempts": parsed.get("attempts", 1),
            "status": parsed.get("status", "ok"),
            "cards": [
                {
                    "position_label": item["position_label"],
                    "card_name": item["card"].name,
                    "is_reversed": item["is_reversed"],
                }
                for item in card_objects
            ],
        })


def _fetch_rag_context(card_objects: list) -> dict:
    """
    Query pgvector for each card and return Waite source text.
    Returns {card_name: [chunk_text, ...]}
    Silently skips if RAG table doesn't exist yet.
    """
    try:
        from rag.retriever import retrieve_context
        result = {}
        for item in card_objects:
            card_name = item["card"].name
            chunks = retrieve_context(card_name, top_k=2)
            result[card_name] = [c["text"] for c in chunks]
        return result
    except Exception:
        return {}


def _build_cards_block(card_objects: list, rag_context: dict = None) -> str:
    """Render the cards section for insertion into prompt templates."""
    lines = []
    for item in card_objects:
        card = item["card"]
        if item["is_reversed"]:
            shadow_themes = card.reversed_required_themes or []
            lines += [
                f"  Position: {item['position_label']}",
                f"  Card: {card.name} (REVERSED)",
                f"  Official keywords [FROM_RECORD]: {card.keywords}",
                f"  Official shadow themes you MUST address [FROM_RECORD]: {shadow_themes}",
            ]
        else:
            upright_themes = card.required_themes or []
            lines += [
                f"  Position: {item['position_label']}",
                f"  Card: {card.name} (upright)",
                f"  Official keywords [FROM_RECORD]: {card.keywords}",
                f"  Official themes [FROM_RECORD]: {upright_themes}",
            ]

        # Append RAG context if available for this card
        if rag_context and card.name in rag_context:
            chunks = rag_context[card.name]
            if chunks:
                lines.append(f"  Waite's original description [FROM_RECORD]:")
                for chunk in chunks:
                    # Trim to first 300 words to keep prompt size reasonable
                    words = chunk.split()[:300]
                    trimmed = " ".join(words)
                    lines.append(f"    \"\"\"")
                    lines.append(f"    {trimmed}")
                    lines.append(f"    \"\"\"")
        lines.append("")

    return "\n".join(lines)


def _build_prompt(user_name, question, spread_label, card_objects):
    from prompts.prompt_manager import prompt_manager

    # Step 1: fetch RAG context for all cards
    rag_context = _fetch_rag_context(card_objects)

    # Step 2: build cards block with RAG context embedded
    cards_block = _build_cards_block(card_objects, rag_context=rag_context)

    return prompt_manager.render(
        "reading_generation",
        user_name=user_name,
        question=question,
        spread_label=spread_label,
        cards_block=cards_block,
    )


READING_TOOL = {
    "name": "submit_reading",
    "description": (
        "Submit a completed tarot reading in structured format. "
        "You MUST call this tool to return your reading. "
        "Every sentence must include a source tag."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "problem": {
                "type": "string",
                "description": "One sentence restating the querent's core question.",
            },
            "cards": {
                "type": "array",
                "description": "One entry per card drawn, in spread order.",
                "items": {
                    "type": "object",
                    "properties": {
                        "position_label": {"type": "string"},
                        "card_name":      {"type": "string"},
                        "is_reversed":    {"type": "boolean"},
                        "interpretation": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "sentence": {"type": "string"},
                                    "source": {
                                        "type": "string",
                                        "enum": ["FROM_RECORD", "FROM_QUERENT", "GUIDELINE", "INFERRED"],
                                    },
                                },
                                "required": ["sentence", "source"],
                            },
                        },
                    },
                    "required": ["position_label", "card_name", "is_reversed", "interpretation"],
                },
            },
            "overall": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "sentence": {"type": "string"},
                        "source": {
                            "type": "string",
                            "enum": ["FROM_RECORD", "FROM_QUERENT", "GUIDELINE", "INFERRED"],
                        },
                    },
                    "required": ["sentence", "source"],
                },
            },
        },
        "required": ["problem", "cards", "overall"],
    },
}


def _call_llm(prompt):
    import json as _json
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2048,
        tools=[READING_TOOL],
        tool_choice={"type": "any"},
        messages=[{"role": "user", "content": prompt}],
    )

    # LLM is forced to call submit_reading via tool_choice="any"
    for block in message.content:
        if block.type == "tool_use" and block.name == "submit_reading":
            return _json.dumps(block.input)

    # Fallback if LLM somehow returns text
    for block in message.content:
        if hasattr(block, "text"):
            return block.text
    return "{}"


class EvaluateReadingView(APIView):
    def get(self, request, reading_id):
        try:
            reading = Reading.objects.prefetch_related(
                'readingcard_set__card'
            ).get(id=reading_id)
        except Reading.DoesNotExist:
            return Response({"error": "Reading not found"}, status=404)

        from .evaluate import evaluate_reading
        result = evaluate_reading(reading)
        return Response(result)


class VerifyReadingView(APIView):
    def get(self, request, reading_id):
        try:
            reading = Reading.objects.prefetch_related(
                'readingcard_set__card'
            ).get(id=reading_id)
        except Reading.DoesNotExist:
            return Response({"error": "Reading not found"}, status=404)

        from .verify import verify_reading
        result = verify_reading(reading)
        return Response(result)


class JudgeReportView(APIView):
    def get(self, request, reading_id):
        try:
            reading = Reading.objects.prefetch_related(
                'readingcard_set__card'
            ).get(id=reading_id)
        except Reading.DoesNotExist:
            return Response({"error": "Reading not found"}, status=404)

        try:
            report = reading.verification_report
        except Exception:
            # Report not yet generated — run it now
            from .judge import run_judge
            report = run_judge(reading)

        return Response({
            "reading_id": reading_id,
            "status": report.status,
            "precision": report.precision,
            "recall": report.recall,
            "f1": report.f1,
            "claims": report.claims,
            "created_at": str(report.created_at),
        })


def _parse_and_validate_llm_output(raw: str) -> dict:
    import json
    from .schemas import validate_reading_output

    # Strip markdown fences if present
    clean = raw.strip()
    if clean.startswith("```"):
        clean = clean.split("\n", 1)[-1]
        clean = clean.rsplit("```", 1)[0].strip()

    # Try to parse as JSON (v3 prompt returns structured JSON)
    try:
        parsed = json.loads(clean)
    except json.JSONDecodeError:
        # plain text output (v1/v2) — skip validation
        return {
            "reading_text": raw,
            "structured": None,
            "validation": {"ok": True, "errors": [], "note": "plain text, skipped schema validation"},
        }

    # Validate against Pydantic schema
    result = validate_reading_output(parsed)

    if result.ok:
        reading_text = _structured_to_text(result.data)
        return {
            "reading_text": reading_text,
            "structured": parsed,
            "validation": {"ok": True, "errors": []},
        }
    else:
        return {
            "reading_text": raw,
            "structured": parsed,
            "validation": {"ok": False, "errors": result.errors},
        }


def _structured_to_text(reading_output) -> str:
    lines = []
    for card in reading_output.cards:
        rev = " (Reversed)" if card.is_reversed else ""
        lines.append(f"## {card.position_label}: {card.card_name}{rev}")
        for s in card.interpretation:
            lines.append(f"{s.sentence} [{s.source.value}]")
        lines.append("")
    lines.append("## Overall Reading")
    for s in reading_output.overall:
        lines.append(f"{s.sentence} [{s.source.value}]")
    return "\n".join(lines)


def _call_llm_with_retry(prompt: str, max_retries: int = 2, reading=None,
                          prompt_version: str = "unknown", rag_chunks_used: dict = None) -> dict:
    """
    Calls LLM with validation + retry loop. Logs every attempt to LLMCallLog
    if a `reading` instance is provided.
    """
    import json as _json
    import time

    last_raw = None
    last_errors = None
    messages = [{"role": "user", "content": prompt}]

    # Flatten rag_chunks_used dict into a simple list for logging
    rag_log_payload = []
    if rag_chunks_used:
        for card_name, chunks in rag_chunks_used.items():
            for chunk_text in chunks:
                rag_log_payload.append({
                    "card_name": card_name,
                    "text_preview": chunk_text[:150],
                })

    for attempt in range(1, max_retries + 2):
        print(f"[LLM] attempt {attempt}...")

        client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
        model_name = "claude-sonnet-4-6"

        start_time = time.monotonic()
        message = client.messages.create(
            model=model_name,
            max_tokens=2048,
            tools=[READING_TOOL],
            tool_choice={"type": "any"},
            messages=messages,
        )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        # Extract tool_use block
        raw = None
        tool_use_block = None
        for block in message.content:
            if block.type == "tool_use" and block.name == "submit_reading":
                raw = _json.dumps(block.input)
                tool_use_block = block
                break

        if raw is None:
            for block in message.content:
                if hasattr(block, "text"):
                    raw = block.text
                    break
            raw = raw or "{}"

        last_raw = raw
        parsed = _parse_and_validate_llm_output(raw)
        validation_ok = parsed["validation"]["ok"]
        validation_errors = parsed["validation"].get("errors", [])

        if validation_ok:
            result_status = "ok" if attempt == 1 else "recovered"
        else:
            result_status = "pending_retry"  # may still retry; final log updated below if exhausted

        # Log this attempt
        if reading is not None:
            LLMCallLog.objects.create(
                reading=reading,
                prompt_version=prompt_version,
                full_prompt=prompt,
                model=model_name,
                rag_chunks_used=rag_log_payload,
                raw_response=raw,
                attempt_number=attempt,
                validation_errors=validation_errors,
                final_status="ok" if (validation_ok and attempt == 1) else (
                    "recovered" if validation_ok else "parse_failed"
                ),
                latency_ms=latency_ms,
                input_tokens=getattr(message.usage, "input_tokens", None),
                output_tokens=getattr(message.usage, "output_tokens", None),
            )

        if validation_ok:
            return {**parsed, "attempts": attempt, "status": result_status}

        # Validation failed — build error feedback for next attempt
        last_errors = validation_errors
        error_summary = "\n".join(
            f"  - field '{e['field']}': {e['message']} (got: {repr(e.get('invalid_value'))})"
            for e in last_errors
        )

        if tool_use_block:
            messages.append({"role": "assistant", "content": message.content})
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": (
                            f"Validation failed. Please fix these errors and call submit_reading again:\n"
                            f"{error_summary}"
                        ),
                    }
                ],
            })
        else:
            messages.append({"role": "assistant", "content": raw})
            messages.append({
                "role": "user",
                "content": f"Validation errors:\n{error_summary}\nPlease fix and resubmit.",
            })

        print(f"[LLM] attempt {attempt} failed: {len(last_errors)} errors")

    # All retries exhausted — the last logged entry already has final_status=parse_failed
    print("[LLM] all attempts failed, marking parse_failed")
    return {
        "reading_text": last_raw,
        "structured": None,
        "validation": {
            "ok": False,
            "errors": last_errors,
            "note": "parse_failed after all retries",
        },
        "attempts": max_retries + 1,
        "status": "parse_failed",
    }