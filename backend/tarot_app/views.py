import anthropic
from django.conf import settings
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from .models import Card, Reading, ReadingCard

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
        reading_text = _call_llm(prompt)

        reading = Reading.objects.create(
            user_name=user_name,
            question=question,
            spread_type=spread_key,
            reading_text=reading_text,
        )
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


def _call_llm(prompt):
    client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text


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