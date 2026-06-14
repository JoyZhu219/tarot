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


def _build_prompt(user_name, question, spread_label, card_objects):
    lines = [
        "You are an experienced tarot reader grounded in the Rider-Waite-Smith tradition.",
        "Give a warm, insightful reading that is faithful to each card's official meanings.",
        "",
        "== SOURCE LABELING RULES ==",
        "Every sentence or claim in your reading MUST end with a source tag.",
        "Use exactly one of these tags per sentence:",
        "",
        "  [FROM_RECORD]  — directly stated in the official keywords or themes provided below",
        "  [FROM_QUERENT] — comes from the querent's name, question, or spread choice",
        "  [GUIDELINE]    — comes from general Rider-Waite-Smith tarot tradition,",
        "                   not in the specific themes listed but widely accepted",
        "  [INFERRED]     — your own synthesis, connection, or interpretive leap.",
        "                   When in doubt, use INFERRED rather than GUIDELINE.",
        "",
        "Rules:",
        "- Every sentence needs exactly one tag at the end, in square brackets.",
        "- Do not skip tagging any sentence.",
        "- Do not invent card meanings. If you add something not in the themes, tag it [INFERRED].",
        "- For REVERSED cards: shadow themes are [FROM_RECORD]. Do NOT reframe them positively.",
        "",
        "== CRITICAL RULES FOR REVERSED CARDS ==",
        "- A REVERSED card carries shadow, blocked, or distorted energy.",
        "- You MUST interpret it using the official shadow themes listed below.",
        "- Do NOT reframe reversed cards as healing or positive turning points",
        "  unless the official shadow themes explicitly support this.",
        "- For each reversed card, explicitly name what energy is blocked or distorted.",
        "",
        f"Querent name: {user_name} [FROM_QUERENT]",
        f"Question: {question} [FROM_QUERENT]",
        f"Spread: {spread_label} [FROM_QUERENT]",
        "",
        "== CARDS DRAWN ==",
    ]
    for item in card_objects:
        card = item["card"]
        if item["is_reversed"]:
            shadow_themes = card.reversed_required_themes or []
            lines += [
                f"  Position: {item['position_label']}",
                f"  Card: {card.name} (REVERSED)",
                f"  Official keywords [FROM_RECORD]: {card.keywords}",
                f"  Official shadow themes you MUST address [FROM_RECORD]: {shadow_themes}",
                "",
            ]
        else:
            upright_themes = card.required_themes or []
            lines += [
                f"  Position: {item['position_label']}",
                f"  Card: {card.name} (upright)",
                f"  Official keywords [FROM_RECORD]: {card.keywords}",
                f"  Official themes [FROM_RECORD]: {upright_themes}",
                "",
            ]
    lines += [
        "== OUTPUT FORMAT ==",
        "Write a reading where every sentence ends with its source tag.",
        "Example format:",
        "  'This card speaks to sudden disruption in your life. [FROM_RECORD]",
        "   The energy feels especially intense given your question about work. [FROM_QUERENT]",
        "   This may point to a need to slow down before making decisions. [INFERRED]'",
        "",
        "Please provide:",
        "1. A brief interpretation of each card in its position, tagging every sentence",
        "2. A synthesized overall reading addressing the querent's question, tagging every sentence",
        "Write in second person (you/your), warm and direct tone.",
    ]
    return "\n".join(lines)


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