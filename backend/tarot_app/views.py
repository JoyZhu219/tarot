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
        f"You are an experienced tarot reader. Give a warm, insightful reading.",
        f"",
        f"Querent: {user_name}",
        f"Question: {question}",
        f"Spread: {spread_label}",
        f"",
        f"Cards drawn:",
    ]
    for item in card_objects:
        orientation = "reversed" if item["is_reversed"] else "upright"
        lines.append(
            f"  - {item['position_label']}: {item['card'].name} ({orientation}) — keywords: {item['card'].keywords}"
        )
    lines += [
        "",
        "Please provide:",
        "1. A brief interpretation of each card in its position",
        "2. A synthesized overall reading that ties everything together and directly addresses the querent's question",
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