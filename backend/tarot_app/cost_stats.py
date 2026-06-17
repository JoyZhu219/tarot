"""
cost_stats.py

Cost estimation and usage stats for LLM calls, based on LLMCallLog records.

Pricing is configurable since this project uses Claude—
default prices below reflect Claude Sonnet pricing, but can be overridden
via query params for experimentation.
"""

from datetime import datetime, timedelta
from django.db.models import Sum, Count
from django.db.models.functions import TruncDate
from .models import LLMCallLog

# Default pricing — Claude Sonnet 4.6, per million tokens (USD)
# Override via query params: ?input_price=3.00&output_price=15.00
DEFAULT_INPUT_PRICE_PER_M = 3.00
DEFAULT_OUTPUT_PRICE_PER_M = 15.00


def compute_cost_stats(start_date: str = None, end_date: str = None,
                       input_price_per_m: float = None,
                       output_price_per_m: float = None) -> dict:
    """
    Returns cost and usage stats for LLMCallLog entries in a date range.

    Args:
        start_date: "YYYY-MM-DD" (inclusive). Defaults to 30 days ago.
        end_date:   "YYYY-MM-DD" (inclusive). Defaults to today.
        input_price_per_m:  $ per million input tokens. Defaults to Claude Sonnet pricing.
        output_price_per_m: $ per million output tokens. Defaults to Claude Sonnet pricing.

    Returns:
        {
            "date_range": {"start": ..., "end": ...},
            "pricing_used": {"input_per_m": ..., "output_per_m": ...},
            "total_calls": int,
            "total_input_tokens": int,
            "total_output_tokens": int,
            "estimated_cost_usd": float,
            "daily_trend": [
                {"date": "2025-10-01", "calls": 5, "input_tokens": ..., "output_tokens": ..., "cost_usd": ...}
            ]
        }
    """
    input_price = input_price_per_m if input_price_per_m is not None else DEFAULT_INPUT_PRICE_PER_M
    output_price = output_price_per_m if output_price_per_m is not None else DEFAULT_OUTPUT_PRICE_PER_M

    # Resolve date range
    if end_date:
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
    else:
        end = datetime.now().date()

    if start_date:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
    else:
        start = end - timedelta(days=30)

    # +1 day on end to make the filter inclusive of the whole end_date
    qs = LLMCallLog.objects.filter(
        created_at__date__gte=start,
        created_at__date__lte=end,
    )

    # Totals
    totals = qs.aggregate(
        total_calls=Count('id'),
        total_input=Sum('input_tokens'),
        total_output=Sum('output_tokens'),
    )
    total_input = totals['total_input'] or 0
    total_output = totals['total_output'] or 0
    total_calls = totals['total_calls'] or 0

    estimated_cost = (
        (total_input / 1_000_000) * input_price +
        (total_output / 1_000_000) * output_price
    )

    # Daily trend
    daily_qs = (
        qs.annotate(date=TruncDate('created_at'))
        .values('date')
        .annotate(
            calls=Count('id'),
            input_tokens=Sum('input_tokens'),
            output_tokens=Sum('output_tokens'),
        )
        .order_by('date')
    )

    daily_trend = []
    for row in daily_qs:
        day_input = row['input_tokens'] or 0
        day_output = row['output_tokens'] or 0
        day_cost = (
            (day_input / 1_000_000) * input_price +
            (day_output / 1_000_000) * output_price
        )
        daily_trend.append({
            "date": row['date'].isoformat(),
            "calls": row['calls'],
            "input_tokens": day_input,
            "output_tokens": day_output,
            "cost_usd": round(day_cost, 4),
        })

    return {
        "date_range": {"start": start.isoformat(), "end": end.isoformat()},
        "pricing_used": {
            "input_per_million_tokens": input_price,
            "output_per_million_tokens": output_price,
        },
        "total_calls": total_calls,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "estimated_cost_usd": round(estimated_cost, 4),
        "daily_trend": daily_trend,
    }