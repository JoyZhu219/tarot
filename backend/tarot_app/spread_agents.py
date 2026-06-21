"""
spread_agents.py

OpenAI Agents SDK implementation for the pre-reading flow:

    TriageAgent
        ├─ question is clear → handoff → SpreadAdvisorAgent
        └─ question is vague/wrong format → handoff → ReQuestionAgent
                                                  ↓
                                          user confirms refined question
                                                  ↓
                                          handoff → SpreadAdvisorAgent

This module sits BEFORE the LangGraph pipeline.
Once SpreadAdvisorAgent returns recommended spreads, the user picks one
in the frontend, then the normal LangGraph pipeline starts.

Connection to LangGraph:
    The output of this module (refined question + chosen spread_key)
    is passed to OrchestratorBridge, which builds the initial ReadingState
    and invokes the graph. Nothing in the LangGraph pipeline changes.

Usage:
    from tarot_app.spread_agents import run_pre_reading_flow
    result = await run_pre_reading_flow(user_input="我最近很迷茫")
"""

import json
from typing_extensions import TypedDict
from agents import Agent, Runner, handoff, function_tool
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Spread definitions — same as SPREADS in views.py, copied here so this
# module has no Django dependency and can run independently.
# ---------------------------------------------------------------------------

SPREADS = {
    "single": {
        "label": "Single Card",
        "card_count": 1,
        "description": "One card for a focused question or daily guidance. Best for simple, direct questions.",
    },
    "past_present_future": {
        "label": "Past · Present · Future",
        "card_count": 3,
        "description": "Three cards showing how the past led to the present and where things are heading. Good for understanding evolving situations.",
    },
    "celtic_cross": {
        "label": "Celtic Cross",
        "card_count": 10,
        "description": "Ten cards for a deep, comprehensive reading. Best for complex life situations with many factors at play.",
    },
    "relationship": {
        "label": "Relationship",
        "card_count": 5,
        "description": "Five cards examining both people in a connection and the dynamic between them. Best for relationship questions.",
    },
    "career": {
        "label": "Career Path",
        "card_count": 5,
        "description": "Five cards covering current situation, obstacles, advice, outcome, and hidden factors. Best for work and career questions.",
    },
}

SPREADS_JSON = json.dumps(
    {k: {"label": v["label"], "description": v["description"]} for k, v in SPREADS.items()},
    ensure_ascii=False,
    indent=2,
)


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

@dataclass
class SpreadRecommendation:
    spread_key: str
    spread_label: str
    reason: str


@dataclass
class PreReadingResult:
    """
    Final output of the pre-reading flow.
    Passed to OrchestratorBridge to start the LangGraph pipeline.
    """
    refined_question: str
    recommended_spreads: list[SpreadRecommendation]
    original_question: str


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------

@function_tool
def get_spread_definitions() -> str:
    """Returns all available tarot spreads with their descriptions."""
    return SPREADS_JSON


class SpreadRecommendationItem(TypedDict):
    spread_key: str
    spread_label: str
    reason: str


@function_tool
def submit_recommendations(
    refined_question: str,
    recommendations: list[SpreadRecommendationItem],
) -> str:
    """
    Submit the final spread recommendations to the user.
    Call this when you have 2-3 spread recommendations ready.

    Args:
        refined_question: the question as it should be used for the reading
                          (may be the original or a refined version)
        recommendations:  list of spread recommendations with spread_key,
                          spread_label, and reason fields,
                          2-3 items, ordered from most to least recommended
    """
    return json.dumps({
        "refined_question": refined_question,
        "recommendations": recommendations,
    }, ensure_ascii=False)


@function_tool
def submit_refined_question(refined_question: str, explanation: str) -> str:
    """
    Submit a refined version of the user's question back to them for confirmation.
    Call this after you have reframed the question into something suitable for tarot.

    Args:
        refined_question: the reframed question
        explanation:      brief explanation of why/how you reframed it
    """
    return json.dumps({
        "refined_question": refined_question,
        "explanation": explanation,
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

spread_advisor_agent = Agent(
    name="SpreadAdvisor",
    model="gpt-4o-mini",
    instructions=f"""
You are an experienced tarot reader helping someone choose the right spread for their question.

Your job:
1. Call get_spread_definitions() to see available spreads.
2. Based on the user's question, recommend 2-3 spreads that are most suitable.
3. Call submit_recommendations() with your picks and a brief reason for each.

Available spreads:
{SPREADS_JSON}

Guidelines for matching questions to spreads:
- Simple, focused questions → Single Card
- "How did I get here / where am I / where am I going" → Past Present Future
- Deep, complex life situations with many unknowns → Celtic Cross
- Questions about a relationship or another person → Relationship
- Work, job, career direction questions → Career Path

Always recommend in order from most to least suitable.
Write reasons in the same language the user used.
Keep reasons concise (1-2 sentences each).
""",
    tools=[get_spread_definitions, submit_recommendations],
)


requestion_agent = Agent(
    name="ReQuestion",
    model="gpt-4o-mini",
    instructions="""
You help people reframe their questions so they are suitable for a tarot reading.

A good tarot question is:
- About the querent themselves (not "will my boyfriend change?" but "how can I navigate this relationship?")
- Open-ended (not yes/no like "will I get the job?" but "what should I focus on in my job search?")
- Focused on one topic (not "tell me everything about my life")
- Specific enough to be meaningful

Your job:
1. Identify why the original question isn't ideal for tarot.
2. Reframe it into a better question while preserving the user's core intent.
3. Call submit_refined_question() with the reframed question and a brief explanation.

After submitting the refined question, hand off to SpreadAdvisor so it can
recommend spreads based on the refined question.

Write in the same language the user used.
""",
    tools=[submit_refined_question],
    handoffs=[],  # populated below after spread_advisor_agent is defined
)

# Set handoff from ReQuestion → SpreadAdvisor after both agents are defined
requestion_agent.handoffs = [handoff(spread_advisor_agent)]


triage_agent = Agent(
    name="Triage",
    model="gpt-4o-mini",
    instructions="""
You are the entry point for a tarot reading session.
Your only job is to decide whether the user's question needs reframing before
choosing a spread, then hand off to the right agent.

Decision rules:
- Hand off to ReQuestion if the question is:
    * Too vague ("I'm confused", "I don't know what to do")
    * About someone else as the subject ("will my partner change?")
    * A yes/no question ("will I get the job?")
    * Too broad ("tell me about my life")

- Hand off to SpreadAdvisor if the question is:
    * Clear and specific enough for a reading
    * About the user themselves
    * Open-ended (not yes/no)
    * Focused on one area of life

Do NOT answer the question yourself. Do NOT ask clarifying questions.
Just assess and hand off immediately.
""",
    handoffs=[
        handoff(spread_advisor_agent),
        handoff(requestion_agent),
    ],
)


# ---------------------------------------------------------------------------
# Runner entry point
# ---------------------------------------------------------------------------

async def run_pre_reading_flow(user_input: str) -> PreReadingResult:
    """
    Run the full pre-reading agent flow.
    Returns a PreReadingResult with refined_question and recommended_spreads.

    This is the only function the Django view needs to call.
    Connect its output to OrchestratorBridge to start LangGraph.

    Example:
        result = await run_pre_reading_flow("我最近工作很迷茫")
        # result.refined_question → pass to LangGraph
        # result.recommended_spreads → show to user in frontend
    """
    result = await Runner.run(triage_agent, user_input)

    # Extract the submit_recommendations tool call output from new_items
    refined_question = user_input
    recommendations = []

    for item in result.new_items:
        # ToolCallOutputItem contains the output of a tool call
        item_type = type(item).__name__
        if "ToolCallOutput" in item_type:
            try:
                data = json.loads(item.output)
                if "recommendations" in data:
                    refined_question = data.get("refined_question", user_input)
                    recommendations = [
                        SpreadRecommendation(
                            spread_key=r["spread_key"],
                            spread_label=r["spread_label"],
                            reason=r["reason"],
                        )
                        for r in data["recommendations"]
                    ]
            except (json.JSONDecodeError, KeyError, TypeError):
                pass

    return PreReadingResult(
        refined_question=refined_question,
        recommended_spreads=recommendations,
        original_question=user_input,
    )