"""
test_graph_scenarios.py

Tests the three core paths through the reading-generation graph by
mocking the Anthropic client and (where needed) run_judge, so no real
API calls happen and each path is deterministically triggered.

Scenarios:
  1. test_one_shot_success      — first generation is valid, no hallucination
                                   -> graph should end via route_after_judge's
                                   "end_success" without ever hitting retry_prompt
  2. test_parse_failure_then_retry — first LLM response is malformed JSON
                                   (missing required field), second is valid
                                   -> route_after_generate should take
                                   "retry_prompt" once, then "judge" on attempt 2
  3. test_hallucination_then_regenerate — first response is schema-valid but
                                   judge finds a semantic HALLUCINATION,
                                   second response is clean
                                   -> route_after_judge should take
                                   "regenerate" once, then "end_success"

How the mocking works:
  - anthropic.Anthropic is patched so client.messages.create() returns
    a pre-built fake message object (with .content containing a
    tool_use block) instead of calling the real API.
  - side_effect is used to return DIFFERENT fake responses on the 1st
    vs 2nd call, which is how retries are simulated.
  - run_judge is patched directly in scenario 3, since hallucination
    detection depends on judge's own LLM call — we don't need to mock
    that LLM call too, just its return value.

Run with:
    docker compose exec backend python manage.py test tarot_app.tests.test_graph_scenarios
"""

import json
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

from django.test import TestCase

from tarot_app.models import Card, Reading, ReadingCard, VerificationReport
from tarot_app.graph_state import build_initial_state
from tarot_app.graph_edges import build_graph


# ---------------------------------------------------------------------------
# Helpers to build fake Anthropic responses
# ---------------------------------------------------------------------------

def _fake_tool_use_message(input_dict: dict, input_tokens=500, output_tokens=300):
    """
    Builds a fake anthropic.types.Message-like object containing a
    tool_use block, matching what generate_node expects to parse.
    """
    tool_block = SimpleNamespace(
        type="tool_use",
        name="submit_reading",
        input=input_dict,
        id="fake_tool_use_id_123",
    )
    return SimpleNamespace(
        content=[tool_block],
        usage=SimpleNamespace(input_tokens=input_tokens, output_tokens=output_tokens),
    )


VALID_READING_JSON = {
    "problem": "Will I find love this year?",
    "cards": [
        {
            "position_label": "Your card",
            "card_name": "The Sun",
            "is_reversed": False,
            "interpretation": [
                {"sentence": "The Sun brings warmth and clarity to your question.", "source": "FROM_RECORD"},
                {"sentence": "This suggests a joyful period ahead.", "source": "INFERRED"},
            ],
        }
    ],
    "overall": [
        {"sentence": "The cards point toward a positive outcome.", "source": "INFERRED"},
    ],
}

# Missing required field "overall" -> Pydantic validation will fail
MALFORMED_READING_JSON = {
    "problem": "Will I find love this year?",
    "cards": [
        {
            "position_label": "Your card",
            "card_name": "The Sun",
            "is_reversed": False,
            "interpretation": [
                {"sentence": "The Sun brings warmth.", "source": "FROM_RECORD"},
            ],
        }
    ],
    # "overall" deliberately omitted
}

# Schema-valid, but content will be judged as hallucination (reads a
# reversed card positively, which is the pattern from earlier conversation)
SCHEMA_VALID_BUT_HALLUCINATING_JSON = {
    "problem": "What does this reversed card mean for me?",
    "cards": [
        {
            "position_label": "Your card",
            "card_name": "Five of Swords",
            "is_reversed": True,
            "interpretation": [
                {"sentence": "Reversed, this card means you are healing and releasing old conflict.",
                 "source": "INFERRED"},
            ],
        }
    ],
    "overall": [
        {"sentence": "You are moving toward reconciliation and peace.", "source": "INFERRED"},
    ],
}


# ---------------------------------------------------------------------------
# Shared test setup
# ---------------------------------------------------------------------------

class GraphScenarioTestCase(TestCase):
    def setUp(self):
        # Minimal card fixtures — enough for card_objects/state construction
        self.sun_card = Card.objects.create(
            name="The Sun", arcana="major", suit="",
            keywords="joy, success",
            required_themes=["joy", "success", "vitality"],
            reversed_required_themes=["temporary sadness", "lack of clarity"],
        )
        self.five_swords = Card.objects.create(
            name="Five of Swords", arcana="minor", suit="swords",
            keywords="conflict, defeat",
            required_themes=["conflict", "winning at a cost"],
            reversed_required_themes=["hostility", "picking fights", "intimidating others"],
        )

    def _patch_rag_and_prompt(self):
        """
        rag_and_prompt_node calls _fetch_rag_context (queries pgvector — not
        available in test DB) and _build_cards_block. Mock it out so the
        graph receives a valid current_prompt without hitting RAG.
        """
        return patch(
            "tarot_app.graph_nodes.rag_and_prompt_node",
            side_effect=lambda state: {"current_prompt": "FAKE PROMPT FOR TESTING"},
        )

    def _patch_router(self):
        """
        router_node calls AgentDecisionLog.objects.create() — the
        AgentDecisionLog migration may not be applied in the test DB yet,
        and we don't want routing tests to depend on that table existing.
        Return a minimal strategy dict that matches what router_node would
        write to state, so the rest of the graph runs with valid state.
        """
        return patch(
            "tarot_app.graph_nodes.router_node",
            side_effect=lambda state: {
                "strategy_name": "simple",
                "prompt_version": "v4",
                "model": "claude-sonnet-4-6",
                "max_retries": 2,
                "max_loop_iterations": 2,
                "rag_top_k": 2,
                "requires_narrative_linking": False,
            },
        )

    def _build_state_for_card(self, card: Card, is_reversed: bool, question: str):
        reading = Reading.objects.create(
            user_name="TestUser",
            question=question,
            spread_type="single",
            reading_text="",
        )
        ReadingCard.objects.create(
            reading=reading, card=card, position=0,
            position_label="Your card", is_reversed=is_reversed,
        )
        card_objects_for_state = [{
            "card_id": card.id,
            "card_name": card.name,
            "position": 0,
            "position_label": "Your card",
            "is_reversed": is_reversed,
        }]
        state = build_initial_state(
            reading_id=reading.id,
            user_name="TestUser",
            question=question,
            spread_key="single",
            spread_label="Single Card",
            card_objects=card_objects_for_state,
        )
        return reading, state


# ---------------------------------------------------------------------------
# Scenario 1: one-shot success
# ---------------------------------------------------------------------------

class OneShotSuccessTest(GraphScenarioTestCase):

    def test_one_shot_success(self):
        """First generation is valid and judge finds no hallucination —
        graph should reach end_success without any retry_prompt loop.

        generate_node is mocked directly (instead of mocking anthropic.Anthropic
        inside it) because LangGraph runs nodes in its own execution context
        where decorator-level patches may not propagate reliably.
        """
        reading, state = self._build_state_for_card(
            self.sun_card, is_reversed=False, question="Will I find love?"
        )

        fake_report = SimpleNamespace(
            f1=1.0, precision=1.0, recall=1.0,
            claims={"claims": [], "covered_themes": ["joy", "success"], "missed_themes": []},
        )

        generate_call_count = {"n": 0}

        def fake_generate(state):
            generate_call_count["n"] += 1
            return {
                "raw_llm_output": "{}",
                "parsed_structured": VALID_READING_JSON,
                "reading_text": "The Sun shines warmly. [FROM_RECORD]",
                "schema_validation_ok": True,
                "schema_validation_errors": [],
                "schema_retry_attempt": generate_call_count["n"],
                "schema_error_history": state["schema_error_history"] + [[]],
                "schema_status": "ok",
            }

        with self._patch_router(), self._patch_rag_and_prompt(), \
             patch("tarot_app.graph_nodes.generate_node", side_effect=fake_generate), \
             patch("tarot_app.graph_nodes.run_judge", return_value=fake_report):
            graph = build_graph()
            final_state = graph.invoke(state)

        self.assertEqual(final_state["schema_status"], "ok")
        self.assertTrue(final_state["schema_validation_ok"])
        self.assertEqual(final_state["hallucination_count"], 0)
        self.assertEqual(generate_call_count["n"], 1)  # only one generation call


# ---------------------------------------------------------------------------
# Scenario 2: parse failure then retry
# ---------------------------------------------------------------------------

class ParseFailureRetryTest(GraphScenarioTestCase):

    def test_parse_failure_then_retry(self):
        """First response is missing the required 'overall' field
        (Pydantic validation fails) -> graph retries -> second response
        is valid -> ends in judge with schema_status='recovered'."""

        reading, state = self._build_state_for_card(
            self.sun_card, is_reversed=False, question="Will I find love?"
        )

        fake_report = SimpleNamespace(
            f1=1.0, precision=1.0, recall=1.0,
            claims={"claims": [], "covered_themes": ["joy"], "missed_themes": []},
        )

        generate_call_count = {"n": 0}

        def fake_generate(state):
            generate_call_count["n"] += 1
            attempt = generate_call_count["n"]
            if attempt == 1:
                # First call: validation fails (missing 'overall')
                errors = [{"field": "overall", "message": "overall must have at least one sentence", "invalid_value": None}]
                return {
                    "raw_llm_output": "{}",
                    "parsed_structured": None,
                    "reading_text": "{}",
                    "schema_validation_ok": False,
                    "schema_validation_errors": errors,
                    "schema_retry_attempt": attempt,
                    "schema_error_history": state["schema_error_history"] + [errors],
                    "schema_status": "parse_failed",
                }
            else:
                # Second call: valid
                return {
                    "raw_llm_output": "{}",
                    "parsed_structured": VALID_READING_JSON,
                    "reading_text": "The Sun shines. [FROM_RECORD]",
                    "schema_validation_ok": True,
                    "schema_validation_errors": [],
                    "schema_retry_attempt": attempt,
                    "schema_error_history": state["schema_error_history"] + [[]],
                    "schema_status": "recovered",
                }

        with self._patch_router(), self._patch_rag_and_prompt(), \
             patch("tarot_app.graph_nodes.generate_node", side_effect=fake_generate), \
             patch("tarot_app.graph_nodes.run_judge", return_value=fake_report):
            graph = build_graph()
            final_state = graph.invoke(state)

        self.assertEqual(generate_call_count["n"], 2)
        self.assertEqual(final_state["schema_retry_attempt"], 2)
        self.assertEqual(final_state["schema_status"], "recovered")
        self.assertTrue(final_state["schema_validation_ok"])
        self.assertEqual(len(final_state["schema_error_history"]), 2)
        self.assertTrue(len(final_state["schema_error_history"][0]) > 0)
        self.assertEqual(len(final_state["schema_error_history"][1]), 0)


# ---------------------------------------------------------------------------
# Scenario 3: hallucination then regenerate
# ---------------------------------------------------------------------------

class HallucinationRegenerateTest(GraphScenarioTestCase):

    def test_hallucination_then_regenerate(self):
        """First response is schema-valid but judge finds a semantic hallucination
        -> route_after_judge sends it to 'regenerate' -> second response is clean
        -> end_success."""

        reading, state = self._build_state_for_card(
            self.five_swords, is_reversed=True,
            question="What does this reversed card mean for me?",
        )

        hallucinating_report = SimpleNamespace(
            f1=0.1, precision=0.0, recall=0.2,
            claims={
                "claims": [{
                    "card_name": "Five of Swords",
                    "claim": "Reversed, this card means you are healing and releasing old conflict.",
                    "verdict": "HALLUCINATION",
                    "reason": "Official reversed themes indicate hostility and picking fights, not healing.",
                }],
                "covered_themes": [],
                "missed_themes": ["hostility", "picking fights", "intimidating others"],
            },
        )
        clean_report = SimpleNamespace(
            f1=1.0, precision=1.0, recall=1.0,
            claims={"claims": [], "covered_themes": ["hostility"], "missed_themes": []},
        )

        generate_call_count = {"n": 0}

        def fake_generate(state):
            generate_call_count["n"] += 1
            return {
                "raw_llm_output": "{}",
                "parsed_structured": VALID_READING_JSON,
                "reading_text": "Some reading text. [FROM_RECORD]",
                "schema_validation_ok": True,
                "schema_validation_errors": [],
                "schema_retry_attempt": generate_call_count["n"],
                "schema_error_history": state["schema_error_history"] + [[]],
                "schema_status": "ok",
            }

        with self._patch_router(), self._patch_rag_and_prompt(), \
             patch("tarot_app.graph_nodes.generate_node", side_effect=fake_generate), \
             patch("tarot_app.graph_nodes.run_judge",
                   side_effect=[hallucinating_report, clean_report]):
            graph = build_graph()
            final_state = graph.invoke(state)

        self.assertEqual(generate_call_count["n"], 2)
        self.assertEqual(final_state["loop_iteration"], 2)
        self.assertEqual(final_state["hallucination_count"], 0)
        self.assertEqual(len(final_state["hallucination_history"]), 2)
        self.assertEqual(len(final_state["hallucination_history"][0]), 1)
        self.assertEqual(len(final_state["hallucination_history"][1]), 0)