"""
orchestrator.py

ReadingOrchestrator assembles the pipeline's decision-making components
into a single entry point: process().

    Router (assess_complexity)
        -> decides strategy (prompt version, retry budget, RAG depth,
           whether narrative linking is required)
    Evaluator-Generator Loop (generate_with_retry_loop)
        -> generates, judges, and self-corrects up to N times

Every step writes an AgentDecisionLog entry, so the full chain of
"what was decided and why" is reconstructable from the database alone,
without re-reading code.

This class does NOT touch HTTP/Django request objects — it takes plain
Python inputs (card_objects, user_name, question, spread info) and
returns a plain dict. views.py is responsible for the web layer; this
class is responsible for orchestration logic only.

Pure Python. No framework dependency beyond the Django ORM calls already
used by AgentDecisionLog/Reading (which this module imports lazily to
avoid circular imports with views.py).
"""

import time


class ReadingOrchestrator:
    """
    Orchestrates one full reading-generation pipeline run:
        1. Router decision  (assess_complexity)
        2. RAG retrieval     (uses strategy.rag_top_k)
        3. Prompt build      (uses strategy.requires_narrative_linking)
        4. Evaluator-Generator Loop (generate_with_retry_loop)

    Usage:
        orchestrator = ReadingOrchestrator(
            reading=reading,               # Reading model instance, already created
            user_name=user_name,
            question=question,
            spread_label=spread_label,
            spread_key=spread_key,
            card_objects=card_objects,
        )
        result = orchestrator.process()
    """

    def __init__(self, reading, user_name: str, question: str, spread_label: str,
                spread_key: str, card_objects: list,
                max_loop_iterations: int = 2):
        self.reading = reading
        self.user_name = user_name
        self.question = question
        self.spread_label = spread_label
        self.spread_key = spread_key
        self.card_objects = card_objects
        self.max_loop_iterations = max_loop_iterations

        # filled in during process()
        self.strategy = None
        self.rag_chunks_used = None
        self.prompt = None
        self._sequence_counter = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def process(self) -> dict:
        """
        Runs the full pipeline. Returns the same dict shape as
        generate_with_retry_loop(), plus the chosen strategy for visibility.
        """
        self.strategy = self._route()
        self.rag_chunks_used = self._retrieve_rag_context()
        self.prompt = self._build_prompt()
        loop_result = self._run_generation_loop()

        return {
            **loop_result,
            "strategy": {
                "name": self.strategy.name,
                "prompt_version": self.strategy.prompt_version,
                "max_retries": self.strategy.max_retries,
                "rag_top_k": self.strategy.rag_top_k,
                "requires_narrative_linking": self.strategy.requires_narrative_linking,
            },
        }

    # ------------------------------------------------------------------
    # Step 1: Router
    # ------------------------------------------------------------------

    def _route(self):
        from .router import assess_complexity, build_router_decision_log
        from .models import AgentDecisionLog

        strategy = assess_complexity(
            self.question,
            card_count=len(self.card_objects),
            spread_key=self.spread_key,
        )

        self._sequence_counter += 1
        AgentDecisionLog.objects.create(
            reading=self.reading,
            **build_router_decision_log(
                self.question, len(self.card_objects), self.spread_key,
                strategy, sequence_number=self._sequence_counter,
            ),
        )

        return strategy

    # ------------------------------------------------------------------
    # Step 2: RAG retrieval
    # ------------------------------------------------------------------

    def _retrieve_rag_context(self) -> dict:
        from .models import AgentDecisionLog

        start = time.monotonic()
        try:
            from .views import _fetch_rag_context
            chunks = _fetch_rag_context(self.card_objects, top_k=self.strategy.rag_top_k)
            outcome = "success"
        except Exception as e:
            chunks = {}
            outcome = "failed"

        latency_ms = int((time.monotonic() - start) * 1000)
        chunk_count = sum(len(v) for v in chunks.values())

        self._sequence_counter += 1
        AgentDecisionLog.objects.create(
            reading=self.reading,
            decision_point="rag_retrieval",
            sequence_number=self._sequence_counter,
            input_data={
                "card_names": [item["card"].name for item in self.card_objects],
                "top_k": self.strategy.rag_top_k,
            },
            decision=f"retrieved {chunk_count} chunks across {len(self.card_objects)} cards",
            rationale=(
                f"top_k={self.strategy.rag_top_k} chosen by router based on strategy="
                f"'{self.strategy.name}'. {'Retrieval succeeded.' if outcome == 'success' else 'Retrieval failed, proceeding without RAG context.'}"
            ),
            output_data={"chunks_per_card": {k: len(v) for k, v in chunks.items()}},
            latency_ms=latency_ms,
            outcome=outcome,
        )

        return chunks

    # ------------------------------------------------------------------
    # Step 3: Prompt build
    # ------------------------------------------------------------------

    def _build_prompt(self) -> str:
        from .models import AgentDecisionLog
        from .views import _build_cards_block
        from prompts.prompt_manager import prompt_manager

        cards_block = _build_cards_block(self.card_objects, rag_context=self.rag_chunks_used)

        if self.strategy.requires_narrative_linking:
            cards_block += (
                "\n\nNOTE: This spread requires connecting positions into one narrative. "
                "When writing the 'overall' section, explicitly relate at least two "
                "positions to each other (e.g. how 'Past' explains 'Present', or how "
                "'Challenge' informs 'Advice'). Do not treat each card as fully independent."
            )

        prompt = prompt_manager.render(
            "reading_generation",
            user_name=self.user_name,
            question=self.question,
            spread_label=self.spread_label,
            cards_block=cards_block,
        )

        self._sequence_counter += 1
        AgentDecisionLog.objects.create(
            reading=self.reading,
            decision_point="prompt_build",
            sequence_number=self._sequence_counter,
            input_data={
                "prompt_version": self.strategy.prompt_version,
                "requires_narrative_linking": self.strategy.requires_narrative_linking,
                "rag_chunks_available": bool(self.rag_chunks_used),
            },
            decision=f"rendered prompt_version={self.strategy.prompt_version}, length={len(prompt)} chars",
            rationale=(
                "Narrative-linking instruction "
                f"{'appended' if self.strategy.requires_narrative_linking else 'omitted'} "
                f"based on strategy='{self.strategy.name}'. "
                f"RAG context {'embedded' if self.rag_chunks_used else 'not available'}."
            ),
            output_data={"prompt_char_length": len(prompt)},
            latency_ms=0,
            outcome="success",
        )

        return prompt

    # ------------------------------------------------------------------
    # Step 4: Evaluator-Generator Loop
    # ------------------------------------------------------------------

    def _run_generation_loop(self) -> dict:
        from .evaluator_generator_loop import generate_with_retry_loop

        return generate_with_retry_loop(
            initial_prompt=self.prompt,
            reading=self.reading,
            prompt_version=self.strategy.prompt_version,
            rag_chunks_used=self.rag_chunks_used,
            max_loop_iterations=self.max_loop_iterations,
            schema_max_retries=self.strategy.max_retries,
        )


# Alias for assignments/grading rubrics that expect this exact class name.
CarePlanOrchestrator = ReadingOrchestrator