# ✦ AI Tarot Reading System

A full-stack AI-powered tarot reading application built as a hands-on learning project for exploring real engineering challenges in LLM systems — including hallucination evaluation, structured output, RAG pipelines, and multi-agent orchestration.

## Stack

| Layer | Technology |
|---|---|
| Backend | Python 3.11, Django 4.2, Django REST Framework |
| Database | PostgreSQL with pgvector extension |
| LLM | Anthropic Claude (claude-sonnet-4-6) via tool use |
| Pre-reading agents | OpenAI Agents SDK (GPT-4o-mini) |
| Embeddings | fastembed (BAAI/bge-small-en-v1.5, ONNX) |
| Graph orchestration | LangGraph (StateGraph) |
| Infrastructure | Docker, docker-compose |
| Frontend | Vanilla HTML/JS, nginx |

## Features

- **Intelligent spread recommendation** — a three-agent pipeline (TriageAgent → SpreadAdvisorAgent / ReQuestionAgent) classifies the user's question quality and recommends the most suitable spreads before the reading begins
- **Structured JSON output** — LLM output is forced through tool use (`tool_choice: any`) and validated with Pydantic; schema errors are fed back to the LLM for self-correction
- **RAG-grounded readings** — Waite's *Pictorial Key to the Tarot* (1910, public domain) is chunked, embedded, and stored in pgvector; relevant passages are injected per card at generation time
- **Source attribution** — every sentence in a reading is tagged `[FROM_RECORD]`, `[FROM_QUERENT]`, `[GUIDELINE]`, or `[INFERRED]`
- **LLM-as-Judge evaluation** — a second Claude call audits each reading for hallucinations against official card themes, producing per-claim VERIFIED / UNVERIFIED / HALLUCINATION verdicts and F1 scores
- **Complexity-aware router** — spread type (not card count alone) determines RAG depth and retry budget; Past-Present-Future is treated as complex because it requires narrative linking between positions
- **Full observability** — every LLM call is logged to `LLMCallLog` (prompt, tokens, latency, cost); every pipeline decision is logged to `AgentDecisionLog` (input, decision, rationale, outcome)
- **Tarot card UI** — 78 Rider-Waite card images displayed as an overlapping fan; shuffle animation before card selection

## Architecture

```
User input (question)
    ↓
OpenAI Agents SDK — Pre-reading flow
    TriageAgent → SpreadAdvisorAgent   (clear question)
    TriageAgent → ReQuestionAgent → SpreadAdvisorAgent   (vague/yes-no)
    ↓
User selects spread + draws cards
    ↓
ReadingOrchestrator (Django)
    → Router (assess_complexity)        writes AgentDecisionLog
    → RAG retrieval (pgvector)          writes AgentDecisionLog
    → Prompt build                      writes AgentDecisionLog
    → LLM generation (Claude, tool use) writes LLMCallLog
        schema validation + retry loop
    → Response returned to user
    → Judge (background thread)         writes VerificationReport
```

## API Endpoints

| Method | Path | Description |
|---|---|---|
| POST | `/api/recommend-spreads/` | Triage + spread recommendations |
| GET | `/api/cards/` | All 78 cards |
| GET | `/api/spreads/` | Available spreads |
| POST | `/api/reading/` | Generate a reading |
| GET | `/api/reading/{id}/judge/` | LLM-as-Judge report |
| GET | `/api/reading/{id}/verify/` | Keyword verification (Layer 1) |
| GET | `/api/reading/{id}/evaluate/` | F1 scores |
| GET | `/api/llm-costs/` | Cost stats with date range + custom pricing |

## Getting Started

**Prerequisites:** Docker, Docker Compose, Anthropic API key, OpenAI API key

```bash
git clone <repo>
cd tarot

# Add API keys to .env
echo "ANTHROPIC_API_KEY=sk-ant-..." >> .env
echo "OPENAI_API_KEY=sk-..." >> .env

docker compose up --build
```

Frontend: `http://localhost:3000`
Backend: `http://localhost:8000`

## Running Tests

```bash
# LangGraph graph routing scenarios (3 tests)
docker compose exec backend python manage.py test tarot_app.tests.test_graph_scenarios -v 2

# OpenAI Agents SDK agent structure and handoff tests (8 tests)
docker compose exec backend python manage.py test tarot_app.tests.test_spread_agents -v 2
```

## Project Structure

```
tarot/
├── docker-compose.yml
├── backend/
│   ├── prompts/                    prompt versioning system (v1–v4)
│   ├── rag/                        knowledge base prep + pgvector ingestion
│   └── tarot_app/
│       ├── models.py               Card, Reading, LLMCallLog, AgentDecisionLog
│       ├── orchestrator.py         ReadingOrchestrator — wires router → RAG → loop
│       ├── router.py               assess_complexity() — spread-aware routing
│       ├── evaluator_generator_loop.py   schema validation retry loop
│       ├── judge.py                LLM-as-Judge (Layer 2 evaluation)
│       ├── verify.py               keyword matching (Layer 1 evaluation)
│       ├── spread_agents.py        OpenAI Agents SDK — Triage/SpreadAdvisor/ReQuestion
│       ├── graph_state.py          LangGraph TypedDict state
│       ├── graph_nodes.py          LangGraph node functions
│       ├── graph_edges.py          LangGraph conditional edges + StateGraph assembly
│       └── tests/
│           ├── test_graph_scenarios.py
│           └── test_spread_agents.py
└── frontend/
    ├── index.html
    └── cards/                      78 Rider-Waite card images (local)
```