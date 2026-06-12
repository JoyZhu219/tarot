# Tarot Reading App

Minimal end-to-end tarot reading app: Django + PostgreSQL + Anthropic API + plain HTML frontend, all in Docker.

## Setup

1. Create a `.env` file in this directory:
```
ANTHROPIC_API_KEY=sk-ant-xxxxxxxx
```

2. Start everything:
```bash
docker compose up --build
```

3. Open http://localhost:3000

## What happens

- Backend starts at http://localhost:8000
- On first boot, Django runs migrations and seeds all 78 tarot cards
- Frontend is served by nginx at http://localhost:3000

## Flow

1. Enter your name and question
2. Choose a spread (Single Card, Past·Present·Future, Celtic Cross, Relationship, Career Path)
3. Pick cards from the full 78-card deck (filter by suit, search by name, mark reversed)
4. Click "Generate Reading" — the backend calls Claude claude-sonnet-4-6 synchronously and returns the full reading

## Project structure

```
tarot/
├── docker-compose.yml
├── .env                  ← you create this
├── backend/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── manage.py
│   ├── config/           ← Django settings & urls
│   └── tarot_app/        ← models, views, urls, seed command
└── frontend/
    ├── Dockerfile
    └── index.html        ← single-page app
```
