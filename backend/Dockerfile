FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["sh", "-c", "\
  until python manage.py migrate; do \
    echo 'DB not ready, retrying in 3s...'; sleep 3; \
  done && \
  python manage.py seed_cards && \
  python manage.py import_themes && \
  python manage.py runserver 0.0.0.0:8000"]