"""
import_themes.py

Fetches tarot_interpretations.json from dariusk/corpora and populates
required_themes (upright) and reversed_required_themes (reversed)
for every Card in the database.

Source: https://github.com/dariusk/corpora
License: CC0 (public domain)
Original interpretations: Mark McElroy, _A Guide to Tarot Meanings_

Run with:
    python manage.py import_themes
"""

import urllib.request
import json
from django.core.management.base import BaseCommand
from tarot_app.models import Card

CORPORA_URL = (
    "https://raw.githubusercontent.com/dariusk/corpora"
    "/master/data/divination/tarot_interpretations.json"
)

NAME_MAP = {
    "The Papess/High Priestess": "The High Priestess",
    "The Pope/Hierophant": "The Hierophant",
    "The Wheel": "Wheel of Fortune",
    "The Judgement": "Judgement",
}


def _normalise_name(raw_name):
    name = NAME_MAP.get(raw_name, raw_name)
    name = name.replace(" of Coins", " of Pentacles").replace(" of coins", " of Pentacles")
    return name


class Command(BaseCommand):
    help = "Import required_themes from dariusk/corpora tarot_interpretations.json"

    def handle(self, *args, **kwargs):
        self.stdout.write("Fetching data from dariusk/corpora...")
        try:
            with urllib.request.urlopen(CORPORA_URL, timeout=15) as resp:
                data = json.loads(resp.read().decode())
        except Exception as exc:
            self.stderr.write(f"Failed to fetch data: {exc}")
            return

        interpretations = data.get("tarot_interpretations", [])
        self.stdout.write(f"Loaded {len(interpretations)} cards from corpora.")

        updated = 0
        not_found = []

        for entry in interpretations:
            raw_name = entry.get("name", "")
            name = _normalise_name(raw_name)

            core_keywords = entry.get("keywords", [])
            light = entry.get("meanings", {}).get("light", [])
            shadow = entry.get("meanings", {}).get("shadow", [])

            required = core_keywords + light
            reversed_required = core_keywords + shadow

            # case-insensitive lookup, Coins already replaced with Pentacles
            try:
                card = Card.objects.get(name__iexact=name)
            except Card.DoesNotExist:
                not_found.append(name)
                self.stdout.write(
                    self.style.WARNING(f"  ? Not found in DB: '{name}' (raw: '{raw_name}')")
                )
                continue

            card.required_themes = required
            card.reversed_required_themes = reversed_required
            card.save(update_fields=["required_themes", "reversed_required_themes"])
            updated += 1
            self.stdout.write(f"  v {card.name}")

        self.stdout.write(self.style.SUCCESS(f"\nDone. Updated {updated} cards."))
        if not_found:
            self.stdout.write(
                self.style.WARNING(f"Could not match {len(not_found)} cards: {not_found}")
            )