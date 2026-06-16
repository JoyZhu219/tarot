"""
schemas.py  —  Pydantic models for LLM reading output validation

Defines the expected JSON structure for a tarot reading and validates
LLM output against it. Used in views.py after _call_llm().

Reading structure:
{
  "problem": "...",                        # querent's question restated
  "cards": [
    {
      "position_label": "Past",
      "card_name": "The Tower",
      "is_reversed": true,
      "interpretation": [
        {
          "sentence": "...",
          "source": "FROM_RECORD"
        }
      ]
    }
  ],
  "overall": [
    {
      "sentence": "...",
      "source": "INFERRED"
    }
  ]
}

Validation rules:
  - problem:          required, non-empty string
  - cards:            required, non-empty list
  - cards[].position_label:   required string
  - cards[].card_name:        required string
  - cards[].is_reversed:      required bool
  - cards[].interpretation:   required, non-empty list of sentences
  - sentence.sentence:        required, non-empty string
  - sentence.source:          required, must be one of the 4 valid tags
  - overall:          required, non-empty list of sentences

Type coercion:
  - is_reversed: "true"/"false" strings auto-converted to bool
  - source tags: stripped of whitespace and brackets if LLM adds them
"""

from enum import Enum
from typing import List
from pydantic import BaseModel, field_validator, model_validator, ValidationError


# ---------------------------------------------------------------------------
# Source tag enum
# ---------------------------------------------------------------------------

class SourceTag(str, Enum):
    FROM_RECORD  = "FROM_RECORD"
    FROM_QUERENT = "FROM_QUERENT"
    GUIDELINE    = "GUIDELINE"
    INFERRED     = "INFERRED"


# ---------------------------------------------------------------------------
# Sentence model
# ---------------------------------------------------------------------------

class Sentence(BaseModel):
    sentence: str
    source: SourceTag

    @field_validator("sentence")
    @classmethod
    def sentence_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("sentence cannot be empty")
        return v

    @field_validator("source", mode="before")
    @classmethod
    def coerce_source_tag(cls, v):
        """
        Strip brackets and whitespace if LLM returns [FROM_RECORD] instead of FROM_RECORD.
        Also handle lowercase variants.
        """
        if isinstance(v, str):
            v = v.strip().strip("[]").upper()
        return v


# ---------------------------------------------------------------------------
# Card interpretation model
# ---------------------------------------------------------------------------

class CardReading(BaseModel):
    position_label: str
    card_name: str
    is_reversed: bool
    interpretation: List[Sentence]

    @field_validator("position_label", "card_name")
    @classmethod
    def not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("field cannot be empty")
        return v

    @field_validator("is_reversed", mode="before")
    @classmethod
    def coerce_bool(cls, v):
        """Auto-convert string 'true'/'false' to bool."""
        if isinstance(v, str):
            if v.lower() == "true":
                return True
            if v.lower() == "false":
                return False
            raise ValueError(f"Cannot convert '{v}' to bool")
        return v

    @field_validator("interpretation")
    @classmethod
    def interpretation_not_empty(cls, v):
        if not v:
            raise ValueError("interpretation must have at least one sentence")
        return v


# ---------------------------------------------------------------------------
# Full reading model
# ---------------------------------------------------------------------------

class ReadingOutput(BaseModel):
    problem: str
    cards: List[CardReading]
    overall: List[Sentence]

    @field_validator("problem")
    @classmethod
    def problem_not_empty(cls, v):
        v = v.strip()
        if not v:
            raise ValueError("problem cannot be empty")
        return v

    @field_validator("cards")
    @classmethod
    def cards_not_empty(cls, v):
        if not v:
            raise ValueError("cards must have at least one entry")
        return v

    @field_validator("overall")
    @classmethod
    def overall_not_empty(cls, v):
        if not v:
            raise ValueError("overall must have at least one sentence")
        return v


# ---------------------------------------------------------------------------
# Validation entry point
# ---------------------------------------------------------------------------

class ValidationResult:
    def __init__(self, ok: bool, data=None, errors=None):
        self.ok = ok
        self.data = data      # ReadingOutput instance if ok
        self.errors = errors  # list of {field, message} if not ok

    def __repr__(self):
        if self.ok:
            return f"<ValidationResult ok=True>"
        return f"<ValidationResult ok=False errors={self.errors}>"


def validate_reading_output(raw: dict) -> ValidationResult:
    """
    Validate a dict (parsed from LLM JSON output) against ReadingOutput schema.

    Returns a ValidationResult:
      .ok      True if valid
      .data    ReadingOutput instance (if ok)
      .errors  list of {field, message} (if not ok)

    Usage:
        result = validate_reading_output(parsed_json)
        if not result.ok:
            print(result.errors)
        else:
            reading = result.data
    """
    try:
        data = ReadingOutput.model_validate(raw)
        return ValidationResult(ok=True, data=data)
    except ValidationError as e:
        errors = []
        for err in e.errors():
            # loc is a tuple like ('cards', 0, 'interpretation', 1, 'source')
            field = " → ".join(str(loc) for loc in err["loc"])
            errors.append({
                "field": field,
                "message": err["msg"],
                "invalid_value": err.get("input"),
            })
        return ValidationResult(ok=False, errors=errors)