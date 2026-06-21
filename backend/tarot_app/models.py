from django.db import models


class Card(models.Model):
    name = models.CharField(max_length=100)
    arcana = models.CharField(max_length=20)
    suit = models.CharField(max_length=50, blank=True)
    keywords = models.TextField()
    required_themes = models.JSONField(default=list)
    reversed_required_themes = models.JSONField(default=list)

    def __str__(self):
        return self.name


class Reading(models.Model):
    user_name = models.CharField(max_length=100)
    question = models.TextField()
    spread_type = models.CharField(max_length=50)
    cards = models.ManyToManyField(Card, through='ReadingCard')
    reading_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user_name} - {self.spread_type} - {self.created_at}"


class ReadingCard(models.Model):
    reading = models.ForeignKey(Reading, on_delete=models.CASCADE)
    card = models.ForeignKey(Card, on_delete=models.CASCADE)
    position = models.IntegerField()
    position_label = models.CharField(max_length=100)
    is_reversed = models.BooleanField(default=False)

    class Meta:
        ordering = ['position']


class VerificationReport(models.Model):
    STATUS_CHOICES = [('ok', 'OK'), ('needs_review', 'Needs Review')]

    reading = models.OneToOneField(
        Reading,
        on_delete=models.CASCADE,
        related_name='verification_report',
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='ok')
    claims = models.JSONField(default=list)
    precision = models.FloatField(null=True, blank=True)
    recall = models.FloatField(null=True, blank=True)
    f1 = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Report for Reading {self.reading_id} — {self.status}"


class LLMCallLog(models.Model):
    STATUS_CHOICES = [
        ('ok', 'OK'),
        ('recovered', 'Recovered after retry'),
        ('parse_failed', 'Parse Failed'),
    ]

    reading = models.ForeignKey(
        Reading,
        on_delete=models.CASCADE,
        related_name='llm_calls',
    )

    # Request
    prompt_version = models.CharField(max_length=20)
    full_prompt = models.TextField()
    model = models.CharField(max_length=100)
    rag_chunks_used = models.JSONField(default=list)

    # Response
    raw_response = models.TextField()
    attempt_number = models.IntegerField()
    validation_errors = models.JSONField(default=list, blank=True)
    final_status = models.CharField(max_length=20, choices=STATUS_CHOICES)

    # Performance
    latency_ms = models.IntegerField()
    input_tokens = models.IntegerField(null=True, blank=True)
    output_tokens = models.IntegerField(null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['reading_id', 'attempt_number']

    def __str__(self):
        return f"Reading {self.reading_id} attempt {self.attempt_number} — {self.final_status}"


class AgentDecisionLog(models.Model):
    """
    Records every decision point in the reading-generation pipeline.
    Designed so the pipeline can be audited end-to-end, and so future
    agentic behavior (self-correction, dynamic retries, tool routing)
    has a structured trace to reason over.
    """
    DECISION_POINTS = [
        ('rag_retrieval',    'RAG Retrieval'),
        ('prompt_build',     'Prompt Build'),
        ('llm_generation',   'LLM Generation'),
        ('schema_validation','Schema Validation'),
        ('retry_decision',   'Retry Decision'),
        ('judge_review',     'Judge Review'),
        ('final_status',     'Final Status Decision'),
    ]

    OUTCOME_CHOICES = [
        ('success', 'Success'),
        ('corrected', 'Corrected (retry/fallback used)'),
        ('failed', 'Failed'),
    ]

    reading = models.ForeignKey(
        Reading,
        on_delete=models.CASCADE,
        related_name='decision_logs',
    )

    decision_point = models.CharField(max_length=50, choices=DECISION_POINTS)
    sequence_number = models.IntegerField()

    # Core four fields
    input_data = models.JSONField(default=dict)
    decision = models.TextField()
    rationale = models.TextField()
    output_data = models.JSONField(default=dict)

    # Cost
    latency_ms = models.IntegerField(default=0)
    input_tokens = models.IntegerField(null=True, blank=True)
    output_tokens = models.IntegerField(null=True, blank=True)
    cost_usd = models.FloatField(default=0.0)

    outcome = models.CharField(max_length=20, choices=OUTCOME_CHOICES, default='success')

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['reading_id', 'sequence_number']

    def __str__(self):
        return f"Reading {self.reading_id} #{self.sequence_number} — {self.decision_point} ({self.outcome})"