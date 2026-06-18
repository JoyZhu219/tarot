from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tarot_app', '0004_llmcalllog'),
    ]

    operations = [
        migrations.CreateModel(
            name='AgentDecisionLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name='ID')),

                # Which decision point in the pipeline
                ('decision_point', models.CharField(
                    max_length=50,
                    choices=[
                        ('rag_retrieval',   'RAG Retrieval'),
                        ('prompt_build',     'Prompt Build'),
                        ('llm_generation',   'LLM Generation'),
                        ('schema_validation','Schema Validation'),
                        ('retry_decision',   'Retry Decision'),
                        ('judge_review',     'Judge Review'),
                        ('final_status',     'Final Status Decision'),
                    ],
                )),

                # Sequencing within a reading's decision chain
                ('sequence_number', models.IntegerField()),

                # The four core fields requested
                ('input_data',  models.JSONField(default=dict)),
                ('decision',    models.TextField()),
                ('rationale',   models.TextField()),
                ('output_data', models.JSONField(default=dict)),

                # Cost tracking (zero for non-LLM decision points)
                ('latency_ms',    models.IntegerField(default=0)),
                ('input_tokens',  models.IntegerField(null=True, blank=True)),
                ('output_tokens', models.IntegerField(null=True, blank=True)),
                ('cost_usd',      models.FloatField(default=0.0)),

                # Outcome classification — did this decision succeed,
                # need correction, or fail outright?
                ('outcome', models.CharField(
                    max_length=20,
                    choices=[
                        ('success', 'Success'),
                        ('corrected', 'Corrected (retry/fallback used)'),
                        ('failed', 'Failed'),
                    ],
                    default='success',
                )),

                ('created_at', models.DateTimeField(auto_now_add=True)),

                ('reading', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='decision_logs',
                    to='tarot_app.reading',
                )),
            ],
            options={'ordering': ['reading_id', 'sequence_number']},
        ),
    ]