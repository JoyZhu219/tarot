from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tarot_app', '0003_verificationreport'),
    ]

    operations = [
        migrations.CreateModel(
            name='LLMCallLog',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name='ID')),
                ('prompt_version', models.CharField(max_length=20)),
                ('full_prompt', models.TextField()),
                ('model', models.CharField(max_length=100)),
                ('rag_chunks_used', models.JSONField(default=list)),
                ('raw_response', models.TextField()),
                ('attempt_number', models.IntegerField()),
                ('validation_errors', models.JSONField(blank=True, default=list)),
                ('final_status', models.CharField(
                    max_length=20,
                    choices=[('ok', 'OK'), ('recovered', 'Recovered after retry'),
                            ('parse_failed', 'Parse Failed')],
                )),
                ('latency_ms', models.IntegerField()),
                ('input_tokens', models.IntegerField(blank=True, null=True)),
                ('output_tokens', models.IntegerField(blank=True, null=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('reading', models.ForeignKey(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='llm_calls',
                    to='tarot_app.reading',
                )),
            ],
            options={'ordering': ['reading_id', 'attempt_number']},
        ),
    ]