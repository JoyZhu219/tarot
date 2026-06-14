from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('tarot_app', '0002_card_themes'),
    ]

    operations = [
        migrations.CreateModel(
            name='VerificationReport',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True,
                                           serialize=False, verbose_name='ID')),
                ('reading', models.OneToOneField(
                    on_delete=django.db.models.deletion.CASCADE,
                    related_name='verification_report',
                    to='tarot_app.reading',
                )),
                ('status', models.CharField(
                    max_length=20,
                    choices=[('ok', 'OK'), ('needs_review', 'Needs Review')],
                    default='ok',
                )),
                ('claims', models.JSONField(default=list)),
                ('precision', models.FloatField(null=True, blank=True)),
                ('recall', models.FloatField(null=True, blank=True)),
                ('f1', models.FloatField(null=True, blank=True)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]