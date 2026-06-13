from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('tarot_app', '0001_initial'),
    ]

    operations = [
        migrations.AddField(
            model_name='card',
            name='required_themes',
            field=models.JSONField(default=list),
        ),
        migrations.AddField(
            model_name='card',
            name='reversed_required_themes',
            field=models.JSONField(default=list),
        ),
    ]