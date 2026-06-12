from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = []

    operations = [
        migrations.CreateModel(
            name='Card',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=100)),
                ('arcana', models.CharField(max_length=20)),
                ('suit', models.CharField(blank=True, max_length=50)),
                ('keywords', models.TextField()),
            ],
        ),
        migrations.CreateModel(
            name='Reading',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('user_name', models.CharField(max_length=100)),
                ('question', models.TextField()),
                ('spread_type', models.CharField(max_length=50)),
                ('reading_text', models.TextField()),
                ('created_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
        migrations.CreateModel(
            name='ReadingCard',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('position', models.IntegerField()),
                ('position_label', models.CharField(max_length=100)),
                ('is_reversed', models.BooleanField(default=False)),
                ('card', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='tarot_app.card')),
                ('reading', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to='tarot_app.reading')),
            ],
            options={'ordering': ['position']},
        ),
        migrations.AddField(
            model_name='reading',
            name='cards',
            field=models.ManyToManyField(through='tarot_app.ReadingCard', to='tarot_app.card'),
        ),
    ]