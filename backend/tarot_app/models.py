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