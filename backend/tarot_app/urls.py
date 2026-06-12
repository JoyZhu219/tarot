from django.urls import path
from .views import CardListView, SpreadListView, GenerateReadingView

urlpatterns = [
    path('cards/', CardListView.as_view()),
    path('spreads/', SpreadListView.as_view()),
    path('reading/', GenerateReadingView.as_view()),
]
