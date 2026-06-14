from django.urls import path
from .views import CardListView, SpreadListView, GenerateReadingView, EvaluateReadingView, VerifyReadingView

urlpatterns = [
    path('cards/', CardListView.as_view()),
    path('spreads/', SpreadListView.as_view()),
    path('reading/', GenerateReadingView.as_view()),
    path('reading/<int:reading_id>/evaluate/', EvaluateReadingView.as_view()),
    path('reading/<int:reading_id>/verify/', VerifyReadingView.as_view()),
]
