from django.urls import path
from .views import (
    CardListView, SpreadListView, GenerateReadingView,
    EvaluateReadingView, VerifyReadingView, JudgeReportView,
    LLMCostsView, RecommendSpreadsView,
)

urlpatterns = [
    path('cards/', CardListView.as_view()),
    path('spreads/', SpreadListView.as_view()),
    path('reading/', GenerateReadingView.as_view()),
    path('reading/<int:reading_id>/evaluate/', EvaluateReadingView.as_view()),
    path('reading/<int:reading_id>/verify/', VerifyReadingView.as_view()),
    path('reading/<int:reading_id>/judge/', JudgeReportView.as_view()),
    path('llm-costs/', LLMCostsView.as_view()),
    path('recommend-spreads/', RecommendSpreadsView.as_view()),
]
