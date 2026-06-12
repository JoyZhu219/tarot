from django.urls import path, include

urlpatterns = [
    path('api/', include('tarot_app.urls')),
]
