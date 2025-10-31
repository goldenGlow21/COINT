"""
URL routing for API endpoints.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import AnalysisJobViewSet

router = DefaultRouter()
router.register(r'jobs', AnalysisJobViewSet, basename='job')

urlpatterns = [
    path('', include(router.urls)),
]
