"""
URL routing for API endpoints.
"""

from django.urls import path, include
from rest_framework.routers import DefaultRouter
from .views import (
    AnalysisJobViewSet,
    AnalysisResultViewSet,
    DetectedIssueViewSet,
)

router = DefaultRouter()
router.register(r'jobs', AnalysisJobViewSet, basename='job')
router.register(r'results', AnalysisResultViewSet, basename='result')
router.register(r'issues', DetectedIssueViewSet, basename='issue')

urlpatterns = [
    path('', include(router.urls)),
]
