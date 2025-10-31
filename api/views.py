"""
API views for token scam detection system.
Provides endpoints for job submission and result retrieval.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import AnalysisJob, Result


class AnalysisJobViewSet(viewsets.ReadOnlyModelViewSet):
    """
    ViewSet for viewing analysis jobs.
    """
    queryset = AnalysisJob.objects.all()
    serializer_class = None  # TODO: Create serializer

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """
        Get current status of analysis job.
        """
        job = self.get_object()
        return Response({
            'job_id': job.id,
            'status': job.status,
            'current_step': job.current_step,
            'error_message': job.error_message
        })
