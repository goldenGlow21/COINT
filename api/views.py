"""
REST API views for contract analysis operations.
Provides endpoints for creating jobs, checking status, and retrieving results.
"""

from rest_framework import viewsets, status
from rest_framework.decorators import action
from rest_framework.response import Response
from django.shortcuts import get_object_or_404

from .models import AnalysisJob, AnalysisResult, DetectedIssue
from .serializers import (
    AnalysisJobSerializer,
    AnalysisJobListSerializer,
    AnalysisJobCreateSerializer,
    AnalysisResultSerializer,
    DetectedIssueSerializer,
)


class AnalysisJobViewSet(viewsets.ModelViewSet):
    """
    ViewSet for managing analysis jobs.

    Endpoints:
        GET    /api/jobs/              - List all jobs
        POST   /api/jobs/              - Create new analysis job
        GET    /api/jobs/{id}/         - Get job details
        GET    /api/jobs/{id}/status/  - Get job status only
        GET    /api/jobs/{id}/result/  - Get job result
        DELETE /api/jobs/{id}/         - Delete job
    """

    queryset = AnalysisJob.objects.all()

    def get_serializer_class(self):
        """
        Use different serializers for different actions.
        """
        if self.action == 'list':
            return AnalysisJobListSerializer
        elif self.action == 'create':
            return AnalysisJobCreateSerializer
        return AnalysisJobSerializer

    def create(self, request, *args, **kwargs):
        """
        Create new analysis job and trigger pipeline execution.
        """
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        job = serializer.save()

        # TODO: Trigger async pipeline execution here
        # Example: from pipeline.orchestrator import PipelineOrchestrator
        # PipelineOrchestrator.execute_async(job.id)

        return Response(
            AnalysisJobSerializer(job).data,
            status=status.HTTP_201_CREATED
        )

    @action(detail=True, methods=['get'])
    def status(self, request, pk=None):
        """
        Get current job status without full details.
        Lightweight endpoint for polling.
        """
        job = self.get_object()
        return Response({
            'id': job.id,
            'status': job.status,
            'updated_at': job.updated_at,
            'error_message': job.error_message,
        })

    @action(detail=True, methods=['get'])
    def result(self, request, pk=None):
        """
        Get analysis result for completed job.
        Returns 404 if job not yet completed.
        """
        job = self.get_object()

        if job.status != 'completed':
            return Response(
                {
                    'error': 'Job not completed yet',
                    'status': job.status
                },
                status=status.HTTP_404_NOT_FOUND
            )

        if not hasattr(job, 'result'):
            return Response(
                {'error': 'Result not found'},
                status=status.HTTP_404_NOT_FOUND
            )

        serializer = AnalysisResultSerializer(job.result)
        return Response(serializer.data)


class AnalysisResultViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only ViewSet for analysis results.

    Endpoints:
        GET /api/results/     - List all results
        GET /api/results/{id}/ - Get specific result
    """

    queryset = AnalysisResult.objects.select_related('job').prefetch_related('issues')
    serializer_class = AnalysisResultSerializer


class DetectedIssueViewSet(viewsets.ReadOnlyModelViewSet):
    """
    Read-only ViewSet for detected issues.

    Endpoints:
        GET /api/issues/              - List all issues
        GET /api/issues/{id}/         - Get specific issue
        GET /api/issues/?result={id}  - Filter by result ID
    """

    queryset = DetectedIssue.objects.select_related('result__job')
    serializer_class = DetectedIssueSerializer

    def get_queryset(self):
        """
        Optionally filter issues by result ID.
        """
        queryset = super().get_queryset()
        result_id = self.request.query_params.get('result', None)

        if result_id is not None:
            queryset = queryset.filter(result_id=result_id)

        return queryset
