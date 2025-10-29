"""
REST API serializers for analysis data.
Handles JSON conversion for API requests and responses.
"""

from rest_framework import serializers
from .models import AnalysisJob, AnalysisResult, DetectedIssue


class DetectedIssueSerializer(serializers.ModelSerializer):
    """
    Serializer for individual detected security issues.
    """

    class Meta:
        model = DetectedIssue
        fields = [
            'id',
            'title',
            'description',
            'severity',
            'category',
            'code_location',
            'confidence',
            'additional_data',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class AnalysisResultSerializer(serializers.ModelSerializer):
    """
    Serializer for complete analysis results.
    Includes nested issue details.
    """

    issues = DetectedIssueSerializer(many=True, read_only=True)

    class Meta:
        model = AnalysisResult
        fields = [
            'id',
            'raw_data',
            'processed_data',
            'analysis_data',
            'detection_data',
            'risk_score',
            'threat_level',
            'issues_count',
            'issues',
            'created_at',
        ]
        read_only_fields = ['id', 'created_at']


class AnalysisJobSerializer(serializers.ModelSerializer):
    """
    Serializer for analysis job tracking.
    Optionally includes full result data.
    """

    result = AnalysisResultSerializer(read_only=True)

    class Meta:
        model = AnalysisJob
        fields = [
            'id',
            'contract_address',
            'contract_name',
            'status',
            'created_at',
            'updated_at',
            'completed_at',
            'error_message',
            'error_step',
            'result',
        ]
        read_only_fields = [
            'id',
            'status',
            'created_at',
            'updated_at',
            'completed_at',
            'error_message',
            'error_step',
            'result',
        ]


class AnalysisJobListSerializer(serializers.ModelSerializer):
    """
    Lightweight serializer for job listings.
    Excludes heavy nested data for better performance.
    """

    class Meta:
        model = AnalysisJob
        fields = [
            'id',
            'contract_address',
            'contract_name',
            'status',
            'created_at',
            'updated_at',
            'completed_at',
        ]
        read_only_fields = [
            'id',
            'status',
            'created_at',
            'updated_at',
            'completed_at',
        ]


class AnalysisJobCreateSerializer(serializers.ModelSerializer):
    """
    Serializer for creating new analysis jobs.
    Only requires contract address to start.
    """

    class Meta:
        model = AnalysisJob
        fields = [
            'contract_address',
            'contract_name',
        ]

    def validate_contract_address(self, value):
        """
        Validate Ethereum contract address format.
        """
        if not value.startswith('0x'):
            raise serializers.ValidationError(
                "Contract address must start with '0x'"
            )
        if len(value) != 42:
            raise serializers.ValidationError(
                "Contract address must be 42 characters (0x + 40 hex chars)"
            )
        return value.lower()
