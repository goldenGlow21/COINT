"""
API serializers for token scam detection system.
"""

from rest_framework import serializers
from .models import AnalysisJob, Result, TokenInfo


class AnalysisJobSerializer(serializers.ModelSerializer):
    """
    Serializer for AnalysisJob model.
    """
    class Meta:
        model = AnalysisJob
        fields = ['id', 'token_addr', 'status', 'current_step', 'created_at', 'updated_at']


class ResultSerializer(serializers.ModelSerializer):
    """
    Serializer for Result model.
    """
    class Meta:
        model = Result
        fields = ['id', 'token_addr', 'risk_score', 'scam_types', 'victim_insights', 'created_at']
