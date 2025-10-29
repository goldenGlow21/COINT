"""
Database models for storing contract analysis results.
"""

from django.db import models
from django.utils import timezone


class AnalysisJob(models.Model):
    """
    Represents a single analysis job for an Ethereum contract.
    Tracks the entire pipeline execution from data collection to detection.
    """

    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('collecting', 'Collecting Data'),
        ('preprocessing', 'Preprocessing'),
        ('analyzing', 'Analyzing'),
        ('detecting', 'Running Detection'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]

    # Contract information
    contract_address = models.CharField(max_length=42, db_index=True)
    contract_name = models.CharField(max_length=255, blank=True, null=True)

    # Job metadata
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(default=timezone.now)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(blank=True, null=True)

    # Error tracking
    error_message = models.TextField(blank=True, null=True)
    error_step = models.CharField(max_length=50, blank=True, null=True)

    class Meta:
        ordering = ['-created_at']
        indexes = [
            models.Index(fields=['status', '-created_at']),
        ]

    def __str__(self):
        return f"Job {self.id}: {self.contract_address} ({self.status})"


class AnalysisResult(models.Model):
    """
    Stores the final analysis results after all pipeline stages.
    Contains processed data, analysis findings, and detection results.
    """

    job = models.OneToOneField(
        AnalysisJob,
        on_delete=models.CASCADE,
        related_name='result'
    )

    # Raw data (JSON)
    raw_data = models.JSONField(
        help_text="Original collected data from Ethereum"
    )

    # Preprocessed data (JSON)
    processed_data = models.JSONField(
        help_text="Data after preprocessing stage"
    )

    # Dynamic analysis results (JSON)
    analysis_data = models.JSONField(
        help_text="Results from dynamic code analysis"
    )

    # Rule-based detection results (JSON)
    detection_data = models.JSONField(
        help_text="Final detection results with risk scores and patterns"
    )

    # Summary fields for quick access
    risk_score = models.FloatField(
        default=0.0,
        help_text="Overall risk score (0-100)"
    )
    threat_level = models.CharField(
        max_length=20,
        choices=[
            ('safe', 'Safe'),
            ('low', 'Low Risk'),
            ('medium', 'Medium Risk'),
            ('high', 'High Risk'),
            ('critical', 'Critical'),
        ],
        default='safe'
    )

    # Detected issues count
    issues_count = models.IntegerField(default=0)

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"Result for Job {self.job.id}: {self.threat_level}"


class DetectedIssue(models.Model):
    """
    Individual security issues or patterns detected during analysis.
    Multiple issues can be associated with a single analysis result.
    """

    SEVERITY_CHOICES = [
        ('info', 'Informational'),
        ('low', 'Low'),
        ('medium', 'Medium'),
        ('high', 'High'),
        ('critical', 'Critical'),
    ]

    result = models.ForeignKey(
        AnalysisResult,
        on_delete=models.CASCADE,
        related_name='issues'
    )

    # Issue details
    title = models.CharField(max_length=255)
    description = models.TextField()
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    category = models.CharField(
        max_length=100,
        help_text="Issue category (e.g., 'reentrancy', 'overflow', etc.)"
    )

    # Location in code
    code_location = models.JSONField(
        blank=True,
        null=True,
        help_text="Location in contract code where issue was found"
    )

    # Additional metadata
    confidence = models.FloatField(
        default=1.0,
        help_text="Detection confidence (0-1)"
    )
    additional_data = models.JSONField(
        blank=True,
        null=True,
        help_text="Any additional detection metadata"
    )

    created_at = models.DateTimeField(default=timezone.now)

    class Meta:
        ordering = ['-severity', '-confidence']

    def __str__(self):
        return f"{self.severity.upper()}: {self.title}"
