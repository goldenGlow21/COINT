"""
Pipeline orchestrator for coordinating analysis workflow.
Manages execution flow across all pipeline stages.
"""

import logging
from typing import Optional
from datetime import datetime

from django.utils import timezone
from api.models import AnalysisJob, AnalysisResult, DetectedIssue

from .interfaces import PipelineData
from .adapters import (
    CollectorAdapter,
    PreprocessorAdapter,
    AnalyzerAdapter,
    DetectorAdapter,
)

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete analysis pipeline.
    Executes stages sequentially and handles errors gracefully.
    """

    def __init__(self):
        """
        Initialize all pipeline modules.
        """
        self.collector = CollectorAdapter()
        self.preprocessor = PreprocessorAdapter()
        self.analyzer = AnalyzerAdapter()
        self.detector = DetectorAdapter()

    def execute(self, job_id: int) -> bool:
        """
        Execute complete pipeline for given job.

        Args:
            job_id: ID of AnalysisJob to process

        Returns:
            True if successful, False otherwise
        """
        try:
            job = AnalysisJob.objects.get(id=job_id)
        except AnalysisJob.DoesNotExist:
            logger.error(f"Job {job_id} not found")
            return False

        logger.info(f"Starting pipeline for job {job_id}")

        # Initialize pipeline data
        data = PipelineData(contract_address=job.contract_address)

        try:
            # Stage 1: Data Collection
            data = self._execute_stage(
                job, 'collecting', self.collector, data
            )

            # Stage 2: Preprocessing
            data = self._execute_stage(
                job, 'preprocessing', self.preprocessor, data
            )

            # Stage 3: Dynamic Analysis
            data = self._execute_stage(
                job, 'analyzing', self.analyzer, data
            )

            # Stage 4: Threat Detection
            data = self._execute_stage(
                job, 'detecting', self.detector, data
            )

            # Save results to database
            self._save_results(job, data)

            # Mark job as completed
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.save()

            logger.info(f"Pipeline completed for job {job_id}")
            return True

        except Exception as e:
            # Handle pipeline failure
            logger.exception(f"Pipeline failed for job {job_id}: {str(e)}")
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
            return False

    def _execute_stage(self, job, status_name, module, data):
        """
        Execute a single pipeline stage with error handling.

        Args:
            job: AnalysisJob instance
            status_name: Status to set during execution
            module: Pipeline module to execute
            data: Current pipeline data

        Returns:
            Updated pipeline data

        Raises:
            Exception: If stage execution fails
        """
        logger.info(f"Job {job.id}: Starting {module.get_name()}")

        # Update job status
        job.status = status_name
        job.save()

        try:
            # Validate input
            if not module.validate_input(data):
                raise ValueError(f"Invalid input for {module.get_name()}")

            # Execute module
            data = module.execute(data)

            logger.info(f"Job {job.id}: Completed {module.get_name()}")
            return data

        except Exception as e:
            job.error_step = module.get_name()
            raise Exception(
                f"{module.get_name()} failed: {str(e)}"
            ) from e

    def _save_results(self, job, data: PipelineData):
        """
        Save pipeline results to database.

        Args:
            job: AnalysisJob instance
            data: Complete pipeline data
        """
        # Extract detection metadata
        detection_data = data.detection_data or {}
        risk_score = detection_data.get('risk_score', 0.0)
        threat_level = detection_data.get('threat_level', 'safe')
        issues = detection_data.get('issues', [])

        # Create or update result
        result, created = AnalysisResult.objects.update_or_create(
            job=job,
            defaults={
                'raw_data': data.raw_data or {},
                'processed_data': data.processed_data or {},
                'analysis_data': data.analysis_data or {},
                'detection_data': detection_data,
                'risk_score': risk_score,
                'threat_level': threat_level,
                'issues_count': len(issues),
            }
        )

        # Create issue records
        if issues:
            # Delete existing issues for this result
            DetectedIssue.objects.filter(result=result).delete()

            # Create new issue records
            for issue_data in issues:
                DetectedIssue.objects.create(
                    result=result,
                    title=issue_data.get('title', 'Unknown Issue'),
                    description=issue_data.get('description', ''),
                    severity=issue_data.get('severity', 'info'),
                    category=issue_data.get('category', 'general'),
                    code_location=issue_data.get('code_location'),
                    confidence=issue_data.get('confidence', 1.0),
                    additional_data=issue_data.get('additional_data'),
                )

        logger.info(
            f"Saved results for job {job.id}: "
            f"{len(issues)} issues, risk_score={risk_score}"
        )

    @classmethod
    def execute_async(cls, job_id: int):
        """
        Execute pipeline asynchronously.

        TODO: Implement async execution with Celery or similar.
        Example:
            from celery import shared_task

            @shared_task
            def run_pipeline(job_id):
                orchestrator = PipelineOrchestrator()
                return orchestrator.execute(job_id)

            run_pipeline.delay(job_id)

        For now, executes synchronously.
        """
        orchestrator = cls()
        return orchestrator.execute(job_id)
