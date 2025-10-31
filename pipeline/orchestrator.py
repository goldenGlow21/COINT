"""
Pipeline orchestrator for token scam detection workflow.
Coordinates all stages from data collection to final result aggregation.
"""

import logging
from typing import Optional, Dict, Any
from django.utils import timezone
from django.db import transaction

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Orchestrates the complete analysis pipeline according to workflow design.

    Workflow:
        1. Check if token already analyzed (result table)
        2. Collect token info (TokenInfo)
        3. Collect pair events (PairEvent)
        4. Collect holder info (HolderInfo)
        5. Preprocess data (HoneypotProcessedData, ExitProcessedData)
        6. Run honeypot dynamic analysis
        7. Run honeypot ML analysis
        8. Run exit scam ML analysis
        9. Aggregate results and save to result table
    """

    def __init__(self):
        """
        Initialize all pipeline adapters.
        """
        from .adapters import (
            TokenCollectorAdapter,
            PairCollectorAdapter,
            HolderCollectorAdapter,
            PreprocessorAdapter,
            HoneypotDynamicAnalyzerAdapter,
            HoneypotMLAnalyzerAdapter,
            ExitMLAnalyzerAdapter,
            ResultAggregatorAdapter,
        )

        self.token_collector = TokenCollectorAdapter()
        self.pair_collector = PairCollectorAdapter()
        self.holder_collector = HolderCollectorAdapter()
        self.preprocessor = PreprocessorAdapter()
        self.honeypot_da = HoneypotDynamicAnalyzerAdapter()
        self.honeypot_ml = HoneypotMLAnalyzerAdapter()
        self.exit_ml = ExitMLAnalyzerAdapter()
        self.aggregator = ResultAggregatorAdapter()

    def check_existing_result(self, token_addr: str) -> Optional['Result']:
        """
        Check if token has already been analyzed.

        Args:
            token_addr: Token contract address

        Returns:
            Result instance if exists, None otherwise
        """
        from api.models import Result

        try:
            return Result.objects.get(token_addr__iexact=token_addr)
        except Result.DoesNotExist:
            return None

    def execute(self, job_id: int) -> bool:
        """
        Execute complete pipeline for given job.

        Args:
            job_id: ID of AnalysisJob to process

        Returns:
            True if successful, False otherwise
        """
        from api.models import AnalysisJob

        try:
            job = AnalysisJob.objects.get(id=job_id)
        except AnalysisJob.DoesNotExist:
            logger.error(f"Job {job_id} not found")
            return False

        token_addr = job.token_addr
        logger.info(f"Starting pipeline for job {job_id}, token {token_addr}")

        try:
            # Step 1: Check if already analyzed
            existing_result = self.check_existing_result(token_addr)
            if existing_result:
                logger.info(f"Token {token_addr} already analyzed, using cached result")
                job.status = 'completed'
                job.completed_at = timezone.now()
                job.current_step = 'Using cached analysis result'
                job.save()
                return True

            # Step 2: Collect token metadata
            token_info = self._collect_token_info(job)

            # Step 3: Collect pair events
            self._collect_pair_events(job, token_info)

            # Step 4: Collect holder information
            self._collect_holder_info(job, token_info)

            # Step 5: Preprocess data
            self._preprocess_data(job, token_info)

            # Step 6: Run honeypot dynamic analysis
            honeypot_da_result = self._run_honeypot_da(job, token_info)

            # Step 7: Run honeypot ML analysis
            honeypot_ml_result = self._run_honeypot_ml(job, token_info)

            # Step 8: Run exit scam ML analysis
            exit_ml_result = self._run_exit_ml(job, token_info)

            # Step 9: Aggregate results and save
            self._aggregate_and_save_results(
                job,
                token_info,
                honeypot_da_result,
                honeypot_ml_result,
                exit_ml_result
            )

            # Mark job as completed
            job.status = 'completed'
            job.completed_at = timezone.now()
            job.current_step = 'Analysis completed successfully'
            job.save()

            logger.info(f"Pipeline completed successfully for job {job_id}")
            return True

        except Exception as e:
            logger.exception(f"Pipeline failed for job {job_id}: {str(e)}")
            job.status = 'failed'
            job.error_message = str(e)
            job.save()
            return False

    def _collect_token_info(self, job: 'AnalysisJob') -> 'TokenInfo':
        """
        Step 2: Collect token metadata and create TokenInfo record.
        """
        job.status = 'collecting_token'
        job.current_step = 'Collecting token metadata from blockchain'
        job.save()

        logger.info(f"Collecting token info for {job.token_addr}")

        try:
            # Call module to collect data
            token_data = self.token_collector.collect(job.token_addr)

            # Save to database
            token_info = self.token_collector.save_to_db(token_data)

            # Link job to token_info
            job.token_info = token_info
            job.save()

            logger.info(f"Token info collected, token_addr_idx={token_info.id}")
            return token_info

        except Exception as e:
            job.error_step = 'collecting_token'
            logger.error(f"Failed to collect token info: {e}")
            raise

    def _collect_pair_events(self, job: 'AnalysisJob', token_info: 'TokenInfo'):
        """
        Step 3: Collect pair event history.
        """
        job.status = 'collecting_pair'
        job.current_step = 'Collecting pair event history'
        job.save()

        logger.info(f"Collecting pair events for token {token_info.id}")

        try:
            # Call module to collect events
            events = self.pair_collector.collect(token_info)

            # Save to database
            count = self.pair_collector.save_to_db(token_info, events)

            logger.info(f"Collected {count} pair events")

        except Exception as e:
            job.error_step = 'collecting_pair'
            logger.error(f"Failed to collect pair events: {e}")
            raise

    def _collect_holder_info(self, job: 'AnalysisJob', token_info: 'TokenInfo'):
        """
        Step 4: Collect token holder information.
        """
        job.status = 'collecting_holder'
        job.current_step = 'Collecting token holder information'
        job.save()

        logger.info(f"Collecting holder info for token {token_info.id}")

        try:
            # Call module to collect holders
            holders = self.holder_collector.collect(token_info)

            # Save to database
            count = self.holder_collector.save_to_db(token_info, holders)

            logger.info(f"Collected {count} holders")

        except Exception as e:
            job.error_step = 'collecting_holder'
            logger.error(f"Failed to collect holder info: {e}")
            raise

    def _preprocess_data(self, job: 'AnalysisJob', token_info: 'TokenInfo'):
        """
        Step 5: Preprocess data for ML models.
        """
        job.status = 'preprocessing'
        job.current_step = 'Preprocessing data for analysis'
        job.save()

        logger.info(f"Preprocessing data for token {token_info.id}")

        try:
            # Generate honeypot features
            honeypot_features = self.preprocessor.process_for_honeypot(token_info)
            self.preprocessor.save_honeypot_to_db(token_info, honeypot_features)
            logger.info(f"Honeypot features generated")

            # Generate exit scam features
            exit_windows = self.preprocessor.process_for_exit(token_info)
            window_count = self.preprocessor.save_exit_to_db(token_info, exit_windows)
            logger.info(f"Exit scam features generated for {window_count} windows")

        except Exception as e:
            job.error_step = 'preprocessing'
            logger.error(f"Failed to preprocess data: {e}")
            raise

    def _run_honeypot_da(self, job: 'AnalysisJob', token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 6: Run honeypot dynamic analysis.
        """
        job.status = 'analyzing_honeypot_da'
        job.current_step = 'Running honeypot dynamic analysis'
        job.save()

        logger.info(f"Running honeypot DA for token {token_info.id}")

        try:
            processed_data = token_info.honeypot_processed
            result = self.honeypot_da.analyze(token_info, processed_data)
            logger.info(f"Honeypot DA completed: {result.get('is_honeypot', False)}")
            return result

        except Exception as e:
            job.error_step = 'analyzing_honeypot_da'
            logger.error(f"Failed to run honeypot DA: {e}")
            raise

    def _run_honeypot_ml(self, job: 'AnalysisJob', token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 7: Run honeypot ML analysis.
        """
        job.status = 'analyzing_honeypot_ml'
        job.current_step = 'Running honeypot ML analysis'
        job.save()

        logger.info(f"Running honeypot ML for token {token_info.id}")

        try:
            processed_data = token_info.honeypot_processed
            result = self.honeypot_ml.predict(processed_data)
            logger.info(f"Honeypot ML completed: {result.get('is_honeypot', False)}")
            return result

        except Exception as e:
            job.error_step = 'analyzing_honeypot_ml'
            logger.error(f"Failed to run honeypot ML: {e}")
            raise

    def _run_exit_ml(self, job: 'AnalysisJob', token_info: 'TokenInfo') -> Dict[str, Any]:
        """
        Step 8: Run exit scam ML analysis.
        """
        job.status = 'analyzing_exit_ml'
        job.current_step = 'Running exit scam ML analysis'
        job.save()

        logger.info(f"Running exit ML for token {token_info.id}")

        try:
            result = self.exit_ml.predict(token_info)
            logger.info(f"Exit ML completed: {result.get('is_exit_scam', False)}")
            return result

        except Exception as e:
            job.error_step = 'analyzing_exit_ml'
            logger.error(f"Failed to run exit ML: {e}")
            raise

    def _aggregate_and_save_results(
        self,
        job: 'AnalysisJob',
        token_info: 'TokenInfo',
        honeypot_da_result: Dict[str, Any],
        honeypot_ml_result: Dict[str, Any],
        exit_ml_result: Dict[str, Any]
    ):
        """
        Step 9: Aggregate all results and save to result table.
        """
        job.status = 'aggregating'
        job.current_step = 'Aggregating analysis results'
        job.save()

        logger.info(f"Aggregating results for token {token_info.id}")

        try:
            # Aggregate results
            aggregated_data = self.aggregator.aggregate(
                token_info,
                honeypot_da_result,
                honeypot_ml_result,
                exit_ml_result
            )

            # Save to result table
            self.aggregator.save_to_db(token_info, aggregated_data)

            logger.info(f"Results aggregated and saved, risk_score={aggregated_data['risk_score']}")

        except Exception as e:
            job.error_step = 'aggregating'
            logger.error(f"Failed to aggregate results: {e}")
            raise

    @classmethod
    def execute_async(cls, job_id: int):
        """
        Execute pipeline asynchronously.
        TODO: Integrate with Celery for true async execution.

        Args:
            job_id: ID of AnalysisJob to process
        """
        orchestrator = cls()
        return orchestrator.execute(job_id)
