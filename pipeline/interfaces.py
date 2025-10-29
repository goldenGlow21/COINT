"""
Standard interfaces for pipeline modules.
Defines base classes and data structures for consistent module integration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class PipelineData:
    """
    Standard data container passed between pipeline stages.
    Each stage appends its output while preserving previous results.
    """

    # Original input
    contract_address: str

    # Stage outputs (populated progressively)
    raw_data: Optional[Dict[str, Any]] = None
    processed_data: Optional[Dict[str, Any]] = None
    analysis_data: Optional[Dict[str, Any]] = None
    detection_data: Optional[Dict[str, Any]] = None

    # Metadata
    timestamp: datetime = None
    errors: list = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        if self.errors is None:
            self.errors = []


class PipelineModule(ABC):
    """
    Base class for all pipeline modules.
    Each module processes data and returns updated PipelineData.
    """

    @abstractmethod
    def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute module logic.

        Args:
            data: Current pipeline data

        Returns:
            Updated pipeline data with module's output

        Raises:
            Exception: On processing error
        """
        pass

    @abstractmethod
    def validate_input(self, data: PipelineData) -> bool:
        """
        Validate that input data meets module requirements.

        Args:
            data: Pipeline data to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    def get_name(self) -> str:
        """
        Get module name for logging/tracking.
        """
        return self.__class__.__name__


class DataCollector(PipelineModule):
    """
    Base interface for Ethereum data collection modules.
    Fetches raw contract data from blockchain/APIs.
    """

    @abstractmethod
    def collect(self, contract_address: str) -> Dict[str, Any]:
        """
        Collect raw data for given contract address.

        Args:
            contract_address: Ethereum contract address (0x...)

        Returns:
            Raw data dictionary
        """
        pass

    def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute collection and update pipeline data.
        """
        raw_data = self.collect(data.contract_address)
        data.raw_data = raw_data
        return data


class DataPreprocessor(PipelineModule):
    """
    Base interface for data preprocessing modules.
    Transforms raw data into standardized format.
    """

    @abstractmethod
    def preprocess(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Preprocess raw data.

        Args:
            raw_data: Raw collected data

        Returns:
            Preprocessed data dictionary
        """
        pass

    def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute preprocessing and update pipeline data.
        """
        processed_data = self.preprocess(data.raw_data)
        data.processed_data = processed_data
        return data


class CodeAnalyzer(PipelineModule):
    """
    Base interface for dynamic code analysis modules.
    Performs runtime behavior analysis on contract code.
    """

    @abstractmethod
    def analyze(self, processed_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze processed contract data.

        Args:
            processed_data: Preprocessed contract data

        Returns:
            Analysis results dictionary
        """
        pass

    def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute analysis and update pipeline data.
        """
        analysis_data = self.analyze(data.processed_data)
        data.analysis_data = analysis_data
        return data


class ThreatDetector(PipelineModule):
    """
    Base interface for rule-based threat detection modules.
    Applies detection rules to analysis results.
    """

    @abstractmethod
    def detect(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detect threats in analysis data.

        Args:
            analysis_data: Results from code analysis

        Returns:
            Detection results with threats, risk scores, etc.
        """
        pass

    def execute(self, data: PipelineData) -> PipelineData:
        """
        Execute detection and update pipeline data.
        """
        detection_data = self.detect(data.analysis_data)
        data.detection_data = detection_data
        return data
