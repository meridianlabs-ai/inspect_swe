"""Reliability evaluation primitives for Inspect SWE.

This package treats Inspect `.eval` logs as canonical run records.
"""

from .baseline import (
    BaselineExecutionError,
    BaselinePhaseConfig,
    BaselinePhaseResult,
    BaselineRepeatResult,
    run_baseline_phase,
)
from .concurrency import (
    ConcurrencyPolicyError,
    OrchestratorConcurrency,
    validate_orchestrator_policy,
)
from .eval_view import BaselineSampleView, extract_baseline_sample_views
from .spec import PhaseName, ReliabilitySpec
from .telemetry import TelemetryCoverageReport, assess_eval_coverage

__all__ = [
    "BaselineExecutionError",
    "BaselinePhaseConfig",
    "BaselinePhaseResult",
    "BaselineRepeatResult",
    "ConcurrencyPolicyError",
    "OrchestratorConcurrency",
    "PhaseName",
    "BaselineSampleView",
    "ReliabilitySpec",
    "TelemetryCoverageReport",
    "assess_eval_coverage",
    "extract_baseline_sample_views",
    "run_baseline_phase",
    "validate_orchestrator_policy",
]
