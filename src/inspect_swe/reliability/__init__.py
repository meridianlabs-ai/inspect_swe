"""Reliability evaluation primitives for Inspect SWE.

This package intentionally treats Inspect `.eval` logs as canonical run records.
Sidecar JSONL artifacts are derived projections used for reliability analysis.
"""

from .artifacts import ReliabilityRecord, SidecarWriter, load_sidecar_records
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
from .hooks import (
    ReliabilityHookConfig,
    assert_reliability_hooks_active,
    configure_reliability_hooks,
    disable_reliability_hooks,
)
from .identity import ReliabilityRunIdentity
from .orchestrator import PhaseShard, build_phase_shards, preflight_reliability_spec
from .spec import PhaseName, ReliabilitySpec
from .telemetry import TelemetryCoverageReport, assess_sidecar_coverage

__all__ = [
    "BaselineExecutionError",
    "BaselinePhaseConfig",
    "BaselinePhaseResult",
    "BaselineRepeatResult",
    "PhaseShard",
    "ConcurrencyPolicyError",
    "OrchestratorConcurrency",
    "PhaseName",
    "ReliabilityHookConfig",
    "ReliabilityRecord",
    "ReliabilityRunIdentity",
    "ReliabilitySpec",
    "SidecarWriter",
    "TelemetryCoverageReport",
    "assess_sidecar_coverage",
    "build_phase_shards",
    "run_baseline_phase",
    "assert_reliability_hooks_active",
    "configure_reliability_hooks",
    "disable_reliability_hooks",
    "load_sidecar_records",
    "preflight_reliability_spec",
    "validate_orchestrator_policy",
]
