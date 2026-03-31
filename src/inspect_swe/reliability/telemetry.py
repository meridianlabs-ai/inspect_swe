"""Telemetry coverage and completeness checks."""

from __future__ import annotations

from inspect_ai.log import EvalLog
from pydantic import BaseModel

from .artifacts import ReliabilityRecord


class TelemetryCoverageReport(BaseModel):
    """Coverage report for a single eval log vs sidecar records."""

    expected_samples: int
    observed_records: int
    missing_sample_uuids: list[str]
    duplicate_identity_keys: list[str]
    identity_warning_count: int

    @property
    def complete(self) -> bool:
        return (
            self.expected_samples == self.observed_records
            and not self.missing_sample_uuids
            and not self.duplicate_identity_keys
        )


def assess_sidecar_coverage(
    eval_log: EvalLog, records: list[ReliabilityRecord]
) -> TelemetryCoverageReport:
    """Assess whether sidecar records cover all eval samples."""
    samples = eval_log.samples or []
    expected_uuids = {sample.uuid for sample in samples if sample.uuid is not None}

    observed_uuids: set[str] = set()
    duplicate_identity_keys: list[str] = []
    seen_identity_keys: set[str] = set()
    identity_warning_count = 0

    for record in records:
        key = record.identity.key()
        if key in seen_identity_keys:
            duplicate_identity_keys.append(key)
        else:
            seen_identity_keys.add(key)

        observed_uuids.add(record.identity.sample_uuid)

        warning = record.metadata.get("reliability_identity_warning")
        if warning is not None:
            identity_warning_count += 1

    missing_sample_uuids = sorted(list(expected_uuids - observed_uuids))
    return TelemetryCoverageReport(
        expected_samples=len(samples),
        observed_records=len(records),
        missing_sample_uuids=missing_sample_uuids,
        duplicate_identity_keys=duplicate_identity_keys,
        identity_warning_count=identity_warning_count,
    )
