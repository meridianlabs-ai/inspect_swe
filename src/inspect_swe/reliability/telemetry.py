"""Eval-native telemetry coverage and completeness checks."""

from __future__ import annotations

from typing import Any

from inspect_ai.log import EvalLog
from pydantic import BaseModel

from .eval_view import extract_baseline_sample_views


class TelemetryCoverageReport(BaseModel):
    """Coverage report for a single eval log."""

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
            and self.identity_warning_count == 0
        )


def assess_eval_coverage(
    eval_log: EvalLog,
    *,
    expected_phase: str,
    expected_agent: str,
    expected_repeat_id: int,
    strict_identity_tags: bool,
) -> TelemetryCoverageReport:
    """Assess whether eval samples satisfy reliability telemetry contracts."""
    samples = eval_log.samples or []
    views, warnings = extract_baseline_sample_views(
        eval_log,
        expected_phase=expected_phase,
        expected_agent=expected_agent,
        expected_repeat_id=expected_repeat_id,
        strict_identity_tags=strict_identity_tags,
    )

    missing_sample_uuids = sorted(
        {
            _sample_uuid_text(sample.uuid)
            for sample in samples
            if not _sample_uuid_text(sample.uuid)
        }
    )

    observed_uuids = {
        view.sample_uuid for view in views if isinstance(view.sample_uuid, str) and view.sample_uuid
    }
    duplicate_identity_keys: list[str] = []
    seen_identity_keys: set[str] = set()
    for view in views:
        key = view.identity_key()
        if key in seen_identity_keys:
            duplicate_identity_keys.append(key)
        else:
            seen_identity_keys.add(key)
    expected_uuids = {
        _sample_uuid_text(sample.uuid)
        for sample in samples
        if _sample_uuid_text(sample.uuid)
    }
    missing_sample_uuids.extend(sorted(list(expected_uuids - observed_uuids)))
    missing_sample_uuids = sorted(set(missing_sample_uuids))
    return TelemetryCoverageReport(
        expected_samples=len(samples),
        observed_records=len(views),
        missing_sample_uuids=missing_sample_uuids,
        duplicate_identity_keys=duplicate_identity_keys,
        identity_warning_count=warnings,
    )


def _sample_uuid_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""
