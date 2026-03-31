"""Reliability sidecar artifacts.

Inspect `.eval` logs remain canonical. Sidecars are denormalized analysis records.
"""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock
from typing import Any

from pydantic import BaseModel, Field

from .identity import ReliabilityRunIdentity


def assert_canonical_eval_log_path(path: str | Path) -> None:
    """Validate that a log path points to Inspect native `.eval` logs."""
    if Path(path).suffix != ".eval":
        raise ValueError(
            f"expected Inspect .eval log path, received: {path!s}. "
            "Reliability execution truth must come from `.eval` logs."
        )


class ReliabilityRecord(BaseModel):
    """One normalized reliability row in sidecar JSONL format."""

    identity: ReliabilityRunIdentity
    outcome: dict[str, Any] = Field(default_factory=dict)
    behavior: dict[str, Any] = Field(default_factory=dict)
    resources: dict[str, Any] = Field(default_factory=dict)
    confidence: dict[str, Any] = Field(default_factory=dict)
    perturbation: dict[str, Any] = Field(default_factory=dict)
    safety: dict[str, Any] = Field(default_factory=dict)
    abstention: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SidecarWriter:
    """Thread-safe JSONL writer for reliability sidecars."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, record: ReliabilityRecord) -> None:
        line = record.model_dump_json()
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")


def load_sidecar_records(path: str | Path) -> list[ReliabilityRecord]:
    """Read all sidecar JSONL records from a path."""
    p = Path(path)
    if not p.exists():
        return []
    records: list[ReliabilityRecord] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            records.append(ReliabilityRecord.model_validate(data))
    return records
