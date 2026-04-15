"""Shared utilities for Codex rollout processing."""

from datetime import datetime


def parse_timestamp(ts_str: str | None) -> datetime | None:
    """Parse an ISO timestamp string to datetime."""
    if not ts_str:
        return None
    try:
        return datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
    except ValueError:
        return None
