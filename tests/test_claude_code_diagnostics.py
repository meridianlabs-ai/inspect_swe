from typing import Any

import pytest
from inspect_ai.util import Store
from inspect_swe._claude_code._events.diagnostics import ClaudeCodeDiagnostics


def test_result_summary_keeps_only_structural_metadata() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    diagnostic.observe(
        {
            "type": "result",
            "subtype": "error_during_execution",
            "is_error": True,
            "duration_ms": 123,
            "num_turns": 4,
            "errors": [
                {"type": "api_error", "code": "rate_limit_error", "message": "SECRET"}
            ],
            "result": "SECRET",
            "session_id": "SECRET",
            "usage": {"SECRET": 123},
        }
    )
    assert diagnostic.result_records == 1
    assert diagnostic.result_subtype == "error_during_execution"
    assert diagnostic.is_error is True
    assert diagnostic.duration_ms == 123
    assert diagnostic.num_turns == 4
    assert diagnostic.error_categories == ["api_error", "rate_limit_error"]
    assert "SECRET" not in diagnostic.model_dump_json()


@pytest.mark.parametrize(
    "error", ["SECRET", {"type": "SECRET", "code": "SECRET"}, 123, None]
)
def test_unknown_errors_are_present_without_their_content(error: Any) -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    diagnostic.observe({"type": "result", "errors": [error], "subtype": "SECRET"})
    assert diagnostic.unknown_error_present
    assert diagnostic.error_entries == 1
    assert diagnostic.result_subtype == "unknown"
    assert "SECRET" not in diagnostic.model_dump_json()


def test_success_and_missing_result_are_distinguishable() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    assert not diagnostic.result_available
    diagnostic.observe({"type": "result", "subtype": "success", "is_error": False})
    assert diagnostic.result_available
    assert diagnostic.is_error is False
    assert not diagnostic.unknown_error_present


def test_error_without_result_is_captured() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    diagnostic.observe({"type": "error", "error": {"type": "authentication_error"}})
    assert diagnostic.error_records == 1
    assert not diagnostic.result_available
    assert diagnostic.error_categories == ["authentication_error"]


def test_malformed_and_unrelated_records_do_not_become_diagnostics() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    records: list[object] = [None, [], 3, "SECRET"]
    for raw in records:
        diagnostic.observe(raw)
    diagnostic.observe({"type": "assistant", "error": "SECRET"})
    assert diagnostic.malformed_records == 4
    assert diagnostic.error_entries == 0
    assert "SECRET" not in diagnostic.model_dump_json()


def test_summary_bounds_untrusted_metadata_and_resets_between_invocations() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    diagnostic.observe(
        {
            "type": "result",
            "subtype": "SECRET" * 10000,
            "is_error": "SECRET",
            "duration_ms": 10**100,
            "num_turns": True,
            "errors": ["SECRET"] * 10000,
        }
    )
    assert diagnostic.duration_ms is None
    assert diagnostic.num_turns is None
    assert diagnostic.is_error is None
    assert diagnostic.errors_truncated
    assert len(diagnostic.model_dump_json()) < 2048
    assert "SECRET" not in diagnostic.model_dump_json()
    diagnostic.reset()
    assert diagnostic.model_dump() == ClaudeCodeDiagnostics(store=Store()).model_dump()


def test_error_count_includes_both_singular_and_plural_fields() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store())
    diagnostic.observe(
        {"type": "result", "error": "SECRET", "errors": ["SECRET"] * 100}
    )
    assert diagnostic.error_entries == 101
    assert diagnostic.errors_truncated


def test_record_counts_saturate_instead_of_growing_without_bound() -> None:
    diagnostic = ClaudeCodeDiagnostics(store=Store(), result_records=1_000_000)
    diagnostic.observe({"type": "result"})
    assert diagnostic.result_records == 1_000_000
