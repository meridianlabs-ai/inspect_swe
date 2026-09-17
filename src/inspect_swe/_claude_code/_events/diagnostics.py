"""Bounded structural diagnostics; never retain CLI result or error prose."""

from inspect_ai.util import Store, StoreModel
from pydantic import Field

MAX_COUNT = 1_000_000
MAX_METADATA_VALUE = 1_000_000_000
MAX_ERRORS_PER_RECORD = 16
RESULT_SUBTYPES = frozenset(
    {
        "success",
        "error_during_execution",
        "error_max_turns",
        "error_max_budget_usd",
        "error_max_structured_output_retries",
    }
)
ERROR_CATEGORIES = frozenset(
    {
        "api_error",
        "authentication_error",
        "billing_error",
        "invalid_request_error",
        "not_found_error",
        "overloaded_error",
        "permission_error",
        "rate_limit_error",
        "request_too_large",
    }
)


class ClaudeCodeDiagnostics(StoreModel):
    """Latest CLI invocation, persisted even when debug collection is disabled."""

    result_available: bool = False
    result_records: int = 0
    error_records: int = 0
    malformed_records: int = 0
    result_subtype: str | None = None
    is_error: bool | None = None
    duration_ms: int | None = None
    num_turns: int | None = None
    error_entries: int = 0
    error_categories: list[str] = Field(default_factory=list)
    unknown_error_present: bool = False
    errors_truncated: bool = False

    def reset(self) -> None:
        for name, value in ClaudeCodeDiagnostics(store=Store()).model_dump().items():
            setattr(self, name, value)

    def malformed(self) -> None:
        self.malformed_records = min(self.malformed_records + 1, MAX_COUNT)

    def observe(self, raw: object) -> None:
        if not isinstance(raw, dict):
            self.malformed()
            return
        record_type = raw.get("type")
        if record_type == "result":
            self.result_available = True
            self.result_records = min(self.result_records + 1, MAX_COUNT)
            subtype = raw.get("subtype")
            self.result_subtype = (
                subtype
                if isinstance(subtype, str) and subtype in RESULT_SUBTYPES
                else "unknown"
            )
            value = raw.get("is_error")
            self.is_error = value if type(value) is bool else None
            self.duration_ms = _bounded_integer(raw.get("duration_ms"))
            self.num_turns = _bounded_integer(raw.get("num_turns"))
        elif record_type == "error":
            self.error_records = min(self.error_records + 1, MAX_COUNT)
        else:
            return

        errors = raw.get("errors", [])
        if not isinstance(errors, list):
            self.malformed()
            errors = [errors]
        error_count = len(errors)
        if "error" in raw:
            error_count += 1
            errors = [raw["error"], *errors[:MAX_ERRORS_PER_RECORD]]
        elif record_type == "error" and not errors:
            errors = [raw]
            error_count = 1
        self.error_entries = min(self.error_entries + error_count, MAX_COUNT)
        if error_count > MAX_ERRORS_PER_RECORD:
            self.errors_truncated = True
        for error in errors[:MAX_ERRORS_PER_RECORD]:
            if not isinstance(error, dict):
                self.unknown_error_present = True
                continue
            recognized = False
            for field in ("type", "code"):
                value = error.get(field)
                if isinstance(value, str) and value in ERROR_CATEGORIES:
                    recognized = True
                    if value not in self.error_categories:
                        self.error_categories.append(value)
                elif value is not None:
                    self.unknown_error_present = True
            if not recognized:
                self.unknown_error_present = True


def _bounded_integer(value: object) -> int | None:
    return value if type(value) is int and 0 <= value <= MAX_METADATA_VALUE else None
