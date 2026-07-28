from typing import Any, Literal, Mapping, cast

from inspect_ai.model import Model, get_model, model_roles
from pydantic import BaseModel, ConfigDict, Field
from typing_extensions import TypedDict

CodexWebSearch = Literal["live", "cached", "disabled"]


class CodexAutoReview(BaseModel):
    """Options for Codex automated approval review (`auto_review`).

    When enabled, Codex runs with its own sandbox active (`workspace-write`)
    and `approval_policy` set to `on-request`; escalation requests are
    adjudicated by a guardian model rather than a human.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    policy: str | None = Field(default=None)
    """Additional policy instructions inserted into the guardian review prompt."""

    model: str | Model | None = Field(default=None)
    """Model that serves guardian review requests.

    A `str` naming an Inspect model role binds that role; any other string is
    treated as a model name. Defaults to `None`, which serves guardian
    requests with the task's main model.
    """


def resolve_codex_auto_review(
    auto_review: bool | CodexAutoReview,
) -> CodexAutoReview | None:
    if auto_review is False:
        return None
    if auto_review is True:
        return CodexAutoReview()
    return auto_review


class CodexDeprecatedArgs(TypedDict, total=False):
    disallowed_tools: list[Literal["web_search"]] | None


def resolve_codex_deprecated_args(
    deprecated_args: Mapping[str, Any],
) -> list[Literal["web_search"]]:
    unexpected_args = set(deprecated_args) - {"disallowed_tools"}
    if unexpected_args:
        unexpected = ", ".join(sorted(unexpected_args))
        raise TypeError(f"Unexpected keyword argument(s): {unexpected}")

    disallowed_tools = deprecated_args.get("disallowed_tools") or []
    unsupported_tools = set(disallowed_tools) - {"web_search"}
    if unsupported_tools:
        unsupported = ", ".join(sorted(unsupported_tools))
        raise ValueError(f"Unsupported Codex disallowed_tools value(s): {unsupported}")

    return list(disallowed_tools)


def resolve_codex_web_search(
    web_search: str,
    disallowed_tools: list[Literal["web_search"]] | None = None,
) -> CodexWebSearch:
    if web_search not in ("live", "cached", "disabled"):
        raise ValueError("web_search must be one of 'live', 'cached', or 'disabled'.")
    if disallowed_tools and "web_search" in disallowed_tools:
        return "disabled"
    return cast(CodexWebSearch, web_search)


def codex_config_options(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: CodexAutoReview | None = None,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "web_search": web_search,
        "features.goals": goals,
    }
    if auto_review is not None:
        # auto_review only functions with on-request approvals and Codex's own
        # sandbox engaged (mirrors Codex's "Approve for me" preset)
        options["approval_policy"] = "on-request"
        options["sandbox_mode"] = "workspace-write"
        options["approvals_reviewer"] = "auto_review"
        options["features.guardian_approval"] = True
        if auto_review.policy is not None:
            options["auto_review"] = {"policy": auto_review.policy}
    return options


def codex_cli_config_overrides(
    web_search: CodexWebSearch,
    goals: bool,
    auto_review: CodexAutoReview | None = None,
) -> dict[str, str]:
    overrides = {
        "web_search": f'"{web_search}"',
        "features.goals": "true" if goals else "false",
    }
    if auto_review is not None:
        overrides["approval_policy"] = '"on-request"'
        overrides["sandbox_mode"] = '"workspace-write"'
        overrides["approvals_reviewer"] = '"auto_review"'
        overrides["features.guardian_approval"] = "true"
        # auto_review.policy is emitted only via config.toml: -c values are
        # parsed as TOML and multiline policies don't survive shell quoting
    return overrides


GUARDIAN_MODEL_SLUG = "codex-auto-review"
"""Model slug Codex uses for guardian (auto_review) requests."""

CODEX_AUTO_REVIEW_MIN_VERSION = "0.137.0"
"""First Codex CLI release where `codex exec` preserves auto_review approvals."""


def resolve_codex_auto_review_model_aliases(
    auto_review: CodexAutoReview | None,
    model_aliases: dict[str, str | Model] | None,
) -> dict[str, str | Model] | None:
    """Bind the guardian model slug to the configured auto_review model.

    Call within a running task (role resolution reads task context). A `str`
    naming a defined Inspect model role binds via `get_model(role=...)`;
    other values pass through for the bridge to resolve.
    """
    if auto_review is None or auto_review.model is None:
        return model_aliases
    guardian: str | Model = auto_review.model
    if isinstance(guardian, str) and guardian in model_roles():
        guardian = get_model(role=guardian)
    return {**(model_aliases or {}), GUARDIAN_MODEL_SLUG: guardian}


def check_codex_auto_review_version(version: str | None) -> None:
    """Raise if the installed Codex CLI can't run auto_review headlessly."""
    if version is None:
        return
    installed = tuple(int(part) for part in version.split(".")[:3])
    required = tuple(int(part) for part in CODEX_AUTO_REVIEW_MIN_VERSION.split("."))
    if installed < required:
        raise RuntimeError(
            f"auto_review requires Codex CLI >= {CODEX_AUTO_REVIEW_MIN_VERSION} "
            f"(found {version}). Pass version='latest' (or an explicit newer "
            "version) to codex_cli()."
        )
