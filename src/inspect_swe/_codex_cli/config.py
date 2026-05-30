from typing import Any, Literal, Mapping, cast

from typing_extensions import TypedDict

CodexWebSearch = Literal["live", "cached", "disabled"]


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
        raise ValueError(
            "web_search must be one of 'live', 'cached', or 'disabled'."
        )
    if disallowed_tools and "web_search" in disallowed_tools:
        return "disabled"
    return cast(CodexWebSearch, web_search)


def codex_config_options(web_search: CodexWebSearch, goals: bool) -> dict[str, Any]:
    return {
        "web_search": web_search,
        "features.goals": goals,
    }


def codex_cli_config_overrides(
    web_search: CodexWebSearch, goals: bool
) -> dict[str, str]:
    return {
        "web_search": f'"{web_search}"',
        "features.goals": "true" if goals else "false",
    }
