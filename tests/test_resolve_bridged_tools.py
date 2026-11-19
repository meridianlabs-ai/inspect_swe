"""Unit tests for resolving plain Tools in bridged_tools (no Docker)."""

from typing import cast

import pytest
from inspect_ai.agent import BridgedToolsSpec
from inspect_ai.tool import Tool, tool
from inspect_swe._util.bridged_tools import (
    DEFAULT_BRIDGED_TOOLS_NAME,
    resolve_bridged_tools,
)


@tool
def lookup() -> Tool:
    async def execute(key: str) -> str:
        """Look up a value by key.

        Args:
            key: The key to look up.
        """
        return key

    return execute


@tool
def search() -> Tool:
    async def execute(query: str) -> str:
        """Search for a query.

        Args:
            query: The search query.
        """
        return query

    return execute


def test_none_and_empty_resolve_to_none() -> None:
    assert resolve_bridged_tools(None) is None
    assert resolve_bridged_tools([]) is None


def test_specs_pass_through_unchanged() -> None:
    spec = BridgedToolsSpec(name="host", tools=[lookup()], require_proposal=False)
    assert resolve_bridged_tools([spec]) == [spec]


def test_plain_tools_are_grouped_into_one_default_spec() -> None:
    lookup_tool, search_tool = lookup(), search()
    specs = resolve_bridged_tools([lookup_tool, search_tool])
    assert specs is not None and len(specs) == 1
    assert specs[0].name == DEFAULT_BRIDGED_TOOLS_NAME
    assert list(specs[0].tools) == [lookup_tool, search_tool]
    assert specs[0].require_proposal is True


def test_plain_tools_mix_with_specs() -> None:
    spec = BridgedToolsSpec(name="host", tools=[search()])
    lookup_tool = lookup()
    specs = resolve_bridged_tools([lookup_tool, spec])
    assert specs is not None
    assert [s.name for s in specs] == ["host", DEFAULT_BRIDGED_TOOLS_NAME]
    assert specs[0] is spec
    assert list(specs[1].tools) == [lookup_tool]


def test_spec_may_use_default_name_without_plain_tools() -> None:
    spec = BridgedToolsSpec(name=DEFAULT_BRIDGED_TOOLS_NAME, tools=[search()])
    assert resolve_bridged_tools([spec]) == [spec]


def test_plain_tools_conflict_with_spec_using_default_name() -> None:
    spec = BridgedToolsSpec(name=DEFAULT_BRIDGED_TOOLS_NAME, tools=[search()])
    with pytest.raises(ValueError, match="already uses"):
        resolve_bridged_tools([spec, lookup()])


def test_plain_tools_with_duplicate_names_are_rejected() -> None:
    with pytest.raises(ValueError, match="share a name: lookup"):
        resolve_bridged_tools([lookup(), lookup()])


def test_non_tool_items_are_rejected() -> None:
    with pytest.raises(TypeError, match="not str"):
        resolve_bridged_tools(cast(list[Tool], ["lookup"]))
