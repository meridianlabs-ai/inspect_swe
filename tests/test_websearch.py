from inspect_swe._util.websearch import web_search_tool_disallowed


def test_tool_disallowed_matches_bare_name() -> None:
    assert web_search_tool_disallowed(["WebSearch"], "WebSearch")
    assert not web_search_tool_disallowed(["WebFetch"], "WebSearch")
    assert not web_search_tool_disallowed(None, "WebSearch")
    assert not web_search_tool_disallowed([], "WebSearch")


def test_tool_disallowed_matches_scoped_form() -> None:
    assert web_search_tool_disallowed(["WebFetch(domain:example.com)"], "WebFetch")
    assert not web_search_tool_disallowed(["WebSearchOther"], "WebSearch")
