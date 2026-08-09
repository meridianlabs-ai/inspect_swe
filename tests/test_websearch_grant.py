from inspect_swe._util.websearch import web_search_grant, web_search_tool_disallowed


def test_grant_is_a_providers_dict_when_enabled() -> None:
    grant = web_search_grant(True)
    assert grant == {}
    assert grant is not None


def test_grant_is_none_when_disabled() -> None:
    assert web_search_grant(False) is None


def test_grant_resolves_to_the_bridge_default_provider_set() -> None:
    """An empty WebSearchProviders must mean "the usual internal providers".

    The grant deliberately does not pin a provider list; it relies on an empty
    config resolving to the same set the bridge applies by default. If that
    equivalence ever breaks, granting would silently narrow which providers can
    serve search.
    """
    from inspect_ai.agent._bridge.util import resolve_web_search_providers

    assert resolve_web_search_providers(
        web_search_grant(True)
    ) == resolve_web_search_providers(None)


def test_tool_disallowed_matches_bare_name() -> None:
    assert web_search_tool_disallowed(["WebSearch"], "WebSearch")
    assert not web_search_tool_disallowed(["WebFetch"], "WebSearch")
    assert not web_search_tool_disallowed(None, "WebSearch")
    assert not web_search_tool_disallowed([], "WebSearch")


def test_tool_disallowed_matches_scoped_form() -> None:
    assert web_search_tool_disallowed(["WebFetch(domain:example.com)"], "WebFetch")
    assert not web_search_tool_disallowed(["WebSearchOther"], "WebSearch")
